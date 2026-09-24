"""Archival layer queries: search the book's text, read a passage in context.

Written to become agent tools in step 4. Search filters first (book, chapter range,
character) and then ranks two ways, by embedding similarity and by keywords (jieba
tokens), merging the two rankings with reciprocal rank fusion: names and exact wording
are found by keywords, paraphrases by embeddings.

Postgres ranks in SQL (pgvector, full-text search). SQLite, used by unit tests, ranks
the same way in Python; it checks filters and fusion, not search quality.
"""

import asyncio
import math
import re
import uuid
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import Float, bindparam, func, literal_column, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.archival.index import Archival, character_names
from webfic.archival.passages import sentence_ends
from webfic.db.models import EMBEDDING_DIM, Book, Chapter, CharacterAlias, PassageRow
from webfic.memory.recall import ChapterRange, find_character
from webfic.services.errors import NotFound

Mode = Literal["hybrid", "vector", "keyword"]

CANDIDATES = 50  # taken from each ranking before fusion
RRF_K = 60  # the usual reciprocal-rank-fusion constant


class PassageHit(BaseModel):
    chapter_number: int
    char_start: int  # in the chapter's text
    char_end: int
    text: str
    score: float  # fused score; only comparable within one search
    matched_by: list[str]  # "vector", "keyword" or both


class PassageText(BaseModel):
    chapter_number: int
    chapter_title: str
    char_start: int
    char_end: int
    text: str
    # Where the requested span sits inside `text`.
    focus_start: int
    focus_end: int


async def _check_book(session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID) -> None:
    owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id)
    if await session.scalar(owned) is None:
        raise NotFound(f"book {book_id}")


def _tsquery(tokens: list[str]) -> str | None:
    """An OR query over the tokens, safe for to_tsquery (only word characters kept)."""
    words = {re.sub(r"[^\w一-鿿]", "", t).lower() for t in tokens}
    words.discard("")
    return " | ".join(f"'{w}'" for w in sorted(words)) or None


async def _rank_sql(
    session: AsyncSession, filters: list, vector: list[float], tokens: list[str], mode: Mode
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    by_vector: list[uuid.UUID] = []
    by_keyword: list[uuid.UUID] = []
    if mode in ("hybrid", "vector"):
        from pgvector.sqlalchemy import Vector

        distance = PassageRow.embedding.op("<=>", return_type=Float)(
            bindparam("query_vector", vector, type_=Vector(EMBEDDING_DIM))
        )
        by_vector = list(
            await session.scalars(
                select(PassageRow.id).where(*filters).order_by(distance).limit(CANDIDATES)
            )
        )
    query = _tsquery(tokens)
    if mode in ("hybrid", "keyword") and query:
        simple = literal_column("'simple'::regconfig")
        # The same expression as the GIN index ix_passages_tokens.
        document = func.to_tsvector(simple, PassageRow.tokens)
        wanted = func.to_tsquery(simple, query)
        rank = func.ts_rank_cd(document, wanted)
        by_keyword = list(
            await session.scalars(
                select(PassageRow.id)
                .where(*filters, document.op("@@")(wanted))
                .order_by(rank.desc(), PassageRow.chapter_number, PassageRow.char_start)
                .limit(CANDIDATES)
            )
        )
    return by_vector, by_keyword


async def _rank_python(
    session: AsyncSession, filters: list, vector: list[float], tokens: list[str], mode: Mode
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    rows = (
        await session.execute(
            select(PassageRow.id, PassageRow.embedding, PassageRow.tokens)
            .where(*filters)
            .order_by(PassageRow.chapter_number, PassageRow.char_start)
        )
    ).all()
    by_vector: list[uuid.UUID] = []
    by_keyword: list[uuid.UUID] = []
    if mode in ("hybrid", "vector"):
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0

        def cosine(embedding: list[float]) -> float:
            dot = sum(a * b for a, b in zip(vector, embedding, strict=True))
            return dot / (norm * (math.sqrt(sum(b * b for b in embedding)) or 1.0))

        by_vector = [r.id for r in sorted(rows, key=lambda r: -cosine(r.embedding))][:CANDIDATES]
    wanted = {t.lower() for t in tokens}
    if mode in ("hybrid", "keyword") and wanted:
        scored = [(len(wanted & set(r.tokens.lower().split())), r.id) for r in rows]
        by_keyword = [i for s, i in sorted(scored, key=lambda x: -x[0]) if s > 0][:CANDIDATES]
    return by_vector, by_keyword


async def search_text(
    session: AsyncSession,
    archival: Archival,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    query: str,
    k: int = 5,
    character: str | None = None,
    chapters: ChapterRange | None = None,
    mode: Mode = "hybrid",
) -> list[PassageHit]:
    """Passages of the book that answer `query`, best first; optionally only those that
    mention a character (by any of their names) or lie in a chapter range."""
    await _check_book(session, user_id, book_id)
    filters = [PassageRow.user_id == user_id, PassageRow.book_id == book_id]
    if chapters is not None:
        filters += [
            PassageRow.chapter_number >= chapters[0],
            PassageRow.chapter_number <= chapters[1],
        ]
    if character is not None:
        found = await find_character(session, user_id=user_id, book_id=book_id, name=character)
        aliases = await session.scalars(
            select(CharacterAlias.alias).where(CharacterAlias.character_id == found.id)
        )
        names = {found.canonical_name, *aliases}
        filters.append(or_(*(PassageRow.text.contains(n) for n in names)))

    archival.tokenizer.add_names(await character_names(session, user_id, book_id))
    tokens = archival.tokenizer.tokens(query)
    vector = await asyncio.to_thread(archival.embedder.embed_query, query)
    rank = _rank_sql if session.get_bind().dialect.name == "postgresql" else _rank_python
    by_vector, by_keyword = await rank(session, filters, vector, tokens, mode)

    fused: dict[uuid.UUID, float] = {}
    matched: dict[uuid.UUID, list[str]] = {}
    for label, ranking in (("vector", by_vector), ("keyword", by_keyword)):
        for place, passage_id in enumerate(ranking, start=1):
            fused[passage_id] = fused.get(passage_id, 0.0) + 1 / (RRF_K + place)
            matched.setdefault(passage_id, []).append(label)
    best = sorted(fused, key=lambda i: -fused[i])[:k]
    rows = {
        r.id: r for r in await session.scalars(select(PassageRow).where(PassageRow.id.in_(best)))
    }
    return [
        PassageHit(
            chapter_number=rows[i].chapter_number,
            char_start=rows[i].char_start,
            char_end=rows[i].char_end,
            text=rows[i].text,
            score=round(fused[i], 6),
            matched_by=matched[i],
        )
        for i in best
    ]


async def read_passage(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter: int,
    start: int,
    end: int,
    context: int = 200,
) -> PassageText:
    """The text around [start, end) of a chapter, extended by up to `context` characters
    on each side and trimmed to whole sentences."""
    await _check_book(session, user_id, book_id)
    row = await session.scalar(
        select(Chapter).where(
            Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number == chapter
        )
    )
    if row is None:
        raise NotFound(f"第 {chapter} 章")
    text = row.content
    start, end = max(0, min(start, len(text))), max(0, min(end, len(text)))
    start, end = min(start, end), max(start, end)
    ends = sentence_ends(text)
    # Begin right after a sentence end within the context window, end at one.
    lead = [e for e in [0, *ends] if start - context <= e <= start]  # 0: chapter start
    begin = lead[0] if lead else max(0, start - context)
    tail = [e for e in ends if end <= e <= end + context]
    stop = tail[-1] if tail else min(len(text), end + context)
    while begin < start and text[begin].isspace():
        begin += 1
    return PassageText(
        chapter_number=chapter,
        chapter_title=row.title,
        char_start=begin,
        char_end=stop,
        text=text[begin:stop],
        focus_start=start - begin,
        focus_end=end - begin,
    )
