"""Build the passage index of a book (Archival layer).

Passages depend only on their chapter's text (and the passage settings and embedding
model), so a chapter is re-indexed only when that changes; recomputing chapters for other
reasons, or renumbering them, keeps their passages.
"""

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass

from pydantic import BaseModel
from sqlalchemy import delete, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.archival.embedding import Embedder, FastEmbedder
from webfic.archival.passages import split_passages
from webfic.archival.tokenize import Tokenizer
from webfic.config import Settings
from webfic.db.models import Book, Chapter, Character, CharacterAlias, PassageRow
from webfic.services.errors import NotFound


@dataclass
class Archival:
    """What building and searching the passage index needs."""

    embedder: Embedder
    tokenizer: Tokenizer
    embed_model: str
    passage_size: int = 500
    passage_overlap: int = 100

    def source_key(self, chapter: Chapter) -> str:
        key = (
            f"{chapter.content_hash}|{self.passage_size}|{self.passage_overlap}|{self.embed_model}"
        )
        return hashlib.sha256(key.encode()).hexdigest()


def platform_archival(settings: Settings) -> Archival:
    return Archival(
        embedder=FastEmbedder(settings.embed_model, settings.model_cache_dir),
        tokenizer=Tokenizer(settings.model_cache_dir),
        embed_model=settings.embed_model,
        passage_size=settings.passage_size,
        passage_overlap=settings.passage_overlap,
    )


async def character_names(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> list[str]:
    """Every canonical name and alias of the book's characters."""
    names = await session.scalars(
        select(Character.canonical_name).where(
            Character.user_id == user_id, Character.book_id == book_id
        )
    )
    aliases = await session.scalars(
        select(CharacterAlias.alias).where(
            CharacterAlias.user_id == user_id, CharacterAlias.book_id == book_id
        )
    )
    return [*names, *aliases]


async def index_chapter(
    session: AsyncSession,
    archival: Archival,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter: Chapter,
    names: list[str],
) -> int:
    """(Re)build one chapter's passages if its text or the settings changed; returns how
    many passages were built (0 when the existing ones still fit)."""
    key = archival.source_key(chapter)
    existing = set(
        await session.scalars(
            select(PassageRow.source_key).where(
                PassageRow.user_id == user_id, PassageRow.chapter_id == chapter.id
            )
        )
    )
    if existing == {key}:
        await session.execute(
            update(PassageRow)
            .where(PassageRow.user_id == user_id, PassageRow.chapter_id == chapter.id)
            .values(chapter_number=chapter.number)
        )
        return 0

    await session.execute(
        delete(PassageRow).where(PassageRow.user_id == user_id, PassageRow.chapter_id == chapter.id)
    )
    spans = split_passages(chapter.content, archival.passage_size, archival.passage_overlap)
    texts = [chapter.content[start:end] for start, end in spans]
    archival.tokenizer.add_names(names)
    tokens = [" ".join(archival.tokenizer.tokens(t)) for t in texts]
    # Embedding is CPU work; keep the event loop responsive.
    vectors = await asyncio.to_thread(archival.embedder.embed_passages, texts)
    session.add_all(
        PassageRow(
            user_id=user_id, book_id=book_id, chapter_id=chapter.id,
            chapter_number=chapter.number, source_key=key,
            char_start=start, char_end=end, text=text, tokens=words, embedding=vector,
        )
        for (start, end), text, words, vector in zip(spans, texts, tokens, vectors, strict=True)
    )  # fmt: skip
    return len(spans)


async def retokenize_for_names(
    session: AsyncSession,
    archival: Archival,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    new_names: list[str],
    names: list[str],
) -> int:
    """Re-cut the keywords of the book's passages that contain a newly known name.

    A name is added to the dictionary only once it is a character's name or alias, so
    passages indexed earlier cut it into pieces ("宇智波鼬" -> "宇智波 / 鼬") and a keyword
    search for the name would miss them. Only the tokens change; embeddings do not depend
    on the dictionary. Returns how many passages were updated."""
    wanted = sorted({n.strip() for n in new_names if len(n.strip()) >= 2})
    if not wanted:
        return 0
    archival.tokenizer.add_names(names)
    rows = (
        await session.execute(
            select(PassageRow.id, PassageRow.text, PassageRow.tokens).where(
                PassageRow.user_id == user_id,
                PassageRow.book_id == book_id,
                or_(*(PassageRow.text.contains(n, autoescape=True) for n in wanted)),
            )
        )
    ).all()
    updated = 0
    for passage_id, text, old in rows:
        tokens = " ".join(archival.tokenizer.tokens(text))
        if tokens != old:
            await session.execute(
                update(PassageRow)
                .where(PassageRow.user_id == user_id, PassageRow.id == passage_id)
                .values(tokens=tokens)
            )
            updated += 1
    return updated


class ReindexResult(BaseModel):
    chapters: int
    rebuilt: int  # chapters whose passages were (re)built
    passages: int  # passages built
    retokenized: int = 0  # kept passages whose keywords were re-cut for today's names
    seconds: float


async def reindex_book(
    factory: async_sessionmaker[AsyncSession],
    archival: Archival,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
) -> ReindexResult:
    """Make sure every chapter of the book has up-to-date passages (for books imported
    before passages existed, or after changing the passage settings), their keywords cut
    with all of today's character names."""
    started = time.monotonic()
    async with factory() as session:
        owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id)
        if await session.scalar(owned) is None:
            raise NotFound(f"book {book_id}")
        chapter_ids = (
            await session.scalars(
                select(Chapter.id)
                .where(Chapter.user_id == user_id, Chapter.book_id == book_id)
                .order_by(Chapter.number)
            )
        ).all()
        names = await character_names(session, user_id, book_id)
    rebuilt = passages = 0
    for chapter_id in chapter_ids:  # one transaction per chapter, like extraction
        async with factory() as session:
            chapter = await session.get_one(Chapter, chapter_id)
            built = await index_chapter(
                session, archival, user_id=user_id, book_id=book_id, chapter=chapter, names=names
            )
            await session.commit()
        rebuilt += built > 0
        passages += built
    async with factory() as session:
        retokenized = await retokenize_for_names(
            session, archival, user_id=user_id, book_id=book_id, new_names=names, names=names
        )
        await session.commit()
    return ReindexResult(
        chapters=len(chapter_ids),
        rebuilt=rebuilt,
        passages=passages,
        retokenized=retokenized,
        seconds=round(time.monotonic() - started, 1),
    )
