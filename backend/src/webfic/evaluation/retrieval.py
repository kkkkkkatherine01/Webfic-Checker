"""Retrieval evaluation: corpora, their passage indexes, questions and scores.

Each corpus text becomes a book of a dedicated evaluation user in Postgres, indexed with
the real embedding model. Books are named after the text's hash and the passage settings,
so a changed text or setting gets its own book, and preparing again is a no-op (chapters
whose passages exist are skipped; an interrupted run resumes).

Corpora:
- golden: the test stories' `retrieval` questions (story05, the held-out set, has none);
  the stories are extracted too (served from the eval LLM cache), because questions may
  filter by character.
- detectiveqa: translated detective novels with human-annotated questions; the relevant
  passages are the annotated clue paragraphs.
"""

import hashlib
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.archival.index import Archival, reindex_book
from webfic.config import Settings
from webfic.db.models import Book
from webfic.evaluation.golden import Span, Story
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import LLMClient
from webfic.memory.archival import PassageHit, search_text
from webfic.services import imports
from webfic.services.errors import NotFound

RETRIEVAL_USER = uuid.UUID("00000000-0000-0000-0000-00000000e7a2")

_PARAGRAPH_NUMBER = re.compile(r"^\[(\d+)\]")


# --- corpora and indexes ----------------------------------------------------------------


@dataclass(frozen=True)
class CorpusText:
    name: str  # e.g. "detectiveqa/117"
    text: str


def detectiveqa_text(path: Path) -> str:
    """A DetectiveQA novel ("[12]段落" per line) as plain text, paragraph numbers removed."""
    lines = path.read_text("utf-8").splitlines()
    return "\n".join(_PARAGRAPH_NUMBER.sub("", line) for line in lines)


def detectiveqa_corpus(folder: Path) -> list[CorpusText]:
    """The novels that have human annotations."""
    annotated = {p.stem for p in (folder / "human_anno").glob("*.json")}
    texts = []
    for path in sorted((folder / "novel_data_zh").glob("*.txt")):
        novel_id = path.name.split("-", 1)[0]
        if novel_id in annotated:
            texts.append(CorpusText(f"detectiveqa/{novel_id}", detectiveqa_text(path)))
    return texts


def golden_corpus(stories: list[Story]) -> list[CorpusText]:
    return [CorpusText(f"golden/{s.id}", s.text) for s in stories if s.golden.retrieval]


def book_title(corpus: CorpusText, archival: Archival) -> str:
    digest = hashlib.sha256(corpus.text.encode()).hexdigest()[:10]
    return f"{corpus.name}#{digest}@{archival.passage_size}-{archival.passage_overlap}"


async def prepare(
    factory: async_sessionmaker[AsyncSession],
    archival: Archival,
    corpus: list[CorpusText],
    on_progress: Callable[[str], None] = print,
    extract: Callable[[CorpusText], tuple[LLMClient, Settings]] | None = None,
) -> dict[str, uuid.UUID]:
    """Import and index each text; returns name -> book id. With `extract`, which gives
    the LLM client and settings for a text, chapters are also extracted (and indexed as
    they go); otherwise they are only split and indexed."""
    books: dict[str, uuid.UUID] = {}
    for n, item in enumerate(corpus, start=1):
        title = book_title(item, archival)
        async with factory() as session:
            book_id = await session.scalar(
                select(Book.id).where(Book.user_id == RETRIEVAL_USER, Book.title == title)
            )
            if book_id is None:
                job = await imports.create_import_job(
                    session, user_id=RETRIEVAL_USER, title=title, text=item.text
                )
                book_id = job.book_id
        started = time.monotonic()
        if extract is not None:
            llm, settings = extract(item)
            await imports.run_import_job(
                factory, llm, settings, user_id=RETRIEVAL_USER, book_id=book_id, archival=archival
            )
        result = await reindex_book(factory, archival, user_id=RETRIEVAL_USER, book_id=book_id)
        on_progress(
            f"[{n}/{len(corpus)}] {item.name}: {result.chapters} 章，新建 {result.passages} 段，"
            f"{time.monotonic() - started:.0f} 秒"
        )
        books[item.name] = book_id
    return books


# --- questions --------------------------------------------------------------------------


@dataclass(frozen=True)
class Question:
    corpus: str  # "golden" / "detectiveqa"
    book: str  # corpus item name, e.g. "golden/story01"
    query: str
    relevant: tuple[Span, ...]  # passages that answer it (chapter coordinates)
    character: str | None = None


def golden_questions(stories: list[Story]) -> list[Question]:
    return [
        Question(
            corpus="golden",
            book=f"golden/{story.id}",
            query=q.query,
            character=q.character,
            relevant=tuple(story.span(e.chapter, e.quote) for e in q.expect),
        )
        for story in stories
        for q in story.golden.retrieval
    ]


_SHORT_PARAGRAPH = 8  # shorter ones are only looked for in the current chapter


def paragraph_spans(path: Path) -> dict[int, Span]:
    """Where each numbered paragraph of a DetectiveQA novel lies in the book as the
    import splits it (chapter coordinates). Paragraphs are found by their text, in
    reading order; headings, which are not chapter text, are not found."""
    chapters = split_chapters(detectiveqa_text(path)).chapters
    spans: dict[int, Span] = {}
    current, cursor = 0, 0
    for line in path.read_text("utf-8").splitlines():
        match = _PARAGRAPH_NUMBER.match(line)
        text = _PARAGRAPH_NUMBER.sub("", line).strip()
        if not match or not text:
            continue
        last = len(chapters) if len(text) >= _SHORT_PARAGRAPH else current + 1
        for i in range(current, min(last, len(chapters))):
            start = chapters[i].content.find(text, cursor if i == current else 0)
            if start != -1:
                spans[int(match.group(1))] = Span(chapters[i].number, start, start + len(text))
                current, cursor = i, start + len(text)
                break
    return spans


_MIN_STATEMENT = 8  # shorter reasoning lines are not usable queries


def detectiveqa_questions(folder: Path) -> tuple[list[Question], list[Question], int]:
    """Two query sets over the same relevant passages (the clue paragraphs and the answer
    paragraph; clues marked -1, reasoned rather than stated, and paragraphs outside chapter
    text are left out):

    - "detectiveqa": the annotated questions ("文中案件的凶手是谁？"). Many need reasoning
      over the whole book, which one search cannot do: a baseline for step 4's agent.
    - "detectiveqa-clues": the annotators' one-line clue statements ("马歇尔太太让波洛
      不要告诉别人她去哪儿"), i.e. concrete facts, like what an agent searches for. Some
      statements copy the text nearly word for word, so this set is on the easy side.

    Returns both sets and how many annotated positions were left out."""
    questions: list[Question] = []
    clue_queries: list[Question] = []
    skipped = 0
    for path in sorted((folder / "novel_data_zh").glob("*.txt")):
        novel_id = path.name.split("-", 1)[0]
        annotation = folder / "human_anno" / f"{novel_id}.json"
        if not annotation.exists():
            continue
        data = json.loads(annotation.read_text("utf-8"))
        data = data[0] if isinstance(data, list) else data
        paragraphs = paragraph_spans(path)
        for item in data["questions"]:
            positions = [int(p) for p in item.get("clue_position", [])]
            positions.append(int(item.get("answer_position", -1)))
            spans: set[Span] = set()
            for p in positions:
                span = paragraphs.get(p) if p >= 0 else None
                if span is None:
                    skipped += 1
                else:
                    spans.add(span)
            if not spans:
                continue
            relevant = tuple(sorted(spans, key=lambda s: (s.chapter, s.start)))
            book = f"detectiveqa/{novel_id}"
            questions.append(
                Question(corpus="detectiveqa", book=book, query=item["question"], relevant=relevant)
            )
            # The last reasoning line is the reasoning process, not a clue.
            clue_queries += [
                Question(corpus="detectiveqa-clues", book=book, query=line, relevant=relevant)
                for line in item.get("reasoning", [])[:-1]
                if len(line) >= _MIN_STATEMENT
            ]
    return questions, clue_queries, skipped


# --- scoring ----------------------------------------------------------------------------

K = 10


class Scores(BaseModel):
    corpus: str
    passage_size: int
    passage_overlap: int
    mode: str
    questions: int
    hit_at_5: float  # at least one relevant passage in the top 5
    hit_at_10: float
    coverage_at_10: float  # share of the relevant passages found in the top 10
    mrr: float  # 1 / rank of the first relevant passage (0 beyond the top 10)
    missed: list[str] = []  # queries with nothing relevant in the top 10


def _found(hit: PassageHit, relevant: tuple[Span, ...]) -> bool:
    return any(r.overlaps(hit.chapter_number, hit.char_start, hit.char_end) for r in relevant)


def score(hits: list[PassageHit], relevant: tuple[Span, ...]) -> tuple[int | None, float]:
    """(rank of the first relevant hit, share of relevant spans covered) in the top K."""
    top = hits[:K]
    first = next((rank for rank, h in enumerate(top, start=1) if _found(h, relevant)), None)
    covered = sum(any(_found(h, (r,)) for h in top) for r in relevant)
    return first, covered / len(relevant)


Outcome = tuple[Question, int | None, float]


def aggregate(corpus: str, archival: Archival, mode: str, outcomes: list[Outcome]) -> Scores:
    n = len(outcomes) or 1
    return Scores(
        corpus=corpus,
        passage_size=archival.passage_size,
        passage_overlap=archival.passage_overlap,
        mode=mode,
        questions=len(outcomes),
        hit_at_5=round(sum(f is not None and f <= 5 for _, f, _ in outcomes) / n, 4),
        hit_at_10=round(sum(f is not None for _, f, _ in outcomes) / n, 4),
        coverage_at_10=round(sum(c for _, _, c in outcomes) / n, 4),
        mrr=round(sum(1 / f for _, f, _ in outcomes if f) / n, 4),
        missed=[f"{q.book} {q.query}" for q, f, _ in outcomes if f is None],
    )


async def evaluate(
    factory: async_sessionmaker[AsyncSession],
    archival: Archival,
    books: dict[str, uuid.UUID],
    questions: list[Question],
    mode: str,
    user_id: uuid.UUID = RETRIEVAL_USER,
) -> list[Outcome]:
    outcomes: list[Outcome] = []
    async with factory() as session:
        for q in questions:
            try:
                hits = await search_text(
                    session, archival, user_id=user_id, book_id=books[q.book],
                    query=q.query, k=K, character=q.character, mode=mode,  # type: ignore[arg-type]
                )  # fmt: skip
            except NotFound:  # the filter names a character extraction did not find
                hits = []
            outcomes.append((q, *score(hits, q.relevant)))
    return outcomes


class RetrievalReport(BaseModel):
    label: str
    created_at: datetime
    embed_model: str
    scores: list[Scores]
    skipped_clues: int = 0  # DetectiveQA positions with no passage to find
