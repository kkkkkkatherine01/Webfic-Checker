"""Import a book: `create_import_job` splits and stores chapters; `run_import_job` extracts
facts chapter by chapter. The CLI calls both in a row; before launch a worker runs the
second one."""

import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.config import Settings
from webfic.db.models import (
    Book,
    Chapter,
    Character,
    CharacterAlias,
    ElapsedTimeFactRow,
    FactRow,
)
from webfic.extraction.extractor import extract_chapter, load_prompt
from webfic.extraction.resolver import CharacterIndex, KnownCharacter
from webfic.facts.registry import AGE
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import LLMClient, LLMError, ProviderError, ProviderErrorKind
from webfic.services.errors import NotFound

log = logging.getLogger(__name__)


class ChapterSummary(BaseModel):
    number: int
    title: str
    char_count: int


class ImportJobCreated(BaseModel):
    book_id: uuid.UUID
    chapters: list[ChapterSummary]
    total_chars: int
    warnings: list[str]


class ProgressEvent(BaseModel):
    chapter_number: int
    chapter_title: str
    done: int
    total: int
    status: str  # "extracted" | "failed"
    error: str | None = None


class ImportJobResult(BaseModel):
    extracted: int
    failed: int
    failures: dict[str, int] = {}  # reason kind -> chapters, e.g. {"content_filter": 2}
    dropped_statements: int
    llm_calls: int
    cache_hits: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


async def create_import_job(
    session: AsyncSession, *, user_id: uuid.UUID, title: str, text: str
) -> ImportJobCreated:
    split = split_chapters(text)
    book = Book(user_id=user_id, title=title)
    session.add(book)
    await session.flush()

    for raw in split.chapters:
        session.add(
            Chapter(
                user_id=user_id,
                book_id=book.id,
                number=raw.number,
                title=raw.title,
                content=raw.content,
                char_count=len(raw.content),
                content_hash=hashlib.sha256(raw.content.encode()).hexdigest(),
                status="pending",
            )
        )
    await session.commit()

    summaries = [
        ChapterSummary(number=c.number, title=c.title, char_count=len(c.content))
        for c in split.chapters
    ]
    return ImportJobCreated(
        book_id=book.id,
        chapters=summaries,
        total_chars=sum(c.char_count for c in summaries),
        warnings=split.warnings,
    )


async def _load_character_index(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> CharacterIndex:
    characters = (
        await session.scalars(
            select(Character).where(Character.user_id == user_id, Character.book_id == book_id)
        )
    ).all()
    aliases = (
        await session.scalars(
            select(CharacterAlias).where(
                CharacterAlias.user_id == user_id, CharacterAlias.book_id == book_id
            )
        )
    ).all()
    by_character: dict[uuid.UUID, list[str]] = {}
    for a in aliases:
        by_character.setdefault(a.character_id, []).append(a.alias)
    return CharacterIndex(
        [KnownCharacter(c.id, c.canonical_name, by_character.get(c.id, [])) for c in characters]
    )


async def _emit(on_progress: ProgressCallback | None, event: ProgressEvent) -> None:
    if on_progress is None:
        return
    maybe = on_progress(event)
    if maybe is not None:
        await maybe


async def run_import_job(
    session_factory: async_sessionmaker[AsyncSession],
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    on_progress: ProgressCallback | None = None,
) -> ImportJobResult:
    """Extract every chapter that is not yet `extracted`, in narrative order. Each chapter
    commits on its own, so an interrupted run can simply be started again."""
    async with session_factory() as session:
        book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
        if book is None:
            raise NotFound(f"book {book_id}")
        todo = (
            await session.execute(
                select(Chapter.id, Chapter.number, Chapter.title)
                .where(
                    Chapter.user_id == user_id,
                    Chapter.book_id == book_id,
                    Chapter.status != "extracted",
                )
                .order_by(Chapter.number)
            )
        ).all()

    system_prompt = load_prompt()
    result = ImportJobResult(
        extracted=0, failed=0, dropped_statements=0, llm_calls=0, cache_hits=0,
        input_tokens=0, output_tokens=0, cost_usd=Decimal(0),
    )  # fmt: skip

    for done, (chapter_id, number, title) in enumerate(todo, start=1):
        async with session_factory() as session:
            chapter = await session.get_one(Chapter, chapter_id)
            index = await _load_character_index(session, user_id, book_id)
            try:
                extraction = await extract_chapter(
                    llm,
                    chapter_number=number,
                    text=chapter.content,
                    known_characters=index.for_prompt(),
                    system_prompt=system_prompt,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
            except LLMError as exc:
                if isinstance(exc, ProviderError) and exc.kind == ProviderErrorKind.AUTH:
                    raise  # a bad or unfunded key fails every chapter; stop here
                kind = exc.kind if isinstance(exc, ProviderError) else "invalid_output"
                chapter.status, chapter.error = "failed", str(exc)[:2000]
                await session.commit()
                result.failed += 1
                result.failures[kind] = result.failures.get(kind, 0) + 1
                await _emit(on_progress, ProgressEvent(
                    chapter_number=number, chapter_title=title, done=done, total=len(todo),
                    status="failed", error=chapter.error,
                ))  # fmt: skip
                continue

            # Re-running a chapter replaces its earlier facts.
            for model in (FactRow, ElapsedTimeFactRow):
                await session.execute(delete(model).where(model.chapter_id == chapter_id))

            # Real names revealed in this chapter first, so statements using them resolve
            # to the right (renamed or merged) character.
            for revealed in extraction.revealed_names:
                index.reveal(revealed.known_as, revealed.real_name)

            fact_rows = []
            for located in extraction.ages:
                s = located.statement
                character_id = index.resolve(s.mention, s.resolved_name)
                if character_id is None:
                    result.dropped_statements += 1
                    continue
                fact_rows.append(
                    FactRow(
                        user_id=user_id, book_id=book_id, chapter_id=chapter_id,
                        chapter_number=number, character_id=character_id,
                        category=AGE.name, attribute=s.statement_type,
                        mention=s.mention, raw_text=s.raw_text,
                        value_num=s.value, value_max=s.value_max,
                        value_text=s.life_stage.value if s.life_stage else None,
                        is_flashback=s.is_flashback,
                        years_before_present=s.years_before_present,
                        years_before_present_quote=s.years_before_present_quote,
                        is_speculative=s.speculative, qualifiers={},
                        char_start=located.char_start, char_end=located.char_end,
                    )
                )  # fmt: skip

            for c in index.new_characters:
                session.add(
                    Character(
                        id=c.id, user_id=user_id, book_id=book_id, canonical_name=c.canonical_name
                    )
                )
            await session.flush()  # characters must exist before rows reference them
            for rename in index.renames:
                await session.execute(
                    update(Character)
                    .where(Character.id == rename.character_id)
                    .values(canonical_name=rename.new_name)
                )
            for merge in index.merges:
                log.info(
                    "chapter %s: merging character %s into %s",
                    number, merge.from_id, merge.into_id,
                )  # fmt: skip
                for model in (FactRow, CharacterAlias):
                    await session.execute(
                        update(model)
                        .where(model.character_id == merge.from_id)
                        .values(character_id=merge.into_id)
                    )
                await session.execute(delete(Character).where(Character.id == merge.from_id))
            for a in index.new_aliases:
                session.add(
                    CharacterAlias(
                        user_id=user_id, book_id=book_id, character_id=a.character_id,
                        alias=a.alias, first_chapter=number, source="extracted",
                    )
                )  # fmt: skip
            session.add_all(fact_rows)
            for located in extraction.elapsed:
                e = located.statement
                session.add(
                    ElapsedTimeFactRow(
                        user_id=user_id, book_id=book_id, chapter_id=chapter_id,
                        chapter_number=number, raw_text=e.raw_text,
                        estimated_years=e.estimated_years, kind=e.kind,
                        is_flashback=e.is_flashback,
                        char_start=located.char_start, char_end=located.char_end,
                    )
                )  # fmt: skip

            chapter.status, chapter.error = "extracted", None
            await session.commit()

        result.extracted += 1
        result.dropped_statements += len(extraction.dropped)
        result.llm_calls += extraction.llm_calls
        result.cache_hits += extraction.cache_hits
        result.input_tokens += extraction.usage.input_tokens
        result.output_tokens += extraction.usage.output_tokens
        result.cost_usd += extraction.cost_usd
        await _emit(on_progress, ProgressEvent(
            chapter_number=number, chapter_title=title, done=done, total=len(todo),
            status="extracted",
        ))  # fmt: skip

    return result
