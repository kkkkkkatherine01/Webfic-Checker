"""Import a book: `create_import_job` splits and stores chapters; `run_import_job` extracts
facts chapter by chapter. The CLI calls both in a row; before launch a worker runs the
second one."""

import hashlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.archival.index import Archival, index_chapter, retokenize_for_names
from webfic.config import Settings
from webfic.db.models import (
    Book,
    Chapter,
    ChapterExtractionRow,
    ChapterNoteRow,
    Character,
    CharacterAlias,
    CharacterStateRow,
    ElapsedTimeFactRow,
    FactRow,
)
from webfic.extraction.author_notes import NoteScan, find_author_notes
from webfic.extraction.author_notes import version as note_version
from webfic.extraction.extractor import (
    dump_extraction,
    extract_chapter,
    extraction_version,
    load_extraction,
    load_prompt,
)
from webfic.extraction.resolver import CharacterIndex, KnownCharacter
from webfic.facts.registry import AGE
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import LLMClient, LLMError, ProviderError, ProviderErrorKind
from webfic.memory import core, events
from webfic.memory.events import EventKind
from webfic.services.errors import InvalidEdit, NotFound

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
    reused: int = 0  # chapters whose stored extraction was reused (no model call)
    note_calls: int = 0  # model calls looking for author's notes (cost included in cost_usd)
    # A chapter that had failed was followed by extracted ones, so everything from it on
    # was recomputed in order (see run_import_job).
    recomputed_from: int | None = None
    # With `from_chapter`: earlier chapters that are still not extracted (left alone).
    left_failed: list[int] = []
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal


ProgressCallback = Callable[[ProgressEvent], Awaitable[None] | None]


async def create_import_job(
    session: AsyncSession, *, user_id: uuid.UUID, title: str, text: str
) -> ImportJobCreated:
    split = split_chapters(text)
    if not split.chapters:
        raise InvalidEdit("没有可导入的正文")
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


async def reset_from(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, number: int
) -> None:
    """Forget everything derived from chapters `number`..: character changes, facts,
    Core snapshots. The chapters themselves, their stored extractions and their passages
    are left in place (unchanged chapters reuse them)."""
    await events.undo_from_chapter(session, user_id=user_id, book_id=book_id, chapter_number=number)
    for model in (FactRow, ElapsedTimeFactRow):
        await session.execute(
            delete(model).where(
                model.user_id == user_id, model.book_id == book_id, model.chapter_number >= number
            )
        )
    await core.restore(session, user_id=user_id, book_id=book_id, chapter_number=number)


async def mark_pending(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, number: int
) -> None:
    await session.execute(
        update(Chapter)
        .where(Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number >= number)
        .values(status="pending", error=None)
    )


async def author_note_scan(
    session: AsyncSession,
    llm: LLMClient | None,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter: Chapter,
    markers: list[str],
) -> NoteScan:
    """The chapter's author's notes: the stored scan while the chapter text and the scan
    setup are unchanged, else a new scan, stored when the model took part (a scan by
    markers only, after a model failure, is tried again next time)."""
    tag = note_version(markers)
    stored = await session.scalar(
        select(ChapterNoteRow).where(
            ChapterNoteRow.user_id == user_id, ChapterNoteRow.chapter_id == chapter.id
        )
    )
    if stored is not None and stored.content_hash == chapter.content_hash and stored.version == tag:
        return NoteScan(ranges=[(a, b) for a, b in stored.ranges], model_used=stored.model_used)
    scan = await find_author_notes(llm, chapter.content, markers)
    # Updated in place, not deleted and re-added: a flush here would hold the database's
    # write lock (SQLite) while the extraction's model calls are being recorded.
    ranges = [list(r) for r in scan.ranges]
    if not scan.model_used:
        if stored is not None:
            await session.delete(stored)
    elif stored is not None:
        stored.content_hash, stored.version, stored.ranges = chapter.content_hash, tag, ranges
        stored.model_used = True
    else:
        session.add(
            ChapterNoteRow(
                user_id=user_id, book_id=book_id, chapter_id=chapter.id,
                content_hash=chapter.content_hash, version=tag, ranges=ranges,
                model_used=True,
            )
        )  # fmt: skip
    return scan


async def _index(
    session: AsyncSession,
    archival: Archival,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter: Chapter,
    index: CharacterIndex,
) -> None:
    await index_chapter(
        session, archival, user_id=user_id, book_id=book_id, chapter=chapter, names=_names(index)
    )


def _names(index: CharacterIndex) -> list[str]:
    return [n for c in index.characters() for n in (c.canonical_name, *c.aliases)]


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
    archival: Archival | None = None,
    from_chapter: int | None = None,
) -> ImportJobResult:
    """Extract every chapter that is not yet `extracted`, in narrative order. Each chapter
    commits on its own, so an interrupted run can simply be started again. With
    `archival`, each chapter's passages are indexed for search in the same transaction.

    Chapters must be processed in order: each one's Core snapshot builds on the previous
    one and its extraction sees the characters known so far. So when a chapter that had
    failed is followed by extracted ones, everything from it on is recomputed (unchanged
    chapters reuse their stored extraction, so only the failed one costs a model call).
    With `from_chapter` (chapter management, which has already reset from there), only
    chapters from that one on are processed; earlier failed ones are left as they are."""
    recomputed_from: int | None = None
    left_failed: list[int] = []
    async with session_factory() as session:
        book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
        if book is None:
            raise NotFound(f"book {book_id}")
        markers = list(book.author_note_markers or [])
        not_extracted = (
            Chapter.user_id == user_id,
            Chapter.book_id == book_id,
            Chapter.status != "extracted",
        )
        if from_chapter is not None:
            left_failed = list(
                await session.scalars(
                    select(Chapter.number)
                    .where(*not_extracted, Chapter.number < from_chapter)
                    .order_by(Chapter.number)
                )
            )
        else:
            first = await session.scalar(select(func.min(Chapter.number)).where(*not_extracted))
            later_done = first is not None and await session.scalar(
                select(Chapter.id)
                .where(
                    Chapter.user_id == user_id,
                    Chapter.book_id == book_id,
                    Chapter.status == "extracted",
                    Chapter.number > first,
                )
                .limit(1)
            )
            if first is not None and later_done:
                await reset_from(session, user_id=user_id, book_id=book_id, number=first)
                await mark_pending(session, user_id=user_id, book_id=book_id, number=first)
                await session.commit()
                recomputed_from = first
        todo = (
            await session.execute(
                select(Chapter.id, Chapter.number, Chapter.title)
                .where(*not_extracted, Chapter.number >= (from_chapter or 0))
                .order_by(Chapter.number)
            )
        ).all()

    system_prompt = load_prompt()
    version = extraction_version(system_prompt, settings.chunk_size, settings.chunk_overlap)
    result = ImportJobResult(
        extracted=0, failed=0, dropped_statements=0, llm_calls=0, cache_hits=0,
        input_tokens=0, output_tokens=0, cost_usd=Decimal(0),
        recomputed_from=recomputed_from, left_failed=left_failed,
    )  # fmt: skip

    for done, (chapter_id, number, title) in enumerate(todo, start=1):
        async with session_factory() as session:
            chapter = await session.get_one(Chapter, chapter_id)
            index = await _load_character_index(session, user_id, book_id)
            # An unchanged chapter keeps its earlier reading: asking the model again gives
            # a slightly different one, which would blur what an edit really changed.
            stored = await session.scalar(
                select(ChapterExtractionRow).where(
                    ChapterExtractionRow.user_id == user_id,
                    ChapterExtractionRow.chapter_id == chapter_id,
                )
            )
            try:
                # Author's notes are left out: only the story between them is extracted,
                # and a stored extraction is reused while that text is unchanged.
                scan = await author_note_scan(
                    session, llm, user_id=user_id, book_id=book_id, chapter=chapter,
                    markers=markers,
                )  # fmt: skip
                result.note_calls += scan.llm_calls
                result.cost_usd += scan.cost_usd
                story_start, story_end = scan.story(chapter.content)
                story = chapter.content[story_start:story_end]
                story_hash = hashlib.sha256(story.encode()).hexdigest()
                if (
                    stored is not None
                    and stored.content_hash == story_hash
                    and stored.version == version
                ):
                    extraction = load_extraction(stored.result)
                    result.reused += 1
                else:
                    extraction = await extract_chapter(
                        llm,
                        chapter_number=number,
                        text=story,
                        known_characters=index.for_prompt(),
                        system_prompt=system_prompt,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        offset=story_start,
                    )
                    if stored is not None:
                        await session.delete(stored)
                        await session.flush()
                    session.add(
                        ChapterExtractionRow(
                            user_id=user_id, book_id=book_id, chapter_id=chapter_id,
                            content_hash=story_hash, version=version,
                            result=dump_extraction(extraction),
                        )
                    )  # fmt: skip
            except LLMError as exc:
                if isinstance(exc, ProviderError) and exc.kind == ProviderErrorKind.AUTH:
                    raise  # a bad or unfunded key fails every chapter; stop here
                kind = exc.kind if isinstance(exc, ProviderError) else "invalid_output"
                chapter.status, chapter.error = "failed", str(exc)[:2000]
                if archival is not None:
                    # Indexing is local work: the text stays searchable, and a replaced
                    # chapter never keeps its old passages, whatever the provider says.
                    await _index(session, archival, user_id, book_id, chapter, index)
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
                await session.execute(
                    delete(model).where(model.user_id == user_id, model.chapter_id == chapter_id)
                )

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

            # Every change to the character table is logged, so recomputing from this
            # chapter can undo it (webfic.memory.events).
            changes: list[tuple[EventKind, dict[str, Any]]] = []
            for c in index.new_characters:
                session.add(
                    Character(
                        id=c.id, user_id=user_id, book_id=book_id, canonical_name=c.canonical_name
                    )
                )
                changes.append(
                    (EventKind.CREATE, {"character_id": str(c.id), "name": c.canonical_name})
                )
            await session.flush()  # characters must exist before rows reference them
            for rename in index.renames:
                await session.execute(
                    update(Character)
                    .where(Character.user_id == user_id, Character.id == rename.character_id)
                    .values(canonical_name=rename.new_name)
                )
                changes.append((EventKind.RENAME, {
                    "character_id": str(rename.character_id),
                    "old_name": rename.old_name, "new_name": rename.new_name,
                }))  # fmt: skip
            for merge in index.merges:
                log.info(
                    "chapter %s: merging character %s into %s",
                    number, merge.from_id, merge.into_id,
                )  # fmt: skip
                moved: dict[str, list[str]] = {}
                for model, key in ((FactRow, "fact_ids"), (CharacterAlias, "alias_ids")):
                    ids = await session.scalars(
                        select(model.id).where(
                            model.user_id == user_id, model.character_id == merge.from_id
                        )
                    )
                    moved[key] = [str(i) for i in ids]
                    await session.execute(
                        update(model)
                        .where(model.user_id == user_id, model.character_id == merge.from_id)
                        .values(character_id=merge.into_id)
                    )
                await session.execute(
                    delete(CharacterStateRow).where(
                        CharacterStateRow.user_id == user_id,
                        CharacterStateRow.character_id == merge.from_id,
                    )
                )
                await session.execute(
                    delete(Character).where(
                        Character.user_id == user_id, Character.id == merge.from_id
                    )
                )
                changes.append((EventKind.MERGE, {
                    "from_id": str(merge.from_id), "from_name": merge.from_name,
                    "into_id": str(merge.into_id), **moved,
                }))  # fmt: skip
            for a in index.new_aliases:
                alias_id = uuid.uuid4()
                session.add(
                    CharacterAlias(
                        id=alias_id, user_id=user_id, book_id=book_id, character_id=a.character_id,
                        alias=a.alias, first_chapter=number, source="extracted",
                    )
                )  # fmt: skip
                changes.append((EventKind.ALIAS, {
                    "alias_id": str(alias_id), "character_id": str(a.character_id),
                    "alias": a.alias,
                }))  # fmt: skip
            session.add_all(fact_rows)
            span_rows = [
                ElapsedTimeFactRow(
                    user_id=user_id, book_id=book_id, chapter_id=chapter_id,
                    chapter_number=number, raw_text=located.statement.raw_text,
                    estimated_years=located.statement.estimated_years,
                    kind=located.statement.kind, is_flashback=located.statement.is_flashback,
                    char_start=located.char_start, char_end=located.char_end,
                )
                for located in extraction.elapsed
            ]  # fmt: skip
            session.add_all(span_rows)
            await session.flush()
            await events.record(
                session, user_id=user_id, book_id=book_id, chapter_id=chapter_id,
                chapter_number=number, events=changes,
            )  # fmt: skip

            # Core: the state after this chapter, from the state after the previous one.
            previous = await core.load_state(
                session, user_id=user_id, book_id=book_id, before_chapter=number
            )
            state = core.advance(
                previous, chapter_number=number, facts=fact_rows, spans=span_rows,
                characters=index.characters(),
                merges=[(m.from_id, m.into_id) for m in index.merges],
            )  # fmt: skip
            await core.save_state(
                session, user_id=user_id, book_id=book_id, chapter_id=chapter_id, state=state
            )
            if archival is not None:
                await _index(session, archival, user_id, book_id, chapter, index)
                # Names first known in this chapter were cut into pieces in passages
                # indexed before; re-cut those passages so keyword search finds them.
                new_names = [
                    *(c.canonical_name for c in index.new_characters),
                    *(r.new_name for r in index.renames),
                    *(a.alias for a in index.new_aliases),
                ]
                await retokenize_for_names(
                    session, archival, user_id=user_id, book_id=book_id,
                    new_names=new_names, names=_names(index),
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
