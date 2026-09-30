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
from webfic.extraction.extractor import ChapterExtraction
from webfic.extraction.kinds import KINDS, ExtractionKind, Where
from webfic.extraction.resolver import USER, CharacterIndex, KnownCharacter
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
    llm_calls: int  # extraction calls, all kinds
    calls_by_kind: dict[str, int] = {}  # extraction calls per kind (step 5-0)
    cache_hits: int
    reused: int = 0  # chapters whose stored extraction was reused (no model call)
    note_calls: int = 0  # model calls looking for author's notes (cost included in cost_usd)
    # A chapter that had failed was followed by extracted ones, so everything from it on
    # was recomputed in order (see run_import_job).
    recomputed_from: int | None = None
    # A kind of extraction the book had not been read with yet: everything from this
    # chapter on was recomputed in order (see run_import_job).
    upgraded_from: int | None = None
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
    """The book's characters, for resolving mentions; kinds of extraction in reading
    order."""
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
    kinds: dict[uuid.UUID, dict[str, str]] = {}
    for a in aliases:
        by_character.setdefault(a.character_id, []).append(a.alias)
        kinds.setdefault(a.character_id, {})[a.alias] = USER if a.source == "user" else a.kind
    return CharacterIndex(
        [
            KnownCharacter(
                c.id, c.canonical_name, by_character.get(c.id, []), c.kind, kinds.get(c.id, {})
            )
            for c in characters
        ],
        [kind.name for kind in KINDS],
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


async def _first_lacking_kind(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> int | None:
    """The first extracted chapter that has stored readings, but not of every kind.
    Chapters with none at all were read before readings were stored (step 3a-3) and
    are left alone: re-reading them would call the model for kinds they already have."""
    rows = await session.execute(
        select(Chapter.number, ChapterExtractionRow.kind)
        .join(ChapterExtractionRow, ChapterExtractionRow.chapter_id == Chapter.id)
        .where(
            Chapter.user_id == user_id,
            Chapter.book_id == book_id,
            Chapter.status == "extracted",
        )
    )
    kinds: dict[int, set[str]] = {}
    for number, kind in rows:
        kinds.setdefault(number, set()).add(kind)
    wanted = {kind.name for kind in KINDS}
    lacking = [number for number, have in kinds.items() if wanted - have]
    return min(lacking) if lacking else None


async def _read(
    session: AsyncSession,
    kind: ExtractionKind,
    llm: LLMClient,
    settings: Settings,
    stored: ChapterExtractionRow | None,
    *,
    chapter: Chapter,
    story: str,
    story_start: int,
    story_hash: str,
    version: str,
    known: str,
) -> ChapterExtraction:
    """One kind's reading of a chapter: the stored one while the story text and the
    kind's setup are unchanged, else a new one, which is stored."""
    if stored is not None and stored.content_hash == story_hash and stored.version == version:
        reading = kind.load(stored.result, offset=story_start)
        # A safety net: a reading whose quotes are not where it says is not reused
        # (step 4.6: positions once went stale this way).
        if kind.positions_hold(reading, chapter.content):
            reading.reused = True
            return reading
        log.warning("chapter %s: stored %s reading out of place", chapter.number, kind.name)
    reading = await kind.extract(
        llm, chapter_number=chapter.number, text=story, known_characters=known,
        settings=settings, offset=story_start,
    )  # fmt: skip
    # Updated in place, not deleted and re-added: a flush here would hold the database's
    # write lock (SQLite) while the next kind's model calls are being recorded.
    if stored is not None:
        stored.content_hash, stored.version = story_hash, version
        stored.result = kind.dump(reading, offset=story_start)
    else:
        session.add(
            ChapterExtractionRow(
                user_id=chapter.user_id, book_id=chapter.book_id, chapter_id=chapter.id,
                kind=kind.name, content_hash=story_hash, version=version,
                result=kind.dump(reading, offset=story_start),
            )
        )  # fmt: skip
    return reading


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
    upgraded_from: int | None = None
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
            # A kind of extraction added since the book was read (step 5-0): read the
            # book again in order from the first chapter without it; the kinds it has
            # are reused, only the new one asks the model.
            lacking = await _first_lacking_kind(session, user_id, book_id)
            if lacking is not None:
                await reset_from(session, user_id=user_id, book_id=book_id, number=lacking)
                await mark_pending(session, user_id=user_id, book_id=book_id, number=lacking)
                await session.commit()
                upgraded_from = lacking
        todo = (
            await session.execute(
                select(Chapter.id, Chapter.number, Chapter.title)
                .where(*not_extracted, Chapter.number >= (from_chapter or 0))
                .order_by(Chapter.number)
            )
        ).all()

    versions = {kind.name: kind.version(settings) for kind in KINDS}
    result = ImportJobResult(
        extracted=0, failed=0, dropped_statements=0, llm_calls=0, cache_hits=0,
        input_tokens=0, output_tokens=0, cost_usd=Decimal(0),
        recomputed_from=recomputed_from, upgraded_from=upgraded_from, left_failed=left_failed,
    )  # fmt: skip

    for done, (chapter_id, number, title) in enumerate(todo, start=1):
        async with session_factory() as session:
            chapter = await session.get_one(Chapter, chapter_id)
            index = await _load_character_index(session, user_id, book_id)
            # An unchanged chapter keeps its earlier readings: asking the model again gives
            # a slightly different one, which would blur what an edit really changed.
            stored = {
                row.kind: row
                for row in await session.scalars(
                    select(ChapterExtractionRow).where(
                        ChapterExtractionRow.user_id == user_id,
                        ChapterExtractionRow.chapter_id == chapter_id,
                    )
                )
            }
            readings: dict[str, ChapterExtraction] = {}
            try:
                # Author's notes are left out: only the story between them is extracted,
                # and a stored reading is reused while that text is unchanged.
                scan = await author_note_scan(
                    session, llm, user_id=user_id, book_id=book_id, chapter=chapter,
                    markers=markers,
                )  # fmt: skip
                result.note_calls += scan.llm_calls
                result.cost_usd += scan.cost_usd
                story_start, story_end = scan.story(chapter.content)
                story = chapter.content[story_start:story_end]
                story_hash = hashlib.sha256(story.encode()).hexdigest()
                # Each kind is told the characters known before this chapter that it or
                # the kinds before it named (step 5-1: adding a kind never changes what
                # the earlier kinds are asked).
                for n, kind in enumerate(KINDS):
                    readings[kind.name] = await _read(
                        session, kind, llm, settings, stored.get(kind.name),
                        chapter=chapter, story=story, story_start=story_start,
                        story_hash=story_hash, version=versions[kind.name],
                        known=index.for_prompt({k.name for k in KINDS[: n + 1]}),
                    )  # fmt: skip
                if all(r.reused for r in readings.values()):
                    result.reused += 1
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
            for kind in KINDS:
                index.kind = kind.name
                for revealed in kind.reveals(readings[kind.name]):
                    index.reveal(revealed.known_as, revealed.real_name)

            where = Where(user_id, book_id, chapter_id, number)
            fact_rows: list[FactRow] = []
            span_rows: list[ElapsedTimeFactRow] = []
            for kind in KINDS:
                index.kind = kind.name
                rows = kind.rows(
                    readings[kind.name], resolve=index.resolve, name_of=index.name_of, where=where
                )
                fact_rows += rows.facts
                span_rows += rows.spans
                result.dropped_statements += rows.unattributed

            # Every change to the character table is logged, so recomputing from this
            # chapter can undo it (webfic.memory.events).
            changes: list[tuple[EventKind, dict[str, Any]]] = []
            # First: they change rows that were there before this chapter.
            for p in index.promotions:
                if p.alias is None:
                    await session.execute(
                        update(Character)
                        .where(Character.user_id == user_id, Character.id == p.character_id)
                        .values(kind=p.new_kind)
                    )
                    changes.append((EventKind.PROMOTE, {
                        "character_id": str(p.character_id), "old_kind": p.old_kind,
                    }))  # fmt: skip
                    continue
                alias_id = await session.scalar(
                    select(CharacterAlias.id).where(
                        CharacterAlias.user_id == user_id,
                        CharacterAlias.book_id == book_id,
                        CharacterAlias.alias == p.alias,
                    )
                )
                if alias_id is not None:
                    await session.execute(
                        update(CharacterAlias)
                        .where(CharacterAlias.user_id == user_id, CharacterAlias.id == alias_id)
                        .values(kind=p.new_kind)
                    )
                    changes.append((EventKind.PROMOTE, {
                        "alias_id": str(alias_id), "old_kind": p.old_kind,
                    }))  # fmt: skip
            for c in index.new_characters:
                session.add(
                    Character(
                        id=c.id,
                        user_id=user_id,
                        book_id=book_id,
                        canonical_name=c.canonical_name,
                        kind=c.kind,
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
                    "from_kind": merge.from_kind,
                    "into_id": str(merge.into_id), **moved,
                }))  # fmt: skip
            for a in index.new_aliases:
                alias_id = uuid.uuid4()
                session.add(
                    CharacterAlias(
                        id=alias_id, user_id=user_id, book_id=book_id, character_id=a.character_id,
                        alias=a.alias, first_chapter=number, source="extracted", kind=a.kind,
                    )
                )  # fmt: skip
                changes.append((EventKind.ALIAS, {
                    "alias_id": str(alias_id), "character_id": str(a.character_id),
                    "alias": a.alias,
                }))  # fmt: skip
            session.add_all(fact_rows)
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
        for name, reading in readings.items():
            result.dropped_statements += len(reading.dropped)
            result.llm_calls += reading.llm_calls
            result.calls_by_kind[name] = result.calls_by_kind.get(name, 0) + reading.llm_calls
            result.cache_hits += reading.cache_hits
            result.input_tokens += reading.usage.input_tokens
            result.output_tokens += reading.usage.output_tokens
            result.cost_usd += reading.cost_usd
        await _emit(on_progress, ProgressEvent(
            chapter_number=number, chapter_title=title, done=done, total=len(todo),
            status="extracted",
        ))  # fmt: skip

    return result
