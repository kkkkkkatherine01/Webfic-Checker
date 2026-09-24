"""Chapter management: append, replace, delete and patch chapters of an imported book.

Web novels are serialised, so appending is the common case; replacing, deleting and
patching share "recompute from chapter N": undo the character changes of chapters N..,
drop their facts, roll the Core back to chapter N-1, extract N.. again (unchanged chapters
hit the LLM cache and cost nothing) and re-run the checks, which keep the author's status
on issues through their content-based fingerprints.

Nothing here assumes the edit comes from the author: the revision agent of step 8 will
use the same services. Every operation can run as a dry run, which reports the issues it
would add and remove and then rolls the database back (the LLM usage ledger excepted:
the calls were really made and paid for).
"""

import hashlib
import uuid
from collections.abc import Awaitable, Callable
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.checkers.types import IssueStatus
from webfic.config import Settings
from webfic.db.models import (
    Book,
    Chapter,
    ChapterExtractionRow,
    ElapsedTimeFactRow,
    FactRow,
    IssueRow,
)
from webfic.db.session import rolled_back
from webfic.ingest.splitter import normalize_newlines, split_chapters
from webfic.llm.base import LLMClient
from webfic.memory import core, events
from webfic.services import checks, imports
from webfic.services.errors import InvalidEdit, NotFound
from webfic.services.reports import IssueView, issue_view, sort_issues

Factory = async_sessionmaker[AsyncSession]


class ChangeResult(BaseModel):
    dry_run: bool  # True: nothing was saved
    chapters: int  # chapters in the book after the change
    extracted: int  # chapters (re-)extracted
    failed: int
    llm_calls: int
    cache_hits: int
    reused: int  # unchanged chapters whose stored extraction was reused
    cost_usd: Decimal
    issues_added: list[IssueView]  # open after the change, not before
    issues_removed: list[IssueView]  # open before the change, not after
    warnings: list[str] = []


# --- shared steps ----------------------------------------------------------------------


async def _book(session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID) -> Book:
    book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
    if book is None:
        raise NotFound(f"book {book_id}")
    return book


async def _chapter(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, number: int
) -> Chapter:
    chapter = await session.scalar(
        select(Chapter).where(
            Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number == number
        )
    )
    if chapter is None:
        raise NotFound(f"第 {number} 章")
    return chapter


async def _open_issues(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> dict[str, IssueView]:
    rows = await session.scalars(
        select(IssueRow).where(
            IssueRow.user_id == user_id,
            IssueRow.book_id == book_id,
            IssueRow.status.in_([IssueStatus.OPEN, IssueStatus.ACKNOWLEDGED]),
        )
    )
    return {row.fingerprint: issue_view(row) for row in rows}


async def _reset_from(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, number: int
) -> None:
    """Forget everything derived from chapters `number`..: character changes, facts,
    Core snapshots. The chapters themselves are left in place."""
    await events.undo_from_chapter(session, user_id=user_id, book_id=book_id, chapter_number=number)
    for model in (FactRow, ElapsedTimeFactRow):
        await session.execute(
            delete(model).where(
                model.user_id == user_id, model.book_id == book_id, model.chapter_number >= number
            )
        )
    await core.restore(session, user_id=user_id, book_id=book_id, chapter_number=number)


async def _mark_pending(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, number: int
) -> None:
    await session.execute(
        update(Chapter)
        .where(Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number >= number)
        .values(status="pending", error=None)
    )


def _set_content(chapter: Chapter, content: str) -> None:
    chapter.content = content
    chapter.char_count = len(content)
    chapter.content_hash = hashlib.sha256(content.encode()).hexdigest()


Edit = Callable[[AsyncSession], Awaitable[list[str]]]  # changes the chapters; returns warnings


async def _apply(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    edit: Edit,
    dry_run: bool,
) -> ChangeResult:
    """Run `edit`, extract what it left pending, re-check the book and report the issues
    that appeared or disappeared; roll it all back for a dry run."""

    async def run(scoped: Factory) -> ChangeResult:
        async with scoped() as session:
            await _book(session, user_id, book_id)
            before = await _open_issues(session, user_id, book_id)
            warnings = await edit(session)
            await session.commit()

        extraction = await imports.run_import_job(
            scoped, llm, settings, user_id=user_id, book_id=book_id
        )
        async with scoped() as session:
            await checks.run_checks(session, user_id=user_id, book_id=book_id)
            after = await _open_issues(session, user_id, book_id)
            chapters = await session.scalar(
                select(func.count()).where(Chapter.user_id == user_id, Chapter.book_id == book_id)
            )
        return ChangeResult(
            dry_run=dry_run,
            chapters=chapters or 0,
            extracted=extraction.extracted,
            failed=extraction.failed,
            llm_calls=extraction.llm_calls,
            cache_hits=extraction.cache_hits,
            reused=extraction.reused,
            cost_usd=extraction.cost_usd,
            issues_added=sort_issues([v for fp, v in after.items() if fp not in before]),
            issues_removed=sort_issues([v for fp, v in before.items() if fp not in after]),
            warnings=warnings,
        )

    if not dry_run:
        return await run(factory)
    async with rolled_back(factory) as scoped:
        return await run(scoped)


# --- operations ------------------------------------------------------------------------


async def append_chapters(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    text: str,
    dry_run: bool = False,
) -> ChangeResult:
    """Add one or more chapters (split like an import) after the last one."""

    async def edit(session: AsyncSession) -> list[str]:
        split = split_chapters(text)
        last = await session.scalar(
            select(func.max(Chapter.number)).where(
                Chapter.user_id == user_id, Chapter.book_id == book_id
            )
        )
        for offset, raw in enumerate(split.chapters, start=1):
            chapter = Chapter(
                user_id=user_id, book_id=book_id, number=(last or 0) + offset,
                title=raw.title, status="pending",
            )  # fmt: skip
            _set_content(chapter, raw.content)
            session.add(chapter)
        return split.warnings

    return await _apply(
        factory, llm, settings, user_id=user_id, book_id=book_id, edit=edit, dry_run=dry_run
    )


async def replace_chapter(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    number: int,
    content: str,
    dry_run: bool = False,
) -> ChangeResult:
    """Replace chapter `number`'s text and recompute from it."""
    content = normalize_newlines(content).strip("\n")
    if not content.strip():
        raise InvalidEdit("新内容为空；要删除这一章请用删除章节")

    async def edit(session: AsyncSession) -> list[str]:
        chapter = await _chapter(session, user_id, book_id, number)
        await _reset_from(session, user_id, book_id, number)
        _set_content(chapter, content)
        await _mark_pending(session, user_id, book_id, number)
        return []

    return await _apply(
        factory, llm, settings, user_id=user_id, book_id=book_id, edit=edit, dry_run=dry_run
    )


async def delete_chapter(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    number: int,
    dry_run: bool = False,
) -> ChangeResult:
    """Delete chapter `number`; later chapters move up one number and are recomputed."""

    async def edit(session: AsyncSession) -> list[str]:
        chapter = await _chapter(session, user_id, book_id, number)
        # Undo first: the chapter's change log goes with the chapter row.
        await _reset_from(session, user_id, book_id, number)
        await session.execute(
            delete(ChapterExtractionRow).where(ChapterExtractionRow.chapter_id == chapter.id)
        )
        await session.delete(chapter)
        await session.flush()
        later = (Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number > number)
        # Two steps, so no two chapters ever share a number (unique per book).
        await session.execute(update(Chapter).where(*later).values(number=-Chapter.number))
        await session.execute(
            update(Chapter)
            .where(Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.number < 0)
            .values(number=-Chapter.number - 1)
        )
        await _mark_pending(session, user_id, book_id, number)
        return []

    return await _apply(
        factory, llm, settings, user_id=user_id, book_id=book_id, edit=edit, dry_run=dry_run
    )


async def patch_chapter(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    number: int,
    old: str,
    new: str,
    dry_run: bool = False,
) -> ChangeResult:
    """Replace one passage of chapter `number`; `old` must occur exactly once. This is
    the form revisions proposed by an agent take."""
    if not old:
        raise InvalidEdit("原片段为空")
    async with factory() as session:
        await _book(session, user_id, book_id)
        content = (await _chapter(session, user_id, book_id, number)).content
    found = content.count(old)
    if found != 1:
        where = "找不到" if found == 0 else f"出现了 {found} 次，无法确定改哪一处"
        raise InvalidEdit(f"第 {number} 章里{where}：「{old}」")
    return await replace_chapter(
        factory, llm, settings, user_id=user_id, book_id=book_id, number=number,
        content=content.replace(old, new), dry_run=dry_run,
    )  # fmt: skip
