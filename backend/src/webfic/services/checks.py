"""Run checkers over a book and sync their findings into consistency_issues."""

import uuid
from collections.abc import Sequence

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.checkers.registry import Checker, registered
from webfic.checkers.types import ConsistencyIssue, IssueStatus
from webfic.db.models import Book, Chapter, IssueRow
from webfic.services.errors import NotFound


class CheckResult(BaseModel):
    found: int
    new: int
    resolved: int  # previously open or acknowledged issues that no longer occur
    # Issues that no longer occur but quote a chapter that is not extracted: kept as they
    # were, since the chapter's facts are missing, not fixed (step 4.6).
    kept_unextracted: int = 0
    unextracted: list[int] = []  # numbers of the book's chapters not extracted
    by_checker: dict[str, int] = {}  # issues found, per checker


async def run_checks(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    checkers: Sequence[Checker] | None = None,
) -> CheckResult:
    """Run the checkers (all registered ones by default) and sync the issue table: new
    issues are added, those found again keep the author's status, those no longer found
    are resolved (see below)."""
    book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
    if book is None:
        raise NotFound(f"book {book_id}")

    ran = list(checkers) if checkers is not None else registered()
    found: list[ConsistencyIssue] = []
    by_checker: dict[str, int] = {}
    for checker in ran:
        issues = await checker.check(session, user_id=user_id, book_id=book_id)
        by_checker[checker.name] = len(issues)
        found += issues

    # Only issues of the checkers that ran are compared with what they found: running
    # one checker must not close the others' issues (step 5-0).
    names = [c.name for c in ran]
    existing = {
        row.fingerprint: row
        for row in await session.scalars(
            select(IssueRow).where(
                IssueRow.user_id == user_id,
                IssueRow.book_id == book_id,
                IssueRow.checker.in_(names),
            )
        )
    }

    new = 0
    for issue in found:
        evidence = [e.model_dump(mode="json") for e in issue.evidence]
        row = existing.pop(issue.fingerprint, None)
        if row is None:
            session.add(
                IssueRow(
                    user_id=user_id,
                    book_id=book_id,
                    checker=issue.checker,
                    issue_type=issue.issue_type,
                    confidence=issue.confidence,
                    status=IssueStatus.OPEN,
                    description=issue.description,
                    evidence=evidence,
                    subjects=issue.subjects,
                    fingerprint=issue.fingerprint,
                )
            )
            new += 1
        else:
            # Keep the author's status; refresh the wording and evidence.
            row.confidence = issue.confidence
            row.description = issue.description
            row.evidence = evidence
            row.subjects = issue.subjects
            if row.status == IssueStatus.RESOLVED:
                # It came back: as it was when a check closed it, else (closed by the
                # author) open.
                row.status = row.resolved_from or IssueStatus.OPEN
                row.resolved_from = None

    # An issue that no longer occurs was fixed, whether or not the author had confirmed
    # it; only "intentional" is kept, as the author's standing decision. An issue quoting
    # a chapter that is not extracted (failed, or waiting) is not known to be fixed: its
    # facts are simply missing, so it is left as it is.
    missing = await _unextracted(session, user_id, book_id)
    resolved = kept = 0
    for row in existing.values():
        if row.status not in (IssueStatus.OPEN, IssueStatus.ACKNOWLEDGED):
            continue
        if any(_chapter_of(e, missing) for e in row.evidence):
            kept += 1
            continue
        row.resolved_from, row.status = row.status, IssueStatus.RESOLVED
        resolved += 1

    await session.commit()
    return CheckResult(
        found=len(found), new=new, resolved=resolved, kept_unextracted=kept,
        unextracted=sorted(missing.values()), by_checker=by_checker,
    )  # fmt: skip


async def _unextracted(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> dict[uuid.UUID, int]:
    """The book's chapters that are not extracted: id -> number."""
    rows = await session.execute(
        select(Chapter.id, Chapter.number).where(
            Chapter.user_id == user_id, Chapter.book_id == book_id, Chapter.status != "extracted"
        )
    )
    return dict(rows.tuples().all())


def _chapter_of(evidence: dict, missing: dict[uuid.UUID, int]) -> bool:
    """Whether a piece of stored evidence lies in one of the `missing` chapters; by id,
    or by number for evidence stored before step 4.6."""
    chapter_id = evidence.get("chapter_id")
    if chapter_id is not None:
        return uuid.UUID(chapter_id) in missing
    return evidence["chapter_number"] in missing.values()
