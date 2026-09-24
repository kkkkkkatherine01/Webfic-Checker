"""Run checkers over a book and sync their findings into consistency_issues."""

import uuid

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.checkers.age import AgeFact, ElapsedFact, check_ages
from webfic.checkers.types import ConsistencyIssue, IssueStatus
from webfic.db.models import Book, Character, ElapsedTimeFactRow, FactRow, IssueRow
from webfic.extraction.schemas import LifeStage
from webfic.facts.registry import AGE
from webfic.services.errors import NotFound


class CheckResult(BaseModel):
    found: int
    new: int
    resolved: int  # previously open or acknowledged issues that no longer occur


async def _load_age_facts(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> tuple[list[AgeFact], list[ElapsedFact]]:
    rows = (
        await session.execute(
            select(FactRow, Character.canonical_name)
            .join(Character, Character.id == FactRow.character_id)
            .where(
                FactRow.user_id == user_id,
                FactRow.book_id == book_id,
                FactRow.category == AGE.name,
            )
        )
    ).all()
    ages = [
        AgeFact(
            id=r.id,
            character_id=r.character_id,
            character_name=name,
            raw_text=r.raw_text,
            statement_type=r.attribute,
            value=r.value_num,
            life_stage=LifeStage(r.value_text) if r.value_text else None,
            is_flashback=r.is_flashback,
            years_before_present=r.years_before_present,
            value_max=r.value_max,
            speculative=r.is_speculative,
            chapter_number=r.chapter_number,
            char_start=r.char_start,
            char_end=r.char_end,
            chapter_id=r.chapter_id,
        )
        for r, name in rows
    ]
    elapsed_rows = (
        await session.scalars(
            select(ElapsedTimeFactRow).where(
                ElapsedTimeFactRow.user_id == user_id, ElapsedTimeFactRow.book_id == book_id
            )
        )
    ).all()
    elapsed = [
        ElapsedFact(
            id=r.id,
            raw_text=r.raw_text,
            estimated_years=r.estimated_years,
            kind=r.kind,
            is_flashback=r.is_flashback,
            chapter_number=r.chapter_number,
            char_start=r.char_start,
            char_end=r.char_end,
        )
        for r in elapsed_rows
    ]
    return ages, elapsed


async def run_checks(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID
) -> CheckResult:
    book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
    if book is None:
        raise NotFound(f"book {book_id}")

    ages, elapsed = await _load_age_facts(session, user_id, book_id)
    found: list[ConsistencyIssue] = check_ages(ages, elapsed)

    existing = {
        row.fingerprint: row
        for row in await session.scalars(
            select(IssueRow).where(IssueRow.user_id == user_id, IssueRow.book_id == book_id)
        )
    }

    new = 0
    for issue in found:
        evidence = [e.model_dump() for e in issue.evidence]
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
                row.status = IssueStatus.OPEN  # it came back

    # An issue that no longer occurs was fixed, whether or not the author had confirmed
    # it; only "intentional" is kept, as the author's standing decision.
    resolved = 0
    for row in existing.values():
        if row.status in (IssueStatus.OPEN, IssueStatus.ACKNOWLEDGED):
            row.status = IssueStatus.RESOLVED
            resolved += 1

    await session.commit()
    return CheckResult(found=len(found), new=new, resolved=resolved)
