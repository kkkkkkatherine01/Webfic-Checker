"""Read-only queries: books, issue reports, LLM usage."""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.checkers.types import Confidence, Evidence, IssueStatus
from webfic.db.models import Book, Chapter, Character, IssueRow, LLMCallRow
from webfic.services.errors import NotFound


class BookSummary(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    chapters: int
    extracted: int
    failed: int
    characters: int


class IssueView(BaseModel):
    id: uuid.UUID
    checker: str
    issue_type: str
    confidence: Confidence
    status: IssueStatus
    description: str
    evidence: list[Evidence]
    subjects: list[str]


class Report(BaseModel):
    book_id: uuid.UUID
    title: str
    issues: list[IssueView]


class UsageLine(BaseModel):
    purpose: str
    model: str
    calls: int
    cache_hits: int
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: Decimal


class UsageSummary(BaseModel):
    lines: list[UsageLine]
    total_cost_usd: Decimal


_CONFIDENCE_ORDER = {c: i for i, c in enumerate(Confidence)}


async def list_books(session: AsyncSession, *, user_id: uuid.UUID) -> list[BookSummary]:
    chapter_stats = (
        select(
            Chapter.book_id,
            func.count().label("chapters"),
            func.sum(case((Chapter.status == "extracted", 1), else_=0)).label("extracted"),
            func.sum(case((Chapter.status == "failed", 1), else_=0)).label("failed"),
        )
        .where(Chapter.user_id == user_id)
        .group_by(Chapter.book_id)
        .subquery()
    )
    character_stats = (
        select(Character.book_id, func.count().label("characters"))
        .where(Character.user_id == user_id)
        .group_by(Character.book_id)
        .subquery()
    )
    rows = await session.execute(
        select(
            Book,
            func.coalesce(chapter_stats.c.chapters, 0),
            func.coalesce(chapter_stats.c.extracted, 0),
            func.coalesce(chapter_stats.c.failed, 0),
            func.coalesce(character_stats.c.characters, 0),
        )
        .outerjoin(chapter_stats, chapter_stats.c.book_id == Book.id)
        .outerjoin(character_stats, character_stats.c.book_id == Book.id)
        .where(Book.user_id == user_id)
        .order_by(Book.created_at.desc())
    )
    return [
        BookSummary(
            id=book.id,
            title=book.title,
            created_at=book.created_at,
            chapters=chapters,
            extracted=extracted,
            failed=failed,
            characters=characters,
        )
        for book, chapters, extracted, failed, characters in rows
    ]


async def get_report(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    include_closed: bool = False,
) -> Report:
    book = await session.scalar(select(Book).where(Book.id == book_id, Book.user_id == user_id))
    if book is None:
        raise NotFound(f"book {book_id}")

    query = select(IssueRow).where(IssueRow.user_id == user_id, IssueRow.book_id == book_id)
    if not include_closed:
        query = query.where(IssueRow.status.in_([IssueStatus.OPEN, IssueStatus.ACKNOWLEDGED]))
    issues = [
        IssueView(
            id=row.id,
            checker=row.checker,
            issue_type=row.issue_type,
            confidence=Confidence(row.confidence),
            status=IssueStatus(row.status),
            description=row.description,
            evidence=[Evidence.model_validate(e) for e in row.evidence],
            subjects=row.subjects or [],
        )
        for row in await session.scalars(query)
    ]
    issues.sort(
        key=lambda i: (
            _CONFIDENCE_ORDER[i.confidence],
            min((e.chapter_number for e in i.evidence), default=0),
        )
    )
    return Report(book_id=book.id, title=book.title, issues=issues)


async def get_usage(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID | None = None
) -> UsageSummary:
    query = (
        select(
            LLMCallRow.purpose,
            LLMCallRow.model,
            func.count(),
            func.sum(case((LLMCallRow.cache_hit, 1), else_=0)),
            func.sum(LLMCallRow.input_tokens),
            func.sum(LLMCallRow.cached_input_tokens),
            func.sum(LLMCallRow.output_tokens),
            func.sum(LLMCallRow.cost_usd),
        )
        .where(LLMCallRow.user_id == user_id)
        .group_by(LLMCallRow.purpose, LLMCallRow.model)
        .order_by(LLMCallRow.purpose, LLMCallRow.model)
    )
    if book_id is not None:
        query = query.where(LLMCallRow.book_id == book_id)
    lines = [
        UsageLine(
            purpose=purpose,
            model=model,
            calls=calls,
            cache_hits=hits or 0,
            input_tokens=inp or 0,
            cached_input_tokens=cached or 0,
            output_tokens=out or 0,
            cost_usd=Decimal(cost or 0),
        )
        for purpose, model, calls, hits, inp, cached, out, cost in await session.execute(query)
    ]
    return UsageSummary(
        lines=lines, total_cost_usd=sum((line.cost_usd for line in lines), Decimal(0))
    )
