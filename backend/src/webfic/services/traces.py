"""Read agent execution records: what a run did, step by step."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.db.models import AgentRunRow, AgentStepRow, Book
from webfic.services.errors import NotFound


class StepView(BaseModel):
    seq: int
    kind: str  # llm / tool / guard
    name: str
    input: Any
    output: Any
    error: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    latency_ms: int


class RunView(BaseModel):
    id: uuid.UUID
    book_id: uuid.UUID
    agent: str
    subject: str | None
    status: str
    config: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    turns: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    created_at: datetime
    ended_at: datetime | None
    steps: list[StepView] = []


def _view(row: AgentRunRow) -> RunView:
    return RunView(
        id=row.id, book_id=row.book_id, agent=row.agent, subject=row.subject,
        status=row.status, config=row.config, result=row.result, error=row.error,
        turns=row.turns, tool_calls=row.tool_calls, input_tokens=row.input_tokens,
        output_tokens=row.output_tokens, cost_usd=row.cost_usd, created_at=row.created_at,
        ended_at=row.ended_at,
    )  # fmt: skip


async def get_run(session: AsyncSession, *, user_id: uuid.UUID, run: str) -> RunView:
    """A run with all its steps, by id or a unique prefix of it."""
    wanted = run.strip().lower()
    try:
        exact = uuid.UUID(wanted)
    except ValueError:
        exact = None
    if exact is not None:
        rows = [
            r
            for r in await session.scalars(
                select(AgentRunRow).where(AgentRunRow.user_id == user_id, AgentRunRow.id == exact)
            )
        ]
    else:
        ids = await session.scalars(select(AgentRunRow.id).where(AgentRunRow.user_id == user_id))
        matches = [i for i in ids if str(i).startswith(wanted)][:2]
        rows = [
            r for r in await session.scalars(select(AgentRunRow).where(AgentRunRow.id.in_(matches)))
        ]
    if len(rows) != 1:
        raise NotFound(f"找不到唯一匹配「{run}」的 agent 运行记录（匹配到 {len(rows)} 条）")
    view = _view(rows[0])
    steps = await session.scalars(
        select(AgentStepRow)
        .where(AgentStepRow.user_id == user_id, AgentStepRow.run_id == view.id)
        .order_by(AgentStepRow.seq)
    )
    view.steps = [
        StepView(
            seq=s.seq,
            kind=s.kind,
            name=s.name,
            input=s.input,
            output=s.output,
            error=s.error,
            input_tokens=s.input_tokens,
            output_tokens=s.output_tokens,
            cost_usd=s.cost_usd,
            latency_ms=s.latency_ms,
        )
        for s in steps
    ]
    return view


async def list_runs(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, limit: int = 20
) -> list[RunView]:
    """The book's most recent runs, newest first (without steps)."""
    owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id)
    if await session.scalar(owned) is None:
        raise NotFound(f"book {book_id}")
    rows = await session.scalars(
        select(AgentRunRow)
        .where(AgentRunRow.user_id == user_id, AgentRunRow.book_id == book_id)
        .order_by(AgentRunRow.created_at.desc())
        .limit(limit)
    )
    return [_view(r) for r in rows]
