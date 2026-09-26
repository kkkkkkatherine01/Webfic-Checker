"""Execution record of agent runs (agent_runs / agent_steps).

A run is a trace, each model call, tool call and guard check a span under the run's
root span; ids have OpenTelemetry's sizes (16-byte trace id, 8-byte span ids) so runs
can be exported to an observability platform later. Every step is committed as it
happens, so a run that crashes still shows how far it got.
"""

import secrets
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.db.models import AgentRunRow, AgentStepRow

# Step inputs and outputs are stored for people to read, not for replay; long ones are cut.
MAX_STORED_CHARS = 20_000


def _span_id() -> str:
    return secrets.token_hex(8)


def _cut(value: Any) -> Any:
    if isinstance(value, str) and len(value) > MAX_STORED_CHARS:
        return value[:MAX_STORED_CHARS] + f"……（已截去 {len(value) - MAX_STORED_CHARS} 字）"
    if isinstance(value, dict):
        return {k: _cut(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cut(v) for v in value]
    return value


class TraceWriter:
    def __init__(
        self,
        factory: async_sessionmaker[AsyncSession],
        *,
        user_id: uuid.UUID,
        book_id: uuid.UUID,
    ):
        self._factory = factory
        self._user_id = user_id
        self._book_id = book_id
        self.run_id: uuid.UUID | None = None
        self._span_id = ""
        self._seq = 0

    async def start(self, *, agent: str, subject: str | None, config: dict[str, Any]) -> None:
        self.run_id = uuid.uuid4()
        self._span_id = _span_id()
        async with self._factory() as session:
            session.add(
                AgentRunRow(
                    id=self.run_id, user_id=self._user_id, book_id=self._book_id,
                    agent=agent, subject=subject, trace_id=uuid.uuid4().hex,
                    span_id=self._span_id, config=config, status="running",
                )
            )  # fmt: skip
            await session.commit()

    async def step(
        self,
        *,
        kind: str,
        name: str,
        input: Any,
        output: Any,
        error: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Decimal = Decimal(0),
        latency_ms: int = 0,
    ) -> None:
        assert self.run_id is not None, "start() first"
        self._seq += 1
        async with self._factory() as session:
            session.add(
                AgentStepRow(
                    user_id=self._user_id, book_id=self._book_id, run_id=self.run_id,
                    seq=self._seq, span_id=_span_id(), parent_span_id=self._span_id,
                    kind=kind, name=name[:64], input=_cut(input), output=_cut(output),
                    error=error, input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=cost_usd, latency_ms=latency_ms,
                )
            )  # fmt: skip
            await session.commit()

    async def finish(
        self,
        *,
        status: str,
        result: dict[str, Any] | None,
        error: str | None,
        turns: int,
        tool_calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_usd: Decimal,
    ) -> None:
        assert self.run_id is not None, "start() first"
        async with self._factory() as session:
            await session.execute(
                update(AgentRunRow)
                .where(AgentRunRow.user_id == self._user_id, AgentRunRow.id == self.run_id)
                .values(
                    status=status, result=result, error=error, turns=turns,
                    tool_calls=tool_calls, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost_usd=cost_usd,
                    ended_at=datetime.now(UTC),
                )
            )  # fmt: skip
            await session.commit()
