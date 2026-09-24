"""CallStore backed by the llm_calls table. May move to Redis before launch; the table
stays as the usage ledger."""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.db.models import LLMCallRow
from webfic.llm.client import CallRecord


class DbCallStore:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        user_id: uuid.UUID | None,
        book_id: uuid.UUID | None = None,
    ):
        self._session_factory = session_factory
        self._user_id = user_id
        self._book_id = book_id

    async def lookup(self, request_hash: str) -> str | None:
        # The cache is keyed on the full request, so a hit can only return what the
        # same input already produced; sharing it across users leaks nothing.
        async with self._session_factory() as session:
            return await session.scalar(
                select(LLMCallRow.response_text)
                .where(LLMCallRow.request_hash == request_hash, LLMCallRow.ok.is_(True))
                .limit(1)
            )

    async def record(self, record: CallRecord) -> None:
        async with self._session_factory() as session:
            session.add(
                LLMCallRow(
                    user_id=self._user_id,
                    book_id=self._book_id,
                    purpose=record.purpose,
                    provider=record.provider,
                    model=record.model,
                    request_hash=record.request_hash,
                    response_text=record.response_text,
                    input_tokens=record.usage.input_tokens,
                    cached_input_tokens=record.usage.cached_input_tokens,
                    output_tokens=record.usage.output_tokens,
                    cost_usd=record.cost_usd,
                    latency_ms=record.latency_ms,
                    cache_hit=record.cache_hit,
                    ok=record.ok,
                )
            )
            await session.commit()
