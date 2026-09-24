import json
from collections.abc import Callable
from typing import Any

from webfic.llm.base import ChatMessage, RawCompletion, Tier, Usage
from webfic.llm.cache import DbCallStore
from webfic.llm.client import CallRecord, JsonLLMClient, TierConfig


def make_llm(backend, factory, *, user_id, book_id) -> JsonLLMClient:
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("deepseek-v4-pro")}
    store = DbCallStore(factory, user_id=user_id, book_id=book_id)
    return JsonLLMClient(backend, tiers, store=store)


class FakeBackend:
    """Answers with `respond(messages)`; counts real (non-cached) calls."""

    provider = "fake"

    def __init__(self, respond: Callable[[list[ChatMessage]], str | dict[str, Any]]):
        self._respond = respond
        self.calls: list[list[ChatMessage]] = []

    async def chat(
        self, *, model, messages, json_mode, extra=None, temperature=None
    ) -> RawCompletion:
        self.calls.append(list(messages))
        answer = self._respond(messages)
        text = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return RawCompletion(text=text, usage=Usage(1000, 200, 100), model=model)


class MemoryCallStore:
    def __init__(self):
        self.records: list[CallRecord] = []

    async def lookup(self, request_hash: str) -> str | None:
        for r in self.records:
            if r.request_hash == request_hash and r.ok:
                return r.response_text
        return None

    async def record(self, record: CallRecord) -> None:
        self.records.append(record)
