import json
from collections.abc import Callable
from typing import Any

from webfic.llm.base import ChatMessage, RawCompletion, Tier, Usage
from webfic.llm.cache import DbCallStore
from webfic.llm.client import CallRecord, JsonLLMClient, TierConfig


def make_llm(backend, factory, *, user_id, book_id) -> JsonLLMClient:
    tiers = {
        Tier.EXTRACT: TierConfig("deepseek-flash"),
        Tier.REASON: TierConfig("deepseek-v4-pro"),
        Tier.VERIFY: TierConfig("deepseek-flash"),
    }
    store = DbCallStore(factory, user_id=user_id, book_id=book_id)
    return JsonLLMClient(backend, tiers, store=store)


class FakeBackend:
    """Answers with `respond(messages)`; counts real (non-cached) calls."""

    provider = "fake"

    def __init__(self, respond: Callable[[list[ChatMessage]], str | dict[str, Any]]):
        self._respond = respond
        self.calls: list[list[ChatMessage]] = []

    async def chat(
        self, *, model, messages, json_mode, extra=None, temperature=None, tools=None
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


class HashEmbedder:
    """Deterministic stand-in for the embedding model: hashed character bigrams. Texts
    that share wording get similar vectors; nothing is downloaded."""

    dim = 512

    def _vector(self, text: str) -> list[float]:
        import hashlib
        import math
        from itertools import pairwise

        vector = [0.0] * self.dim
        for a, b in pairwise(text):
            bucket = int(hashlib.md5((a + b).encode()).hexdigest()[:8], 16) % self.dim
            vector[bucket] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def make_archival(tmp_path, size: int = 300, overlap: int = 60):
    from webfic.archival.index import Archival
    from webfic.archival.tokenize import Tokenizer

    return Archival(
        embedder=HashEmbedder(), tokenizer=Tokenizer(tmp_path), embed_model="hash",
        passage_size=size, passage_overlap=overlap,
    )  # fmt: skip


class ScriptedBackend:
    """A fake model for agents. Each turn replies with the next scripted item: a list of
    (tool name, arguments) calls, plain text, or an exception to raise. A callable item
    gets the messages so far and returns one of those."""

    provider = "fake"

    def __init__(self, turns):
        self._turns = iter(turns)
        self.calls: list[list[ChatMessage]] = []
        self.tools_seen = []

    async def chat(
        self, *, model, messages, json_mode, extra=None, temperature=None, tools=None
    ) -> RawCompletion:
        from webfic.llm.base import ToolCall

        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        turn = next(self._turns)
        if callable(turn):
            turn = turn(messages)
        if isinstance(turn, Exception):
            raise turn
        usage = Usage(1000, 200, 100)
        if isinstance(turn, str):
            return RawCompletion(text=turn, usage=usage, model=model)
        calls = [
            ToolCall(
                id=f"call_{len(self.calls)}_{i}",
                name=name,
                arguments=args if isinstance(args, str) else json.dumps(args, ensure_ascii=False),
            )
            for i, (name, args) in enumerate(turn)
        ]
        return RawCompletion(
            text="", usage=usage, model=model, tool_calls=calls, reasoning="先查证"
        )


def agent_llm(backend, store=None) -> JsonLLMClient:
    tiers = {
        Tier.EXTRACT: TierConfig("deepseek-flash"),
        Tier.REASON: TierConfig("deepseek-v4-pro"),
        Tier.VERIFY: TierConfig("deepseek-flash"),
    }
    return JsonLLMClient(backend, tiers, store=store or MemoryCallStore())
