"""The LLMClient implementation shared by all providers: request caching, usage
recording, JSON parsing, schema validation and retry-with-feedback."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from webfic.llm.base import ChatBackend, ChatMessage, LLMError, LLMResult, Tier, Usage
from webfic.llm.pricing import cost_of


@dataclass
class CallRecord:
    purpose: str
    provider: str
    model: str
    request_hash: str
    response_text: str
    usage: Usage
    cost_usd: Decimal
    latency_ms: int
    cache_hit: bool
    ok: bool


class CallStore(Protocol):
    """Persists every call (usage ledger) and serves cached responses."""

    async def lookup(self, request_hash: str) -> str | None: ...

    async def record(self, record: CallRecord) -> None: ...


class NullCallStore:
    async def lookup(self, request_hash: str) -> str | None:
        return None

    async def record(self, record: CallRecord) -> None:
        return None


@dataclass
class TierConfig:
    model: str
    extra: dict[str, Any] = field(default_factory=dict)
    temperature: float | None = None


def _request_hash(
    provider: str, tier: TierConfig, messages: list[ChatMessage], schema: type[BaseModel]
) -> str:
    payload = {
        "provider": provider,
        "model": tier.model,
        "extra": tier.extra,
        "temperature": tier.temperature,
        "messages": [[m.role, m.content] for m in messages],
        "schema": schema.model_json_schema(),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _parse[T: BaseModel](text: str, schema: type[T]) -> T:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Some models wrap JSON in a markdown fence even in JSON mode.
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()
    return schema.model_validate_json(cleaned)


class JsonLLMClient:
    def __init__(
        self,
        backend: ChatBackend,
        tiers: dict[Tier, TierConfig],
        store: CallStore | None = None,
        max_retries: int = 2,
    ):
        self._backend = backend
        self._tiers = tiers
        self._store = store or NullCallStore()
        self._max_retries = max_retries

    async def generate_json[T: BaseModel](
        self,
        *,
        tier: Tier,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
    ) -> LLMResult[T]:
        tier_config = self._tiers[tier]
        provider = self._backend.provider
        messages = [ChatMessage("system", system), ChatMessage("user", user)]

        total_usage = Usage()
        total_cost = Decimal(0)
        all_cached = True
        last_error = ""

        for _attempt in range(self._max_retries + 1):
            request_hash = _request_hash(provider, tier_config, messages, schema)
            cached_text = await self._store.lookup(request_hash)

            if cached_text is not None:
                text, usage, cost, latency_ms = cached_text, Usage(), Decimal(0), 0
                model = tier_config.model
            else:
                all_cached = False
                started = time.monotonic()
                completion = await self._backend.chat(
                    model=tier_config.model,
                    messages=messages,
                    json_mode=True,
                    extra=tier_config.extra,
                    temperature=tier_config.temperature,
                )
                latency_ms = int((time.monotonic() - started) * 1000)
                text, usage, model = completion.text, completion.usage, completion.model
                cost = cost_of(tier_config.model, usage)

            total_usage += usage
            total_cost += cost

            try:
                data = _parse(text, schema)
                ok = True
            except (ValidationError, ValueError) as exc:
                data, ok = None, False
                last_error = str(exc)

            await self._store.record(
                CallRecord(
                    purpose=purpose,
                    provider=provider,
                    model=model,
                    request_hash=request_hash,
                    response_text=text,
                    usage=usage,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    cache_hit=cached_text is not None,
                    ok=ok,
                )
            )

            if data is not None:
                return LLMResult(
                    data=data,
                    usage=total_usage,
                    provider=provider,
                    model=model,
                    cost_usd=total_cost,
                    cache_hit=all_cached,
                )

            messages = [
                *messages,
                ChatMessage("assistant", text),
                ChatMessage(
                    "user",
                    f"上面的输出不是符合要求的 JSON，校验错误：\n{last_error[:1000]}\n"
                    "请重新输出完整、合法的 JSON，不要附加任何其他文字。",
                ),
            ]

        raise LLMError(
            f"{purpose}: no valid JSON after {self._max_retries + 1} attempts: {last_error}"
        )
