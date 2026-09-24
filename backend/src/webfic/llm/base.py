"""Provider-neutral LLM interface. Business code depends only on this module."""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel


class Tier(StrEnum):
    EXTRACT = "extract"  # cheap model: extraction, consolidation
    REASON = "reason"  # stronger model: semantic checkers


@dataclass
class Usage:
    input_tokens: int = 0  # all prompt tokens, including cached ones
    cached_input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass
class LLMResult[T: BaseModel]:
    data: T
    usage: Usage
    provider: str
    model: str
    cost_usd: Decimal
    cache_hit: bool


class LLMError(Exception):
    """The model could not produce a valid response after retries."""


class ProviderErrorKind(StrEnum):
    CONTENT_FILTER = "content_filter"  # the provider refused the text (moderation)
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    NETWORK = "network"
    AUTH = "auth"  # bad or unfunded key: every further call will fail too
    BAD_REQUEST = "bad_request"


class ProviderError(LLMError):
    """A provider-side failure, translated by the protocol adapter so business code
    never handles vendor SDK exceptions."""

    def __init__(self, kind: ProviderErrorKind, message: str):
        super().__init__(f"[{kind}] {message}")
        self.kind = kind
        self.message = message


class LLMClient(Protocol):
    async def generate_json[T: BaseModel](
        self,
        *,
        tier: Tier,
        system: str,
        user: str,
        schema: type[T],
        purpose: str,
    ) -> LLMResult[T]:
        """`system` should be stable across calls (instructions, few-shot) so providers
        can reuse their prefix cache; put per-call content in `user`."""
        ...


# --- lower-level pieces used by the implementation -------------------------------------


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class RawCompletion:
    text: str
    usage: Usage
    model: str


class ChatBackend(Protocol):
    """One protocol adapter (OpenAI-compatible, Anthropic...). No retries, no parsing."""

    provider: str

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        json_mode: bool,
        extra: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> RawCompletion: ...
