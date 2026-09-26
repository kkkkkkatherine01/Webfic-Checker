"""Provider-neutral LLM interface. Business code depends only on this module."""

from dataclasses import dataclass, field
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


@dataclass
class ToolCall:
    """A tool the model asked to run. `arguments` is the model's JSON text, unchecked."""

    id: str
    name: str
    arguments: str


@dataclass
class ToolSpec:
    """A tool offered to the model: `parameters` is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)  # assistant: tools it called
    tool_call_id: str | None = None  # tool: which call this is the result of


@dataclass
class ChatTurn:
    """One reply of the model in a tool-using conversation."""

    text: str
    tool_calls: list[ToolCall]
    # The model's thinking, when the provider returns it (DeepSeek thinking mode). Kept
    # for the execution record only: it is not sent back, which providers do not need.
    reasoning: str | None
    usage: Usage
    cost_usd: Decimal
    model: str
    cache_hit: bool


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

    async def chat(
        self,
        *,
        tier: Tier,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        purpose: str,
    ) -> ChatTurn:
        """One model turn with native function calling (agents). Cached and recorded
        like `generate_json`; tool arguments are returned unchecked."""
        ...


# --- lower-level pieces used by the implementation -------------------------------------


@dataclass
class RawCompletion:
    text: str
    usage: Usage
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str | None = None


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
        tools: list[ToolSpec] | None = None,
    ) -> RawCompletion: ...
