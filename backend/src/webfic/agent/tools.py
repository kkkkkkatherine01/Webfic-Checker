"""Tools an agent may call: a name, a description for the model, a Pydantic model for
the arguments (which also gives the JSON Schema the model sees) and the function that
runs it.

The user and the book are never arguments: the harness passes them in a `ToolContext`,
so a model cannot ask for anyone else's data, and argument models forbid extra fields.
"""

import json
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.archival.index import Archival
from webfic.llm.base import ToolSpec

# Tool results longer than this are cut: a long result costs tokens on every later turn.
MAX_RESULT_CHARS = 4000


class ToolArgs(BaseModel):
    """Base for tool argument models: unknown fields are an error, not ignored."""

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ToolContext:
    factory: async_sessionmaker[AsyncSession]
    user_id: uuid.UUID
    book_id: uuid.UUID
    archival: Archival | None = None


@dataclass
class RunState:
    """What a run has done so far, for guards ("read the text before dismissing")."""

    tools_used: list[str]


# Checks a terminal tool's arguments before they are accepted: None if fine, otherwise the
# reason, which is sent back to the model so it can fix its answer.
Guard = Callable[[ToolContext, Any, RunState], Awaitable[str | None]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[ToolArgs]
    # Runs the tool; returns something JSON-serialisable (Pydantic models included).
    # Not called for a terminal tool, whose accepted arguments are the run's result.
    run: Callable[[ToolContext, Any], Awaitable[Any]] | None = None
    terminal: bool = False
    guard: Guard | None = None

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, json_schema(self.args))


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]):
        self._tools = {t.name: t for t in tools}
        if not any(t.terminal for t in self._tools.values()):
            raise ValueError("an agent needs a terminal tool to submit its answer")

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)


def json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema with references inlined and titles dropped: plain and
    small, which every OpenAI-compatible provider accepts."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return clean(defs[node["$ref"].rsplit("/", 1)[-1]])
            return {k: clean(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node

    return clean(schema)


def to_text(result: Any) -> str:
    """A tool result as the text the model reads, cut at MAX_RESULT_CHARS."""
    text = result if isinstance(result, str) else json.dumps(_plain(result), ensure_ascii=False)
    if len(text) > MAX_RESULT_CHARS:
        cut = len(text) - MAX_RESULT_CHARS
        text = text[:MAX_RESULT_CHARS] + f"……（结果过长，已截去 {cut} 字；请缩小范围再查）"
    return text


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value
