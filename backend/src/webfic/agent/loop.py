"""The agent loop: native function calling, ReAct style (think -> call tools -> read
the results -> think again) until the agent submits its answer through the terminal
tool, or a limit is reached.

Everything the model gets wrong is sent back to it as a tool result instead of raising:
arguments that do not validate, unknown tools, "not found" answers, a guard that rejects
the submitted answer. A run ends without an answer when the budget is used up, the guard
keeps rejecting, or the provider refuses the request (moderation and the like); an
authentication error stops the whole job, as in extraction.

Working memory: every turn resends the conversation, and old tool results pile up —
the model has to find its way through them, and they cost tokens every turn. Results
older than the last `keep_turns` turns are folded into a one-line note (tools are
read-only, so the model can simply ask again). Each turn the model is also told how
many turns it has used, and to conclude when few are left.
"""

import json
import time
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from webfic.agent.budget import Budget, Spend
from webfic.agent.tools import RunState, ToolContext, ToolRegistry, to_text
from webfic.agent.trace import TraceWriter
from webfic.llm.base import (
    ChatMessage,
    LLMClient,
    ProviderError,
    ProviderErrorKind,
    Tier,
    ToolCall,
)
from webfic.services.errors import InvalidEdit, NotFound

# Rejections of the submitted answer before the run gives up.
MAX_GUARD_REJECTIONS = 2

NUDGE = "请继续：调用工具查证，或调用 {terminal} 提交结论。"
# Tool results shorter than this are left alone when folding (errors, short answers).
FOLD_MIN_CHARS = 200
# Warn the model when this many turns or less are left.
WARN_TURNS_LEFT = 2

Status = Literal["done", "budget_exhausted", "guard_failed", "failed"]


class AgentResult(BaseModel):
    run_id: Any  # uuid of the agent_runs row
    status: Status
    result: dict[str, Any] | None  # the terminal tool's accepted arguments
    error: str | None = None
    spend: Spend


async def run_agent(
    llm: LLMClient,
    *,
    agent: str,
    system: str,
    task: str,
    tools: ToolRegistry,
    context: ToolContext,
    trace: TraceWriter,
    budget: Budget | None = None,
    tier: Tier = Tier.REASON,
    subject: str | None = None,
    config: dict[str, Any] | None = None,
    keep_turns: int | None = 2,
    progress_note: bool = True,
) -> AgentResult:
    """`keep_turns`: tool results of the last this many turns stay whole, older ones are
    folded (None: never fold). `progress_note`: tell the model its budget use each turn."""
    budget = budget or Budget()
    terminal = next(n for n in tools.names() if tools.get(n).terminal)  # type: ignore[union-attr]
    await trace.start(
        agent=agent, subject=subject,
        config={
            **(config or {}), "tier": str(tier), "budget": budget.model_dump(mode="json"),
            "keep_turns": keep_turns, "progress_note": progress_note,
        },
    )  # fmt: skip
    messages = [ChatMessage("system", system), ChatMessage("user", task)]
    labels: dict[str, str] = {}  # tool call id -> what it was, for folded results
    spend = Spend()
    state = RunState(tools_used=[])
    rejections = 0
    status: Status | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    try:
        while status is None:
            if why := spend.exceeded(budget):
                status, error = "budget_exhausted", why
                break
            visible, folded = fold(messages, keep_turns, labels)
            if progress_note and spend.turns:
                visible.append(ChatMessage("user", _progress(spend, budget, terminal)))
            started = time.monotonic()
            try:
                turn = await llm.chat(
                    tier=tier, messages=visible, tools=tools.specs(), purpose=f"agent.{agent}"
                )
            except ProviderError as exc:
                await trace.step(kind="llm", name="error", input=None, output=None, error=str(exc))
                if exc.kind == ProviderErrorKind.AUTH:
                    await _finish(trace, "failed", None, str(exc), spend)
                    raise
                status, error = "failed", str(exc)
                break
            spend.add_turn(turn.usage, turn.cost_usd)
            await trace.step(
                kind="llm", name=turn.model,
                input={"messages": len(visible), "folded": folded},
                output={
                    "text": turn.text, "reasoning": turn.reasoning,
                    "tool_calls": [
                        {"name": c.name, "arguments": c.arguments} for c in turn.tool_calls
                    ],
                    "cache_hit": turn.cache_hit,
                },
                input_tokens=turn.usage.input_tokens, output_tokens=turn.usage.output_tokens,
                cost_usd=turn.cost_usd, latency_ms=int((time.monotonic() - started) * 1000),
            )  # fmt: skip
            messages.append(ChatMessage("assistant", turn.text, tool_calls=turn.tool_calls))
            if not turn.tool_calls:
                messages.append(ChatMessage("user", NUDGE.format(terminal=terminal)))
                continue

            for call in turn.tool_calls:
                spend.tool_calls += 1
                labels[call.id] = f"{call.name} {call.arguments[:80]}"
                outcome = await _call(call, tools, context, state, trace)
                messages.append(ChatMessage("tool", outcome.reply, tool_call_id=call.id))
                if outcome.accepted is not None:
                    status, result = "done", outcome.accepted
                    break
                if outcome.rejected:
                    rejections += 1
                    if rejections > MAX_GUARD_REJECTIONS:
                        status, error = "guard_failed", outcome.reply
                        break

    except ProviderError:
        raise  # an authentication error, already recorded above
    except Exception as exc:
        # A bug, not the model's doing: record where the run stopped, then let it
        # surface instead of leaving the run 'running' forever (step 4.6).
        why = f"{type(exc).__name__}: {exc}"
        await trace.step(kind="error", name=type(exc).__name__, input=None, output=None,
                         error=why)  # fmt: skip
        await _finish(trace, "failed", None, why, spend)
        raise

    await _finish(trace, status, result, error, spend)
    return AgentResult(run_id=trace.run_id, status=status, result=result, error=error, spend=spend)


def fold(
    messages: list[ChatMessage], keep_turns: int | None, labels: dict[str, str]
) -> tuple[list[ChatMessage], int]:
    """The conversation as the model is shown it: tool results from before the last
    `keep_turns` model turns replaced by a note. Returns the messages and how many were
    folded; `messages` itself is not changed."""
    turns = [i for i, m in enumerate(messages) if m.role == "assistant"]
    if keep_turns is None or len(turns) <= keep_turns:
        return list(messages), 0
    cutoff = turns[-keep_turns]
    shown, folded = [], 0
    for i, m in enumerate(messages):
        if i < cutoff and m.role == "tool" and len(m.content) > FOLD_MIN_CHARS:
            label = labels.get(m.tool_call_id or "", "工具调用")
            note = f"〔已折叠：{label} 的结果，原 {len(m.content)} 字；需要时可以重新调用〕"
            shown.append(ChatMessage("tool", note, tool_call_id=m.tool_call_id))
            folded += 1
        else:
            shown.append(m)
    return shown, folded


def _progress(spend: Spend, budget: Budget, terminal: str) -> str:
    # Turns only: spend in money differs between a real call and a cached replay, and a
    # note that differs would make every replayed request miss the cache.
    note = f"〔进度：已用 {spend.turns}/{budget.max_turns} 轮。〕"
    if budget.max_turns - spend.turns <= WARN_TURNS_LEFT:
        note += f"剩余轮数不多，请根据已有证据调用 {terminal} 提交结论。"
    return note


class _Outcome(BaseModel):
    reply: str  # what the model is told
    accepted: dict[str, Any] | None = None  # the terminal tool's answer, if accepted
    rejected: bool = False  # the guard turned the answer down


async def _call(
    call: ToolCall, tools: ToolRegistry, context: ToolContext, state: RunState, trace: TraceWriter
) -> _Outcome:
    started = time.monotonic()
    tool = tools.get(call.name)
    error: str | None = None
    outcome: _Outcome
    try:
        if tool is None:
            raise _Refused(f"没有名为 {call.name} 的工具。可用的工具：{'、'.join(tools.names())}")
        try:
            args = tool.args.model_validate(json.loads(call.arguments or "{}"))
        except (ValidationError, json.JSONDecodeError) as exc:
            raise _Refused(f"参数不合法：{_brief(exc)}。请按工具说明重新调用。") from exc

        if tool.terminal:
            problem = await tool.guard(context, args, state) if tool.guard else None
            if problem is not None:
                await trace.step(
                    kind="guard", name=tool.name, input=_json(call.arguments), output=None,
                    error=problem,
                )  # fmt: skip
                return _Outcome(reply=f"结论未通过检查：{problem}请修改后重新提交。", rejected=True)
            outcome = _Outcome(reply="已收到结论。", accepted=args.model_dump(mode="json"))
        else:
            assert tool.run is not None
            try:
                value = await tool.run(context, args)
            except (NotFound, InvalidEdit, ValueError) as exc:
                raise _Refused(f"查询失败：{exc}") from exc
            state.tools_used.append(tool.name)
            outcome = _Outcome(reply=to_text(value))
    except _Refused as exc:
        error = str(exc)
        outcome = _Outcome(reply=error)
    await trace.step(
        kind="tool", name=call.name, input=_json(call.arguments),
        output=outcome.accepted if outcome.accepted is not None else outcome.reply,
        error=error, latency_ms=int((time.monotonic() - started) * 1000),
    )  # fmt: skip
    return outcome


class _Refused(Exception):
    """A tool call that could not run; the message goes back to the model."""


def _brief(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "；".join(
            f"{'.'.join(str(p) for p in e['loc']) or '参数'}：{e['msg']}" for e in exc.errors()
        )
    return str(exc)


def _json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


async def _finish(
    trace: TraceWriter,
    status: str,
    result: dict[str, Any] | None,
    error: str | None,
    spend: Spend,
) -> None:
    await trace.finish(
        status=status, result=result, error=error, turns=spend.turns,
        tool_calls=spend.tool_calls, input_tokens=spend.input_tokens,
        output_tokens=spend.output_tokens, cost_usd=spend.cost_usd,
    )  # fmt: skip
