"""The agent harness (step 4-1): tool registry, the function-calling loop, budgets,
guards, provider errors, the execution record and data isolation. The model is scripted;
the tools are real memory queries over a book imported with the fake extraction model."""

import asyncio
import uuid
from decimal import Decimal

import pytest
from pydantic import BaseModel, Field
from sqlalchemy import select
from typer.testing import CliRunner

from tests.fakes import MemoryCallStore, ScriptedBackend, agent_llm
from tests.test_pipeline import USER, import_book
from tests.test_pipeline import respond as extraction_respond
from webfic.agent.budget import Budget
from webfic.agent.loop import run_agent
from webfic.agent.tools import (
    MAX_RESULT_CHARS,
    Tool,
    ToolArgs,
    ToolContext,
    ToolRegistry,
    json_schema,
    to_text,
)
from webfic.agent.trace import TraceWriter
from webfic.db.models import AgentRunRow, AgentStepRow, Base
from webfic.db.session import make_engine, make_session_factory
from webfic.llm.base import ProviderError, ProviderErrorKind
from webfic.memory import recall
from webfic.services import traces
from webfic.services.errors import NotFound

# --- a small agent: look facts up, then answer --------------------------------------------


class FactsArgs(ToolArgs):
    character: str | None = None


class Quote(BaseModel):
    chapter: int
    quote: str


class AnswerArgs(ToolArgs):
    answer: str
    quotes: list[Quote] = Field(default_factory=list)


async def list_facts(ctx: ToolContext, args: FactsArgs):
    async with ctx.factory() as session:
        return await recall.list_facts(
            session, user_id=ctx.user_id, book_id=ctx.book_id, character=args.character
        )


async def must_have_looked(ctx, args, state):
    return None if "list_facts" in state.tools_used else "还没有查过任何事实。"


def registry(guard=must_have_looked):
    return ToolRegistry(
        [
            Tool("list_facts", "列出角色的年龄事实", FactsArgs, run=list_facts),
            Tool("answer", "提交答案", AnswerArgs, terminal=True, guard=guard),
        ]
    )


LOOK = [("list_facts", {"character": "林远"})]
ANSWER = [
    ("answer", {"answer": "林远十八岁", "quotes": [{"chapter": 1, "quote": "林远今年十八岁"}]})
]


async def book(factory):
    job, _, _ = await import_book(factory, _extraction_backend())
    return job.book_id


def _extraction_backend():
    from tests.fakes import FakeBackend

    return FakeBackend(extraction_respond)


async def run(
    factory, book_id, turns, *, user_id=USER, budget=None, store=None, guard=must_have_looked
):
    backend = ScriptedBackend(turns)
    context = ToolContext(factory=factory, user_id=user_id, book_id=book_id)
    result = await run_agent(
        agent_llm(backend, store), agent="test", system="你是助手。", task="林远多大？",
        tools=registry(guard), context=context,
        trace=TraceWriter(factory, user_id=user_id, book_id=book_id),
        budget=budget, subject="s1",
    )  # fmt: skip
    return result, backend


async def steps(factory, run_id):
    async with factory() as session:
        return (
            await session.scalars(
                select(AgentStepRow).where(AgentStepRow.run_id == run_id).order_by(AgentStepRow.seq)
            )
        ).all()


def tool_replies(backend, call_index):
    """The tool results the model saw in its `call_index`-th request."""
    return [m.content for m in backend.calls[call_index] if m.role == "tool"]


# --- registry ------------------------------------------------------------------------------


def test_tool_schemas_are_plain_and_never_ask_for_the_user_or_book():
    schema = json_schema(AnswerArgs)
    assert "$defs" not in schema and "title" not in str(schema)
    assert schema["properties"]["quotes"]["items"]["properties"]["chapter"]["type"] == "integer"
    specs = {s.name: s for s in registry().specs()}
    assert set(specs) == {"list_facts", "answer"}
    assert all("user_id" not in str(s.parameters) and "book_id" not in str(s.parameters)
               for s in specs.values())  # fmt: skip


def test_a_registry_needs_a_terminal_tool():
    with pytest.raises(ValueError):
        ToolRegistry([Tool("list_facts", "x", FactsArgs, run=list_facts)])


def test_long_results_are_cut():
    text = to_text("字" * (MAX_RESULT_CHARS + 50))
    assert len(text) < MAX_RESULT_CHARS + 40 and "已截去 50 字" in text


# --- the loop --------------------------------------------------------------------------------


async def test_a_run_looks_up_then_answers_and_every_step_is_recorded(factory):
    book_id = await book(factory)
    result, backend = await run(factory, book_id, [LOOK, ANSWER])

    assert result.status == "done"
    assert result.result == {
        "answer": "林远十八岁",
        "quotes": [{"chapter": 1, "quote": "林远今年十八岁"}],
    }
    assert (result.spend.turns, result.spend.tool_calls) == (2, 2)
    assert result.spend.cost_usd > 0
    assert "林远今年十八岁" in tool_replies(backend, 1)[0]  # the model read the real facts
    assert [t.name for t in backend.tools_seen[0]] == ["list_facts", "answer"]

    rows = await steps(factory, result.run_id)
    assert [(s.kind, s.name) for s in rows] == [
        ("llm", "deepseek-v4-pro"), ("tool", "list_facts"),
        ("llm", "deepseek-v4-pro"), ("tool", "answer"),
    ]  # fmt: skip
    assert rows[0].output["reasoning"] == "先查证" and rows[0].input_tokens == 1000
    assert len({s.span_id for s in rows}) == 4 and len({s.parent_span_id for s in rows}) == 1
    async with factory() as session:
        run_row = await session.get_one(AgentRunRow, result.run_id)
    assert (run_row.status, run_row.turns, run_row.tool_calls, run_row.subject) == (
        "done",
        2,
        2,
        "s1",
    )
    assert len(run_row.trace_id) == 32 and run_row.ended_at is not None
    assert run_row.config["budget"]["max_turns"] == 10


async def test_mistakes_are_sent_back_to_the_model(factory):
    book_id = await book(factory)
    mistakes = [
        ("list_facts", {"character": 5}),  # wrong type
        ("list_facts", {"character": "林远", "user_id": str(uuid.uuid4())}),  # not an argument
        ("list_facts", "{not json"),
        ("delete_everything", {}),  # no such tool
        ("list_facts", {"character": "无名氏"}),  # nobody of that name
    ]
    result, backend = await run(factory, book_id, [mistakes, LOOK, ANSWER])
    assert result.status == "done"
    replies = tool_replies(backend, 1)
    starts = [
        "参数不合法：",
        "参数不合法：",
        "参数不合法：",
        "没有名为 delete",
        "查询失败：character",
    ]
    assert all(r.startswith(s) for r, s in zip(replies, starts, strict=True)), replies
    assert "user_id" in replies[1] and "list_facts、answer" in replies[3]
    rows = await steps(factory, result.run_id)
    assert sum(1 for s in rows if s.error) == 5


async def test_a_reply_without_tool_calls_gets_a_nudge(factory):
    book_id = await book(factory)
    result, backend = await run(factory, book_id, ["我想想。", LOOK, ANSWER])
    assert result.status == "done" and result.spend.turns == 3
    assert backend.calls[1][-1].role == "user" and "answer" in backend.calls[1][-1].content


async def test_the_guard_sends_an_answer_back_until_it_holds(factory):
    book_id = await book(factory)
    result, backend = await run(factory, book_id, [ANSWER, LOOK, ANSWER])
    assert result.status == "done"
    assert tool_replies(backend, 1)[0].startswith("结论未通过检查：还没有查过任何事实。")
    assert [s.kind for s in await steps(factory, result.run_id)].count("guard") == 1


async def test_an_answer_the_guard_keeps_refusing_ends_the_run_without_one(factory):
    book_id = await book(factory)
    result, _ = await run(factory, book_id, [ANSWER, ANSWER, ANSWER])
    assert (result.status, result.result) == ("guard_failed", None)
    assert "还没有查过任何事实" in result.error


async def test_the_budget_stops_a_run_without_an_answer(factory):
    book_id = await book(factory)
    result, backend = await run(factory, book_id, [LOOK] * 5, budget=Budget(max_turns=3))
    assert (result.status, result.result) == ("budget_exhausted", None)
    assert len(backend.calls) == 3 and "3 轮" in result.error

    result, backend = await run(
        factory, book_id, [LOOK] * 5, budget=Budget(max_cost_usd=Decimal("0.0005"))
    )
    assert result.status == "budget_exhausted" and len(backend.calls) == 1


async def test_a_refused_request_fails_the_run_and_a_bad_key_stops_everything(factory):
    book_id = await book(factory)
    refused = ProviderError(ProviderErrorKind.CONTENT_FILTER, "Content Exists Risk")
    result, _ = await run(factory, book_id, [LOOK, refused])
    assert result.status == "failed" and "content_filter" in result.error

    with pytest.raises(ProviderError):
        await run(factory, book_id, [ProviderError(ProviderErrorKind.AUTH, "Invalid API key")])
    async with factory() as session:
        statuses = list(await session.scalars(select(AgentRunRow.status)))
    assert sorted(statuses) == ["failed", "failed"]


async def test_the_same_run_again_is_served_from_the_cache(factory):
    book_id = await book(factory)
    store = MemoryCallStore()
    first, _ = await run(factory, book_id, [LOOK, ANSWER], store=store)
    again, backend_again = await run(factory, book_id, [], store=store)  # no model left
    assert again.status == "done" and again.result == first.result
    assert backend_again.calls == [] and again.spend.cost_usd == 0
    assert [
        s.output["cache_hit"] for s in await steps(factory, again.run_id) if s.kind == "llm"
    ] == [
        True,
        True,
    ]


async def test_tools_only_see_the_users_own_book(factory):
    book_id = await book(factory)
    stranger = uuid.uuid4()
    result, backend = await run(factory, book_id, [LOOK, ANSWER, ANSWER, ANSWER], user_id=stranger)
    assert tool_replies(backend, 1)[0].startswith("查询失败：book")
    assert result.status == "guard_failed"  # it never managed to look anything up


# --- reading the record ---------------------------------------------------------------------


async def test_runs_can_be_read_back_by_prefix_and_only_by_their_user(factory):
    book_id = await book(factory)
    result, _ = await run(factory, book_id, [LOOK, ANSWER])
    async with factory() as session:
        view = await traces.get_run(session, user_id=USER, run=str(result.run_id)[:6])
        listed = await traces.list_runs(session, user_id=USER, book_id=book_id)
        assert [s.kind for s in view.steps] == ["llm", "tool", "llm", "tool"]
        assert [r.id for r in listed] == [result.run_id]
        with pytest.raises(NotFound):
            await traces.get_run(session, user_id=uuid.uuid4(), run=str(result.run_id))


def test_the_trace_command_prints_a_run(tmp_path, monkeypatch):
    from webfic import cli
    from webfic.config import Settings

    url = f"sqlite+aiosqlite:///{(tmp_path / 'cli.db').as_posix()}"

    async def prepare():
        engine = make_engine(url)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = make_session_factory(engine)
        book_id = await book(factory)
        result, _ = await run(factory, book_id, [LOOK, ANSWER])
        await engine.dispose()
        return result.run_id

    run_id = asyncio.run(prepare())
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(database_url=url, dev_user_id=USER))
    output = CliRunner().invoke(cli.app, ["trace", str(run_id)[:8]])
    assert output.exit_code == 0, output.output
    assert (
        "list_facts" in output.output and "林远十八岁" in output.output and "完成" in output.output
    )
