from decimal import Decimal

import pytest
from pydantic import BaseModel

from tests.fakes import FakeBackend, MemoryCallStore
from webfic.llm.base import LLMError, Tier
from webfic.llm.client import JsonLLMClient, TierConfig


class Answer(BaseModel):
    value: int


def make_client(backend, store=None, max_retries=2):
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("deepseek-v4-pro")}
    return JsonLLMClient(backend, tiers, store=store, max_retries=max_retries)


async def ask(client, user="q"):
    return await client.generate_json(
        tier=Tier.EXTRACT, system="sys", user=user, schema=Answer, purpose="test"
    )


async def test_valid_json_is_parsed_and_costed():
    client = make_client(FakeBackend(lambda m: {"value": 3}))
    result = await ask(client)
    assert result.data.value == 3
    assert result.model == "deepseek-flash"
    assert result.cost_usd > Decimal(0)
    assert not result.cache_hit


async def test_markdown_fenced_json_is_accepted():
    client = make_client(FakeBackend(lambda m: '```json\n{"value": 5}\n```'))
    assert (await ask(client)).data.value == 5


async def test_invalid_output_is_retried_with_error_feedback():
    answers = iter(['{"value": "not a number"}', '{"value": 7}'])
    backend = FakeBackend(lambda m: next(answers))
    result = await ask(make_client(backend))

    assert result.data.value == 7
    assert len(backend.calls) == 2
    retry_messages = backend.calls[1]
    assert retry_messages[-2].role == "assistant"
    assert "校验错误" in retry_messages[-1].content


async def test_gives_up_after_max_retries():
    backend = FakeBackend(lambda m: "not json at all")
    with pytest.raises(LLMError):
        await ask(make_client(backend, max_retries=1))
    assert len(backend.calls) == 2


async def test_identical_request_is_served_from_cache():
    store = MemoryCallStore()
    backend = FakeBackend(lambda m: {"value": 1})
    client = make_client(backend, store)

    await ask(client)
    second = await ask(client)
    other = await ask(client, user="different question")

    assert len(backend.calls) == 2  # first and "different question"
    assert second.cache_hit and second.cost_usd == 0
    assert not other.cache_hit
    assert [r.cache_hit for r in store.records] == [False, True, False]


async def test_failed_responses_are_not_cached():
    store = MemoryCallStore()
    answers = iter(["bad", '{"value": 2}', '{"value": 2}'])
    backend = FakeBackend(lambda m: next(answers))
    client = make_client(backend, store)

    await ask(client)
    await ask(client)
    # Second run: the first attempt's hash only has a failed record, so it is retried
    # for real, and that attempt now succeeds.
    assert len(backend.calls) == 3


# --- function calling (step 4-1) ------------------------------------------------------------


async def test_openai_compatible_tool_calls_round_trip():
    from openai.types.chat import ChatCompletion

    from webfic.llm.base import ChatMessage, ToolCall, ToolSpec
    from webfic.llm.openai_compat import OpenAICompatBackend

    sent = {}

    class Completions:
        async def create(self, **kwargs):
            sent.update(kwargs)
            return ChatCompletion.model_validate({
                "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-v4-pro",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "reasoning_content": "先读原文",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {
                        "name": "read_passage", "arguments": '{"chapter": 2}'}}]}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57,
                          "prompt_cache_hit_tokens": 40},
            })  # fmt: skip

    backend = OpenAICompatBackend(base_url="https://example.invalid", api_key="k")
    backend._client = type(
        "Client", (), {"chat": type("Chat", (), {"completions": Completions()})()}
    )()
    history = [
        ChatMessage("user", "查一下"),
        ChatMessage(
            "assistant", "", tool_calls=[ToolCall("c0", "search_text", '{"query": "林远"}')]
        ),
        ChatMessage("tool", "没有结果", tool_call_id="c0"),
    ]
    tool = ToolSpec("read_passage", "读原文", {"type": "object", "properties": {}})
    result = await backend.chat(
        model="deepseek-v4-pro", messages=history, json_mode=False, tools=[tool]
    )

    assert "response_format" not in sent
    function = {"name": "read_passage", "description": "读原文", "parameters": tool.parameters}
    assert sent["tools"] == [{"type": "function", "function": function}]
    assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == "search_text"
    assert sent["messages"][2] == {"role": "tool", "content": "没有结果", "tool_call_id": "c0"}
    assert result.tool_calls == [ToolCall("c1", "read_passage", '{"chapter": 2}')]
    assert result.reasoning == "先读原文"
    assert (result.usage.input_tokens, result.usage.cached_input_tokens) == (50, 40)
