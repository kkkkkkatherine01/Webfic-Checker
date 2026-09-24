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
