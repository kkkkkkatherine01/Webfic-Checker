"""The original-text review page (step 5-1c)."""

from tests.fakes import FakeBackend, MemoryCallStore
from tests.test_chapters import USER, World, respond
from tests.test_verify_facts import BOOK, facts
from webfic.evaluation import review
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig


async def test_the_page_lists_each_issue_with_its_context_and_escapes_the_text(factory):
    world = World(factory)
    world.backend = FakeBackend(respond, facts=facts)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    world.llm = JsonLLMClient(world.backend, tiers, store=MemoryCallStore())
    await world.load(BOOK)
    items = await review.collect(factory, {"book<1>": world.book_id}, USER, "character_facts")
    assert len(items) == 1 and [c[0] for c in items[0].contexts] == [2, 3]
    _, text, a, b = items[0].contexts[1]
    assert text[a:b] == "林震推门而入"
    page = review.page(items, title="t", intro="i")
    assert "book&lt;1&gt;" in page and "<mark>林震推门而入</mark>" in page
    assert "未核实" in page and "死后出场" in page
