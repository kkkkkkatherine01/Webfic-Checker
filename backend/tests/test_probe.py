"""The probe of real text (step 5-1a): located facts only, and the counts that decide
whether a kind of check is worth adding."""

import json

from tests.fakes import FakeBackend, MemoryCallStore
from webfic.evaluation import probe
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

BOOK = "第1章 起\n林远有一双绿眼睛。\n第2章 承\n林远碧色的眼睛。\n第3章 转\n林远的眼睛是黑色的。\n"


def fact(character, attribute, value, quote, status="present", category="外貌"):
    return {"character": character, "category": category, "attribute": attribute,
            "value": value, "quote": quote, "status": status}  # fmt: skip


def answer(messages):
    text = messages[1].content
    facts = []
    if "绿眼睛" in text:
        facts = [fact("林远", "瞳色", "绿色", "一双绿眼睛"), fact("林远", "瞳色", "绿", "没这句")]
    elif "碧色" in text:
        facts = [fact("林远", "瞳色", "绿色", "碧色的眼睛")]
    elif "黑色" in text:
        facts = [
            fact("林远", "瞳色", "黑色", "眼睛是黑色的"),
            fact("x", "", "", "林远", "guess", "怪"),
        ]
    return {"facts": facts}


async def test_the_probe_keeps_located_facts_and_counts_repeats_and_differences():
    samples = probe.choose([("wnb/0", BOOK)], "", {}, books=1)
    assert [s.chapter for s in samples] == [1, 2, 3]
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash")}
    llm = JsonLLMClient(FakeBackend(answer), tiers, store=MemoryCallStore())
    found = await probe.read(llm, samples)
    assert len(found) == 4  # "没这句" cannot be located
    looks = next(c for c in probe.stats(found) if c.category == "外貌")
    assert (looks.facts, looks.present, looks.groups, looks.repeated, looks.differing) == (
        3, 3, 1, 1, 1,
    )  # fmt: skip
    assert looks.examples == ["wnb/0 林远·瞳色：第1章「绿色」；第3章「黑色」"]
    other = next(c for c in probe.stats(found) if c.category == "其他")
    assert other.facts == 1 and other.present == 0  # unknown categories count as 其他
    json.dumps([f.model_dump() for f in found], ensure_ascii=False)


def test_the_probe_prompt_has_no_examples_to_leak():
    # Written without example passages: nothing from test stories or real works in it.
    assert "示例" not in probe.load_prompt()
