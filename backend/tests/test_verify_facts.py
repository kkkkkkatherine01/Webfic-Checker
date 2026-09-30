"""The verify agent on character facts (step 5-1): its own prompt, reasons and tools,
chosen by the issue's checker; ages keep theirs."""

import re

from sqlalchemy import select

from tests.fakes import FakeBackend, MemoryCallStore, ScriptedBackend, agent_llm
from tests.test_chapters import USER, World, respond
from tests.test_verify import _texts
from webfic.agent.verify import FactsReason, load_prompt, tools_for
from webfic.checkers.types import IssueType
from webfic.config import Settings
from webfic.db.models import IssueRow
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig
from webfic.services import verification

PROMPT = load_prompt("verify_facts_v1")

BOOK = """第1章 起
林震走了进来。
第2章 承
林震病故了。
第3章 转
林震推门而入。
"""


def facts(messages):
    text = messages[1].content.rsplit("）：\n", 1)[1]
    out = {"traits": [], "kinship": [], "deaths": [], "presence": []}
    if "病故" in text:
        out["deaths"] = [{"mention": "林震", "resolved_name": "林震", "raw_text": "林震病故了"}]
    for quote in ("林震走了进来", "林震推门而入"):
        if quote in text:
            out["presence"] = [{"mention": "林震", "resolved_name": "林震", "raw_text": quote}]
    return out


def test_the_prompt_names_only_its_own_reasons():
    from typing import get_args

    named = set(re.findall(r"reason=(\w+)", PROMPT)) | set(re.findall(r"（([a-z_]+)）：", PROMPT))
    assert named and named <= set(get_args(FactsReason))


def test_no_test_or_real_text_leaks_into_the_prompt():
    fragments = set(re.findall(r"[一-鿿]{6,}", PROMPT))
    leaks = [
        (path.name, f)
        for path in _texts()
        for f in fragments
        if f in path.read_text("utf-8-sig", errors="ignore")
    ]
    assert leaks == []


async def test_a_revival_is_verified_with_the_character_facts_prompt_and_tools(factory):
    world = World(factory)
    world.backend = FakeBackend(respond, facts=facts)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    world.llm = JsonLLMClient(world.backend, tiers, store=MemoryCallStore())
    await world.load(BOOK)
    async with factory() as session:
        issue = (await session.scalars(select(IssueRow))).one()
    assert issue.issue_type == IssueType.TIMELINE_REVIVAL

    backend = ScriptedBackend([
        [("read_passage", {"chapter": 3, "start": 0, "end": 6})],
        [("submit_verdict", {"verdict": "contradiction", "reason": "genuine",
                             "explanation": "死后又出场，原文没有交代",
                             "evidence": [{"chapter": 3, "quote": "林震推门而入"}]})],
    ])  # fmt: skip
    result = await verification.verify_issues(
        factory, agent_llm(backend), Settings(), user_id=USER, book_id=world.book_id
    )
    (outcome,) = result.verified
    assert (outcome.verification.verdict, outcome.verification.reason) == (
        "contradiction", "genuine",
    )  # fmt: skip
    system, task = backend.calls[0][0].content, backend.calls[0][1].content
    assert system == PROMPT
    assert "死亡" in task and "亲自出场" in task
    assert [t.name for t in backend.tools_seen[0]] == [
        t.name for t in tools_for("verify_facts_v1").specs()
    ]
    assert "category" in {k for t in backend.tools_seen[0] if t.name == "list_facts"
                          for k in t.parameters["properties"]}  # fmt: skip
