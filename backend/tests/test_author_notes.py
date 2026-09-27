"""Author's notes (step 4.5): which paragraphs at a chapter's start and end are the
author talking to readers, from explicit markers and the model's labels."""

import json
import re

from tests.fakes import FakeBackend, MemoryCallStore
from webfic.extraction.author_notes import (
    REGION,
    find_author_notes,
    load_prompt,
    paragraphs,
    version,
)
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

STORY = '暮色四合，林远推开了山门。\n师父站在台阶上等他。\n"回来了？"师父问。'


def llm_labelling(author: set[str]):
    """A fake model marking the paragraphs whose text contains one of `author` as the
    author's; it also records what it was shown."""
    seen = []

    def answer(messages):
        user = messages[1].content
        seen.append(user)
        labels = [
            {"i": int(n), "who": "author" if any(a in text for a in author) else "story"}
            for n, text in re.findall(r"^\[(\d+)\] (.*)$", user, re.MULTILINE)
        ]
        return {"paragraphs": labels}

    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    backend = FakeBackend(lambda m: {}, notes=answer)
    return JsonLLMClient(backend, tiers, store=MemoryCallStore()), seen


def test_paragraphs_keep_their_offsets():
    content = "第一段。\n\n第二段。"
    assert [(p.start, p.end, p.text) for p in paragraphs(content)] == [
        (0, 4, "第一段。"), (6, 10, "第二段。"),
    ]  # fmt: skip


async def test_markers_are_recognised_without_a_model():
    for marker in (
        "作者有话说：今天只有一更。",
        "PS：谢谢打赏！",
        "【作者的话】下周恢复双更",
        "p.s. 求月票",
    ):
        content = STORY + "\n" + marker + "\n谢谢大家。"
        scan = await find_author_notes(None, content)
        start = content.index(marker)
        assert scan.ranges == [(start, len(content))], marker
        assert content[slice(*scan.story(content))] == STORY + "\n"
    # "PS" inside a word or a sentence is not a marker.
    assert (await find_author_notes(None, STORY + "\nPSP游戏机摆在桌上。")).ranges == []


async def test_the_author_can_register_their_own_marker():
    content = STORY + "\nNote: 灵脉的设定见第三章。"
    assert (await find_author_notes(None, content)).ranges == []
    scan = await find_author_notes(None, content, markers=["Note"])
    assert scan.ranges == [(content.index("Note"), len(content))]
    assert version(["Note"]) != version([]) and version(["Note", " "]) == version(["Note"])


async def test_the_model_finds_unmarked_notes_at_either_end():
    content = "今天加班，只有一更。\n" + STORY + "\n感谢书友的打赏，明天见！"
    llm, _ = llm_labelling({"加班", "打赏"})
    scan = await find_author_notes(llm, content)
    assert scan.model_used and scan.llm_calls == 1
    assert scan.ranges == [(0, content.index("暮色") - 1), (content.index("感谢"), len(content))]
    assert content[slice(*scan.story(content))].strip() == STORY


async def test_notes_must_touch_the_start_or_the_end():
    # The model calls a middle paragraph the author's: it is not a note (a story
    # paragraph lost would cost its facts).
    content = STORY + "\n谢谢大家的支持\n林远点了点头。"
    llm, _ = llm_labelling({"谢谢大家"})
    assert (await find_author_notes(llm, content)).ranges == []


async def test_only_the_two_ends_of_a_long_chapter_are_shown():
    middle = "林远一直走。" * 600  # a long paragraph in the middle
    content = "开头一段。\n" + middle + "\n结尾一段。"
    llm, seen = llm_labelling(set())
    await find_author_notes(llm, content)
    assert "开头一段" in seen[0] and "结尾一段" in seen[0]
    assert len(seen[0]) < 2 * REGION  # the middle is not sent in full


async def test_a_failing_model_leaves_markers_only():
    def refuse(messages):
        return "not json"

    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    backend = FakeBackend(lambda m: {}, notes=refuse)
    llm = JsonLLMClient(backend, tiers, store=MemoryCallStore(), max_retries=0)
    content = STORY + "\nPS：明天请假。"
    scan = await find_author_notes(llm, content)
    assert not scan.model_used and scan.ranges == [(content.index("PS"), len(content))]


def test_the_prompt_examples_are_consistent():
    prompt = load_prompt()
    examples = re.findall(r"输入：\n(.*?)\n\n输出：\n(\{.*?\]\})", prompt, re.S)
    assert len(examples) == 2
    for text, output in examples:
        shown = {int(n) for n in re.findall(r"^\[(\d+)\]", text, re.MULTILINE)}
        labelled = {p["i"] for p in json.loads(output)["paragraphs"]}
        assert shown == labelled
