"""The verify agent (step 4-2): its tools, the task it is given, the guard on its verdict,
storing verdicts and how reports show them. The model is scripted; the book is the
chapter-management test book, whose chapter 3 ("林远今年16岁") contradicts chapter 2
("林远今年21岁")."""

import re
import uuid
from pathlib import Path

import pytest
from sqlalchemy import select

from tests.fakes import ScriptedBackend, agent_llm, make_archival
from tests.story01_ideal import GOLDEN_DIR
from tests.test_chapters import USER, World
from webfic.agent.budget import Budget
from webfic.agent.tools import RunState, ToolContext
from webfic.agent.verify import (
    MAX_ITEMS,
    CharacterArgs,
    FactsArgs,
    SearchTextArgs,
    SpansArgs,
    VerdictArgs,
    check_verdict,
    get_character,
    list_facts,
    list_time_spans,
    load_prompt,
    search_text,
)
from webfic.config import Settings
from webfic.db.models import IssueRow, IssueVerificationRow
from webfic.services import chapters, reports, verification

READ_CH3 = [("read_passage", {"chapter": 3, "start": 0, "end": 7})]


def verdict(kind="false_alarm", reason="flashback", quotes=(("3", "林远今年16岁"),)):
    return [("submit_verdict", {
        "verdict": kind, "reason": reason, "explanation": "说明",
        "evidence": [{"chapter": int(c), "quote": q} for c, q in quotes],
    })]  # fmt: skip


@pytest.fixture
async def world(factory, tmp_path):
    return await World(factory, make_archival(tmp_path, size=40, overlap=10)).load()


def context(world, user_id=USER):
    return ToolContext(world.factory, user_id, world.book_id, world.archival)


async def verify(world, turns, **kwargs):
    backend = ScriptedBackend(turns)
    result = await verification.verify_issues(
        world.factory, agent_llm(backend), Settings(), user_id=USER, book_id=world.book_id,
        archival=world.archival, **kwargs,
    )  # fmt: skip
    return result, backend


async def report(world, **kwargs):
    async with world.factory() as session:
        return await reports.get_report(session, user_id=USER, book_id=world.book_id, **kwargs)


# --- the task ------------------------------------------------------------------------------


async def test_the_task_describes_each_piece_of_evidence(world):
    async with world.factory() as session:
        issue = await session.scalar(select(IssueRow))
        task = await verification.build_task(session, USER, world.book_id, issue)
    assert "待核实的矛盾（确定矛盾）" in task
    assert "[1] 第 2 章 5–12「林远今年21岁」" in task and "[2] 第 3 章 0–7「林远今年16岁」" in task
    assert task.count("角色：林远；原文称呼：林远") == 2
    assert "抽取结果：年龄 16 岁；现在时" in task


# --- tools ---------------------------------------------------------------------------------


async def test_tools_give_compact_views(world):
    ctx = context(world)
    facts = await list_facts(ctx, FactsArgs(character="林远"))
    assert [(f.chapter, f.text, f.reading) for f in facts] == [
        (1, "林远今年18岁", "年龄 18 岁；现在时"),
        (2, "林远今年21岁", "年龄 21 岁；现在时"),
        (3, "林远今年16岁", "年龄 16 岁；现在时"),
    ]
    spans = await list_time_spans(ctx, SpansArgs(first_chapter=1, last_chapter=4))
    assert [(s.chapter, s.text, s.reading) for s in spans] == [(2, "3年后", "推进，3 年")]
    person = await get_character(ctx, CharacterArgs(name="林远", as_of_chapter=2))
    assert person.stated_age == "21 岁（第 2 章「林远今年21岁」）"
    hits = await search_text(ctx, SearchTextArgs(query="林远", k=2))
    assert hits and all("林远" in h.text for h in hits)


async def test_long_lists_ask_for_a_narrower_range(world, monkeypatch):
    monkeypatch.setattr("webfic.agent.verify.MAX_ITEMS", 2)
    answer = await list_facts(context(world), FactsArgs(character="林远"))
    assert isinstance(answer, str) and "first_chapter" in answer
    assert MAX_ITEMS == 40


async def test_search_needs_an_index(world):
    ctx = ToolContext(world.factory, USER, world.book_id, None)
    with pytest.raises(ValueError, match="检索不可用"):
        await search_text(ctx, SearchTextArgs(query="林远"))


# --- the guard -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "used", "problem"),
    [
        ({"verdict": "contradiction", "reason": "flashback"}, ["read_passage"], "genuine"),
        ({"verdict": "false_alarm", "reason": "genuine"}, ["read_passage"], "genuine"),
        ({"verdict": "false_alarm", "reason": "flashback"}, ["list_facts"], "必须读过原文"),
        ({"verdict": "needs_author", "reason": "ambiguous", "evidence": []}, [], "至少"),
        ({"verdict": "contradiction", "reason": "genuine",
          "evidence": [{"chapter": 3, "quote": "林远今年61岁"}]}, [], "找不到"),
        ({"verdict": "contradiction", "reason": "genuine",
          "evidence": [{"chapter": 9, "quote": "林远今年16岁"}]}, [], "找不到"),
    ],
)  # fmt: skip
async def test_the_guard_turns_down_bad_verdicts(world, args, used, problem):
    full = {"explanation": "x", "evidence": [{"chapter": 3, "quote": "林远今年16岁"}], **args}
    found = await check_verdict(context(world), VerdictArgs(**full), RunState(tools_used=used))
    assert found is not None and problem in found


async def test_the_guard_accepts_a_sound_verdict_and_forgives_punctuation(world):
    args = VerdictArgs(
        verdict="false_alarm", reason="flashback", explanation="x",
        evidence=[{"chapter": 3, "quote": "林远今年16岁"}, {"chapter": 2, "quote": "3 年后"}],
    )  # fmt: skip
    assert await check_verdict(context(world), args, RunState(["read_passage"])) is None


# --- verifying and reporting --------------------------------------------------------------


async def test_a_dismissed_issue_is_hidden_but_kept(world):
    result, backend = await verify(world, [READ_CH3, verdict()])
    assert result.pending == 1 and len(result.verified) == 1 and result.already == 0
    v = result.verified[0].verification
    assert (v.status, v.verdict, v.reason) == ("done", "false_alarm", "flashback")
    assert "第 2 章 5–12「林远今年21岁」" in backend.calls[0][1].content  # the task

    assert (await report(world)).issues == []
    everything = await report(world, include_closed=True)
    assert [i.verification.verdict for i in everything.issues] == ["false_alarm"]
    async with world.factory() as session:
        issue = await session.scalar(select(IssueRow))
    assert issue.status == "open"  # the author's status is left alone


async def test_a_confirmed_issue_stays_in_the_report_with_its_verdict(world):
    await verify(world, [READ_CH3, verdict("contradiction", "genuine")])
    shown = (await report(world)).issues
    assert [i.verification.verdict for i in shown] == ["contradiction"]


async def test_a_valid_verdict_is_not_asked_for_again(world):
    await verify(world, [READ_CH3, verdict()])
    result, backend = await verify(world, [])
    assert (result.pending, result.already, backend.calls) == (0, 1, [])
    result, backend = await verify(
        world, [READ_CH3, verdict("needs_author", "ambiguous")], again=True
    )
    assert result.verified[0].verification.verdict == "needs_author"
    assert [i.verification.verdict for i in (await report(world)).issues] == ["needs_author"]


async def test_an_unfinished_run_leaves_the_issue_in_the_report(world):
    result, _ = await verify(world, [READ_CH3] * 3, budget=Budget(max_turns=2))
    v = result.verified[0].verification
    assert (v.status, v.verdict) == ("budget_exhausted", None) and "2 轮" in v.explanation
    shown = (await report(world)).issues
    assert len(shown) == 1 and shown[0].verification.verdict is None
    assert (await verify(world, []))[0].already == 1  # not retried unless asked


async def test_verdicts_survive_renumbering_but_not_editing_a_quoted_chapter(world):
    await verify(world, [READ_CH3, verdict("contradiction", "genuine")])
    # Chapter 1 goes: the issue is now between chapters 1 and 2, same text, same verdict.
    await world.do(chapters.delete_chapter, number=1)
    shown = (await report(world)).issues
    assert [sorted({e.chapter_number for e in i.evidence}) for i in shown] == [[1, 2]]
    assert shown[0].verification.verdict == "contradiction"
    # Editing the quoted chapter (the contradiction stays) invalidates the verdict.
    await world.do(chapters.replace_chapter, number=2, content="林远今年16岁。又是一年。")
    shown = (await report(world)).issues
    assert len(shown) == 1 and shown[0].verification is None


async def test_nobody_else_can_verify_the_book(world):
    with pytest.raises(verification.NotFound):
        await verification.verify_issues(
            world.factory, agent_llm(ScriptedBackend([])), Settings(), user_id=uuid.uuid4(),
            book_id=world.book_id,
        )  # fmt: skip
    async with world.factory() as session:
        assert (await session.scalars(select(IssueVerificationRow))).all() == []


# --- the prompt ----------------------------------------------------------------------------

PROMPT = load_prompt()
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def test_the_prompt_names_only_real_reasons():
    from typing import get_args

    from webfic.agent.verify import Reason

    named = set(re.findall(r"reason=(\w+)", PROMPT)) | set(re.findall(r"（([a-z_]+)）：", PROMPT))
    assert named and named <= set(get_args(Reason))


def _texts():
    folders = [GOLDEN_DIR, EVAL_DIR / "private", EVAL_DIR / "external" / "realtext",
               EVAL_DIR / "external" / "shushan"]  # fmt: skip
    for folder in folders:
        if folder.exists():
            # A holdout's free-form answers (step 5-1) quote text.txt, which is checked;
            # their own wording ("血缘亲属关系") is not story text.
            yield from (p for p in folder.rglob("*.txt") if p.is_file() and p.name != "answers.txt")


def test_no_test_or_real_text_leaks_into_the_prompt():
    fragments = set(re.findall(r"[一-鿿]{6,}", PROMPT))
    leaks = [
        (path.name, f)
        for path in _texts()
        for f in fragments
        if f in path.read_text("utf-8-sig", errors="ignore")
    ]
    assert leaks == []


# --- reading limits (step 4-3) --------------------------------------------------------------


async def test_reads_are_capped_and_a_reversed_range_is_refused(world, monkeypatch):
    from webfic.agent import verify as v

    with pytest.raises(ValueError, match="end 不能小于 start"):
        v.ReadPassageArgs(chapter=3, start=5, end=2)
    monkeypatch.setattr(v, "MAX_SPAN", 3)
    monkeypatch.setattr(v, "MAX_READ", 5)
    text = await v.read_passage(context(world), v.ReadPassageArgs(chapter=3, start=0, end=7))
    assert "只返回前 3 字" in text and "【林远今】" in text
    with pytest.raises(ValueError):
        v.SearchTextArgs(query="林远", k=6)
