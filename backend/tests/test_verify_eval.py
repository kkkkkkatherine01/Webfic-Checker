"""The verify agent's evaluation (step 4-3): corrupting stored extraction to make known
false alarms, running a case in a rolled-back transaction, and scoring. On the
chapter-management test book with a scripted model."""

import uuid

import pytest
from sqlalchemy import select

from tests.fakes import ScriptedBackend, agent_llm, make_archival
from tests.test_chapters import USER, World, dump
from webfic.config import Settings
from webfic.db.models import FactRow
from webfic.evaluation import verify_eval as ve


@pytest.fixture
async def world(factory, tmp_path):
    return await World(factory, make_archival(tmp_path, size=40, overlap=10)).load()


async def corruption(world, type_):
    found = await ve._corruptions(world.factory, USER, world.book_id)
    return next(c for c in found if c.type == type_)


async def test_a_value_corruption_makes_a_known_false_alarm_and_changes_nothing(world):
    before = await dump(world.factory)
    c = await corruption(world, "value")  # chapter 1: 18 -> 30, older than chapter 2's 21
    fingerprint = await ve._produce(world.factory, USER, world.book_id, c)
    assert fingerprint is not None
    assert await dump(world.factory) == before  # rolled back


async def test_corruptions_that_cause_nothing_are_skipped(world):
    # Moving chapter 4's 苏晚晴 age to 林远 (20 after 16) contradicts nothing new in a way
    # that involves chapter 4 only if 林远's ages disagree; the result is either a new
    # issue at that statement or None, never an unrelated one.
    for c in await ve._corruptions(world.factory, USER, world.book_id):
        fingerprint = await ve._produce(world.factory, USER, world.book_id, c)
        if fingerprint is not None:
            assert len(fingerprint) == 32


def case_for(world, c, fingerprint):
    return ve.Case(
        id="t", set="synthetic", kind=f"synthetic:{c.type}", book="t", user_id=USER,
        book_id=world.book_id, expect=["false_alarm"], reason=c.type,
        fingerprint=fingerprint, corruption=c,
    )  # fmt: skip


READ = [("read_passage", {"chapter": 1, "start": 0, "end": 7})]


def dismiss(reason):
    return [("submit_verdict", {
        "verdict": "false_alarm", "reason": reason, "explanation": "原文是十八岁",
        "evidence": [{"chapter": 1, "quote": "林远今年18岁"}],
    })]  # fmt: skip


async def test_a_case_runs_rolled_back(world):
    c = await corruption(world, "value")
    case = case_for(world, c, await ve._produce(world.factory, USER, world.book_id, c))
    before = await dump(world.factory)
    backend = ScriptedBackend([READ, dismiss("value")])
    result = await ve.run_case(
        case, 0, world.factory, world.llm, agent_llm(backend), Settings(), world.archival,
        keep_traces=False,
    )  # fmt: skip
    assert result.set_up and result.right and result.reason == "value"
    assert (result.turns, result.tool_calls) == (2, 2)
    assert "林远今年30岁" not in backend.calls[0][1].content  # the text is untouched...
    assert "年龄 30 岁" in backend.calls[0][1].content  # ...the corrupted reading is shown
    assert await dump(world.factory) == before  # everything rolled back
    async with world.factory() as session:  # the corruption did not stay
        values = sorted(await session.scalars(select(FactRow.value_num)))
    assert values == [16, 18, 20, 21]


async def test_a_case_whose_issue_does_not_appear_is_not_scored(world):
    c = await corruption(world, "value")
    case = case_for(world, c, "0" * 32)  # no such issue
    result = await ve.run_case(
        case, 0, world.factory, world.llm, agent_llm(ScriptedBackend([])), Settings(), None,
        keep_traces=False,
    )  # fmt: skip
    assert not result.set_up
    assert ve.score([result])[0].not_set_up == 1


def result(set_, kind, verdict, *, reason=None, expect_reason=None, case_id=None, sample=0):
    expect = ["false_alarm"] if set_ == "synthetic" else ["contradiction", "needs_author"]
    case = ve.Case(
        id=case_id or f"{kind}-{uuid.uuid4().hex[:4]}", set=set_, kind=kind, book="b",
        user_id=USER, book_id=uuid.uuid4(), expect=expect, reason=expect_reason,
    )  # fmt: skip
    return ve.CaseResult(
        case=case, sample=sample, set_up=True, status="done" if verdict else "budget_exhausted",
        verdict=verdict, reason=reason, turns=3, tool_calls=4,
    )  # fmt: skip


def test_scores_per_set():
    results = [
        result("keep", "golden", "contradiction"),
        result("keep", "golden", "false_alarm", reason="flashback"),  # wrongly dismissed
        result("keep", "injected:insert_jump", "needs_author"),
        result("keep", "injected:insert_jump", None),
        result("synthetic", "synthetic:value", "false_alarm", reason="value",
               expect_reason="value"),
        result("synthetic", "synthetic:flashback", "false_alarm", reason="other",
               expect_reason="flashback"),
        result("synthetic", "synthetic:flashback", "contradiction", reason="genuine",
               expect_reason="flashback"),
    ]  # fmt: skip
    keep, synthetic = ve.score(results)
    assert (keep.set, str(keep.right), str(keep.dismissed), str(keep.unfinished)) == (
        "keep", "50% (2/4)", "25% (1/4)", "25% (1/4)",
    )  # fmt: skip
    assert str(synthetic.right) == "67% (2/3)" and str(synthetic.reason_right) == "50% (1/2)"
    assert str(synthetic.by_kind["synthetic:flashback"]) == "50% (1/2)"


def test_stability_counts_cases_with_one_verdict_across_samples():
    results = [
        result("keep", "golden", "contradiction", case_id="a", sample=0),
        result("keep", "golden", "contradiction", case_id="a", sample=1),
        result("keep", "golden", "contradiction", case_id="b", sample=0),
        result("keep", "golden", "needs_author", case_id="b", sample=1),
    ]
    (keep,) = ve.score(results)
    assert str(keep.stable) == "50% (1/2)" and keep.cases == 2 and keep.runs == 4


def test_labels_file_is_optional_and_parsed(tmp_path):
    assert ve.load_labels(tmp_path) == []
    path = tmp_path / ve.LABELS_FILE
    path.parent.mkdir(parents=True)
    path.write_text(
        "issues:\n  - {version: abc, book: shushan, fingerprint: f1, expect: [false_alarm],"
        " reason: misattributed}\n",
        "utf-8",
    )
    (label,) = ve.load_labels(tmp_path)
    assert (label.book, label.expect, label.reason) == ("shushan", ["false_alarm"], "misattributed")
