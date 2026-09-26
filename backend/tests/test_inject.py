import random
import re
import uuid

import pytest

from tests.fakes import FakeBackend
from tests.story01_ideal import a, elapsed
from webfic.checkers.age import AgeFact, ElapsedFact
from webfic.checkers.types import Confidence
from webfic.config import Settings
from webfic.db.models import Chapter
from webfic.evaluation.inject import (
    INJECT_USER,
    Base,
    BookView,
    candidates,
    chinese,
    load_view,
    plan,
    prepare,
    restyle,
    run_injection,
    score,
)
from webfic.ingest.splitter import parse_number
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig


def test_chinese_numerals_round_trip():
    numbers = [(1, "一"), (10, "十"), (17, "十七"), (20, "二十"), (23, "二十三"), (100, "一百"),
               (105, "一百零五"), (110, "一百一十"), (150, "一百五十")]  # fmt: skip
    for n, text in numbers:
        assert chinese(n) == text
        assert parse_number(text) == n
    assert restyle(12, "18") == "12" and restyle(12, "十八") == "十二"
    with pytest.raises(ValueError):
        chinese(1000)


# --- choosing injections --------------------------------------------------------------

CH1 = "林远今年十八岁，在山上练剑。\n师父看着他。"
CH2 = "又过了两年。\n林远下山了。"
CH3 = "林远今年二十岁了。\n林远进了城。"
LIN = uuid.uuid4()


def fact(chapter: int, content: str, raw: str, value: float, **kw) -> AgeFact:
    start = content.index(raw)
    return AgeFact(uuid.uuid4(), LIN, "林远", raw, "absolute_age", value, None, False, None,
                   chapter, start, start + len(raw), **kw)  # fmt: skip


def view(issues=(), ages=None) -> BookView:
    ages = ages or [
        fact(1, CH1, "林远今年十八岁", 18),
        fact(3, CH3, "林远今年二十岁", 20),
    ]
    two_years = ElapsedFact(uuid.uuid4(), "又过了两年", 2, False, 2, 0, 5, "advance")
    return BookView("demo", uuid.uuid4(), {1: CH1, 2: CH2, 3: CH3}, {LIN: ["林远"]},
                    ages, [two_years], list(issues))  # fmt: skip


def test_modify_rewrites_the_later_age():
    [backwards] = candidates(view(), "modify_backwards", random.Random(0))
    assert backwards.chapter == 3 and backwards.expect == Confidence.CONFIRMED
    assert backwards.new.startswith("林远今年十四岁")
    new_content = CH3.replace(backwards.old, backwards.new)
    assert new_content[slice(*backwards.span)] == "林远今年十四岁"
    [jump] = candidates(view(), "modify_jump", random.Random(0))
    assert "二十七岁" in jump.new  # 18 + 2 years + 7
    assert jump.expect == Confidence.SUSPECTED_REVIEW


def test_insertions_go_between_the_two_ages_and_after_a_mention():
    rng = random.Random(0)
    for kind, sentence in [
        ("insert_backwards", "林远今年十四岁。"),
        ("control_consistent", None),  # 18, or 20 after the two years
        ("control_speculative", None),
    ]:
        found = candidates(view(), kind, rng)
        # First pair: chapter 2 is the only line mentioning 林远 in between; the last
        # age pairs with the end of the book (chapter 3's second line).
        assert {(i.chapter, i.old) for i in found} == {(2, "林远下山了。"), (3, "林远进了城。")}
        first = next(i for i in found if i.chapter == 2)
        added = first.new.removeprefix(first.old)
        assert CH2.replace(first.old, first.new)[slice(*first.span)] == added
        if sentence:
            assert added == sentence
        if kind == "control_consistent":
            assert added == "林远今年二十岁了。"
        if kind == "control_speculative":
            assert added == "看样子，林远大概有三十五岁吧。"
    # A future duration needs a later age to disturb, and can go on any line before it.
    future = candidates(view(), "control_future", rng)
    assert {(i.chapter, i.old) for i in future} == {
        (1, "师父看着他。"), (2, "又过了两年。"), (2, "林远下山了。")
    }  # fmt: skip
    assert all(i.expect is None and "林远" in i.new for i in future)
    # Flashbacks agree with the present.
    [flash, _] = candidates(view(), "control_flashback", rng)
    k, m = re.match(r"(.+)年前，林远才(.+)岁。", flash.new.removeprefix(flash.old)).groups()
    assert parse_number(k) + parse_number(m) == 20


def test_ages_already_in_an_issue_and_immortals_are_skipped():
    from webfic.checkers.types import Evidence
    from webfic.services.reports import IssueView

    issue = IssueView(
        id=uuid.uuid4(), checker="age_arithmetic", issue_type="character_age",
        confidence=Confidence.CONFIRMED, status="open", description="",
        evidence=[Evidence(chapter_number=3, quote="林远今年二十岁", char_start=0, char_end=7)],
        subjects=[str(LIN)],
    )  # fmt: skip
    assert candidates(view([issue]), "modify_backwards", random.Random(0)) == []
    old = [fact(1, CH1, "林远今年十八岁", 800), fact(3, CH3, "林远今年二十岁", 820)]
    assert candidates(view(ages=old), "insert_backwards", random.Random(0)) == []


def test_plan_spreads_quotas_over_books():
    views = [view() for _ in range(4)]
    for n, v in enumerate(views):
        v.name = f"book{n}"
    chosen = plan(views, {"insert_backwards": 5, "control_future": 2}, seed=1, per_book=2)
    assert sum(i.kind == "insert_backwards" for i in chosen) == 5
    assert sum(i.kind == "control_future" for i in chosen) == 2
    assert max(sum(i.book == v.name for i in chosen if i.kind == "insert_backwards")
               for v in views) <= 2  # fmt: skip
    assert chosen == plan(views, {"insert_backwards": 5, "control_future": 2}, seed=1, per_book=2)


# --- running on a real (SQLite) database ------------------------------------------------

AGE = re.compile(r"(林远)今年([一二三四五六七八九十百]+)岁")
YEARS = re.compile(r"[一两二三四五六七八九十]+年")


def respond(messages):
    """An ideal extractor for the demo book, except that it takes any "N年" (even a future
    duration) as story time passing."""
    text = messages[1].content
    ages = [a(m[1], m[1], m[0], parse_number(m[2])) for m in AGE.finditer(text)]
    spans = [elapsed(m[0], parse_number(m[0][:-1])) for m in YEARS.finditer(text)]
    return {"age_statements": ages, "elapsed_time_statements": spans}


async def test_run_injection_end_to_end(factory):
    backend = FakeBackend(respond)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}

    def make_llm(book_id):
        return JsonLLMClient(backend, tiers)

    text = f"第1章 上山\n{CH1}\n第2章 下山\n{CH2}\n第3章 进城\n{CH3}"
    books = await prepare(factory, make_llm, Settings(), [Base("demo", text)],
                          on_progress=lambda _: None)  # fmt: skip
    async with factory() as session:
        v = await load_view(session, "demo", books["demo"])
    assert v.issues == [] and len(v.ages) == 2
    calls = len(backend.calls)

    rng = random.Random(0)
    outcomes = []
    for kind in ("insert_backwards", "insert_jump", "control_consistent", "control_future"):
        inj = next(i for i in candidates(v, kind, rng) if i.old == "林远下山了。")
        outcomes.append(
            await run_injection(factory, make_llm(v.book_id), Settings(), v.book_id, inj)
        )
    backwards, jump, consistent, future = outcomes
    assert backwards.detected and backwards.confidence == Confidence.CONFIRMED
    assert backwards.age_right and backwards.collateral_added == []
    assert jump.detected and jump.confidence == Confidence.SUSPECTED_REVIEW
    assert consistent.involving == [] and consistent.age_right
    # The fake extractor takes the future duration as time passing: 18 + 2 + N ≠ 20.
    assert future.elapsed_on_span and future.elapsed_on_span[0].startswith("advance")
    # The false issue quotes the injected sentence as evidence: a false alarm, not collateral.
    assert future.involving and not future.collateral_added
    assert len(backend.calls) == calls + 4  # one chapter read per injection

    # Every dry run was rolled back.
    async with factory() as session:
        chapter2 = await session.scalar(
            Chapter.__table__.select()
            .with_only_columns(Chapter.content)
            .where(Chapter.user_id == INJECT_USER, Chapter.number == 2)
        )
    assert chapter2 == CH2

    rows = {m.kind: m for m in score(outcomes)}
    assert rows["insert_backwards"].detected.value == 1.0
    assert rows["control_consistent"].false_alarm.value == 0.0
    assert rows["control_future"].taken_as_advance.value == 1.0
    assert rows["control_future"].false_alarm.value == 1.0
