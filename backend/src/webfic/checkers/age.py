"""AgeArithmeticChecker: do a character's stated ages agree with the story time that
passes between them? Pure rules, no LLM.

Each comparable age statement implies a birth time B = story_time - age. Statements
about the same character should imply (roughly) the same B. Only narratively adjacent
statements are compared, so one mistake is reported once rather than against every
other statement.
"""

import uuid
from dataclasses import dataclass
from itertools import pairwise

from webfic.checkers.types import (
    Confidence,
    ConsistencyIssue,
    Evidence,
    IssueType,
    make_fingerprint,
)
from webfic.extraction.schemas import LIFE_STAGE_AGE_RANGE, LifeStage

CHECKER_NAME = "age_arithmetic"

# Ages are often rounded, and a birthday may fall between two statements.
TOLERANCE_YEARS = 1.0

_LIFE_STAGE_LABEL = {
    LifeStage.INFANT: "婴儿",
    LifeStage.CHILD: "孩童",
    LifeStage.TEEN: "少年",
    LifeStage.YOUNG_ADULT: "青年",
    LifeStage.MIDDLE_AGED: "中年",
    LifeStage.ELDERLY: "老年",
}
_LIFE_STAGE_ORDER = list(LifeStage)


@dataclass(frozen=True)
class AgeFact:
    id: uuid.UUID
    character_id: uuid.UUID
    character_name: str
    raw_text: str
    statement_type: str
    value: float | None
    life_stage: LifeStage | None
    is_flashback: bool
    years_before_present: float | None
    chapter_number: int
    char_start: int
    char_end: int
    value_max: float | None = None  # approximate ages are ranges: value .. value_max
    speculative: bool = False  # a guess, hypothetical or hearsay: never compared
    # Stable across re-extraction and chapter renumbering, unlike `id` and
    # `chapter_number`; None falls back to the chapter number.
    chapter_id: uuid.UUID | None = None

    @property
    def pos(self) -> tuple[int, int]:
        return (self.chapter_number, self.char_start)

    @property
    def low(self) -> float:
        assert self.value is not None
        return self.value

    @property
    def high(self) -> float:
        assert self.value is not None
        return max(self.value, self.value_max if self.value_max is not None else self.value)


@dataclass(frozen=True)
class ElapsedFact:
    id: uuid.UUID
    raw_text: str
    estimated_years: float | None
    is_flashback: bool
    chapter_number: int
    char_start: int
    char_end: int
    kind: str = "advance"

    @property
    def pos(self) -> tuple[int, int]:
        return (self.chapter_number, self.char_start)


def _evidence(fact: AgeFact | ElapsedFact) -> Evidence:
    return Evidence(
        chapter_number=fact.chapter_number,
        quote=fact.raw_text,
        char_start=fact.char_start,
        char_end=fact.char_end,
    )


def _fmt_years(years: float) -> str:
    return str(int(years)) if float(years).is_integer() else f"{years:g}"


def _fmt_range(low: float, high: float) -> str:
    return _fmt_years(low) if high == low else f"{_fmt_years(low)}–{_fmt_years(high)}"


def _range_distance(lo1: float, hi1: float, lo2: float, hi2: float) -> float:
    """0 if the ranges overlap, otherwise the size of the gap between them."""
    return max(0.0, lo2 - hi1, lo1 - hi2)


class _Timeline:
    """Story time built from statements that move the present forward. Time inside a
    flashback, and summaries of time already passed, do not."""

    def __init__(self, elapsed: list[ElapsedFact]):
        self._events = sorted(
            (e for e in elapsed if not e.is_flashback and e.kind == "advance"),
            key=lambda e: e.pos,
        )

    def between(self, a: tuple[int, int], b: tuple[int, int]) -> list[ElapsedFact]:
        return [e for e in self._events if a < e.pos < b]


class _FactKeys:
    """Content-based keys for fingerprints. Fact ids are regenerated whenever a chapter is
    extracted again, so a fingerprint built on them would lose the author's status
    (intentional, acknowledged) after every recomputation. A key names the chapter, the
    quote and, for a quote that occurs several times in the chapter, which occurrence.
    Character ids are left out on purpose: recomputing from the chapter that introduced
    a character creates it again under a new id, and the two statements already pin the
    issue down."""

    def __init__(self, facts: list[AgeFact]):
        seen: dict[tuple[str, str], int] = {}
        self._keys: dict[uuid.UUID, str] = {}
        for f in sorted(facts, key=lambda f: f.pos):
            chapter = str(f.chapter_id) if f.chapter_id else f"#{f.chapter_number}"
            n = seen.get((chapter, f.raw_text), 0)
            seen[(chapter, f.raw_text)] = n + 1
            self._keys[f.id] = f"{chapter}|{f.raw_text}|{n}"

    def fingerprint(self, a: AgeFact, b: AgeFact) -> str:
        return make_fingerprint(CHECKER_NAME, self._keys[a.id], self._keys[b.id])


def _flashback_offset(fact: AgeFact) -> float | None:
    """How far before the narrative present this statement is set; None = unknown."""
    if not fact.is_flashback:
        return 0.0
    return fact.years_before_present


def _is_comparable(fact: AgeFact) -> bool:
    return _flashback_offset(fact) is not None


def _story_gap(timeline: _Timeline, a: AgeFact, b: AgeFact) -> tuple[float | None, bool]:
    """Story years from a to b, and whether any elapsed-time statement lies between.
    Gap is None if an unquantifiable statement ("多年以后") lies between them."""
    between = timeline.between(a.pos, b.pos)
    if any(e.estimated_years is None for e in between):
        return None, bool(between)
    elapsed = sum(e.estimated_years or 0.0 for e in between)
    offset_a = _flashback_offset(a) or 0.0
    offset_b = _flashback_offset(b) or 0.0
    return elapsed - offset_b + offset_a, bool(between)


def _check_absolute_pair(
    timeline: _Timeline, keys: _FactKeys, a: AgeFact, b: AgeFact
) -> ConsistencyIssue | None:
    assert a.value is not None and b.value is not None
    name = b.character_name
    where_a = f"第 {a.chapter_number} 章为 {_fmt_range(a.low, a.high)} 岁"
    where_b = f"第 {b.chapter_number} 章为 {_fmt_range(b.low, b.high)} 岁"
    evidence = [_evidence(a), *(_evidence(e) for e in timeline.between(a.pos, b.pos)), _evidence(b)]

    # Story time only moves forward, so a present-time age that goes down is impossible
    # however much time passed in between (even an unquantified amount).
    # Approximate ages ("三十来岁") are ranges; exact ages are one-point ranges.
    both_present = not a.is_flashback and not b.is_flashback
    gap, has_elapsed = _story_gap(timeline, a, b)
    if both_present and b.high < a.low:
        confidence = Confidence.CONFIRMED
        description = f"{name}在{where_a}，{where_b}，年龄倒退。"
        if not has_elapsed:
            description = description.removesuffix("。") + "，且两者之间没有任何时间流逝的描写。"
    elif gap is None:
        return None
    else:
        mismatch = _range_distance(a.low + gap, a.high + gap, b.low, b.high)
        if mismatch <= TOLERANCE_YEARS:
            return None
        confidence = Confidence.SUSPECTED_REVIEW
        if not has_elapsed and both_present:
            description = (
                f"{name}在{where_a}，{where_b}，但两者之间没有时间流逝的描写。"
                "可能存在未写明的时间跨度，请确认。"
            )
        else:
            description = (
                f"{name}在{where_a}，{where_b}；按文中时间描写推算，"
                f"期间约过去 {_fmt_years(gap)} 年，"
                f"年龄变化与之相差约 {_fmt_years(mismatch)} 岁。"
            )
            if a.is_flashback or b.is_flashback:
                description += "（涉及回忆段落，推算可能不准。）"

    return ConsistencyIssue(
        checker=CHECKER_NAME,
        issue_type=IssueType.CHARACTER_AGE,
        confidence=confidence,
        description=description,
        evidence=evidence,
        fingerprint=keys.fingerprint(a, b),
        subjects=[str(b.character_id)],
    )


def _check_life_stage(
    timeline: _Timeline, keys: _FactKeys, stage_fact: AgeFact, ref: AgeFact
) -> ConsistencyIssue | None:
    """Compare a life-stage statement with the nearest absolute age of the character."""
    assert stage_fact.life_stage is not None and ref.value is not None
    first, second = sorted((ref, stage_fact), key=lambda f: f.pos)
    gap, _ = _story_gap(timeline, first, second)
    if gap is None:
        return None
    shift = gap if first is ref else -gap
    expected_low, expected_high = ref.low + shift, ref.high + shift

    low, high = LIFE_STAGE_AGE_RANGE[stage_fact.life_stage]
    stage_high = high if high is not None else float("inf")
    if _range_distance(expected_low, expected_high, low, stage_high) <= TOLERANCE_YEARS:
        return None

    label = _LIFE_STAGE_LABEL[stage_fact.life_stage]
    description = (
        f"{stage_fact.character_name}在第 {stage_fact.chapter_number} 章被描述为「{label}」，"
        f"但按第 {ref.chapter_number} 章的 {_fmt_range(ref.low, ref.high)} 岁推算，此时约 "
        f"{_fmt_range(expected_low, expected_high)} 岁。"
    )
    return ConsistencyIssue(
        checker=CHECKER_NAME,
        issue_type=IssueType.CHARACTER_AGE,
        confidence=Confidence.SUSPECTED_REVIEW,
        description=description,
        evidence=[_evidence(first), _evidence(second)],
        fingerprint=keys.fingerprint(stage_fact, ref),
        subjects=[str(stage_fact.character_id)],
    )


def _check_stage_regression(
    timeline: _Timeline, keys: _FactKeys, a: AgeFact, b: AgeFact
) -> ConsistencyIssue | None:
    """Two life stages only: flag a clear step backwards (e.g. 老年 → 少年)."""
    assert a.life_stage is not None and b.life_stage is not None
    if a.is_flashback or b.is_flashback:
        return None
    _, b_high = LIFE_STAGE_AGE_RANGE[b.life_stage]
    a_low, _ = LIFE_STAGE_AGE_RANGE[a.life_stage]
    if b_high is None or b_high >= a_low:
        return None  # ranges overlap: not a clear regression
    if _LIFE_STAGE_ORDER.index(b.life_stage) >= _LIFE_STAGE_ORDER.index(a.life_stage):
        return None

    label_a, label_b = _LIFE_STAGE_LABEL[a.life_stage], _LIFE_STAGE_LABEL[b.life_stage]
    description = (
        f"{a.character_name}在第 {a.chapter_number} 章被描述为「{label_a}」，"
        f"第 {b.chapter_number} 章却是「{label_b}」。"
        "缺少具体年龄，无法确定是否矛盾，请自行确认。"
    )
    return ConsistencyIssue(
        checker=CHECKER_NAME,
        issue_type=IssueType.CHARACTER_AGE,
        confidence=Confidence.INSUFFICIENT_INFO,
        description=description,
        evidence=[
            _evidence(a),
            *(_evidence(e) for e in timeline.between(a.pos, b.pos)),
            _evidence(b),
        ],
        fingerprint=keys.fingerprint(a, b),
        subjects=[str(a.character_id)],
    )


def check_ages(
    age_facts: list[AgeFact], elapsed_facts: list[ElapsedFact]
) -> list[ConsistencyIssue]:
    timeline = _Timeline(elapsed_facts)
    keys = _FactKeys(age_facts)
    issues: list[ConsistencyIssue] = []

    by_character: dict[uuid.UUID, list[AgeFact]] = {}
    for fact in age_facts:
        by_character.setdefault(fact.character_id, []).append(fact)

    for all_facts in by_character.values():
        # Guesses, hypotheticals and hearsay are not facts of the story.
        facts = sorted((f for f in all_facts if not f.speculative), key=lambda f: f.pos)
        absolutes = [
            f
            for f in facts
            if f.statement_type == "absolute_age" and f.value is not None and _is_comparable(f)
        ]
        stages = [
            f
            for f in facts
            if f.statement_type == "life_stage" and f.life_stage is not None and _is_comparable(f)
        ]

        for a, b in pairwise(absolutes):
            if issue := _check_absolute_pair(timeline, keys, a, b):
                issues.append(issue)

        if absolutes:
            for stage_fact in stages:
                before = [f for f in absolutes if f.pos < stage_fact.pos]
                ref = before[-1] if before else absolutes[0]
                if issue := _check_life_stage(timeline, keys, stage_fact, ref):
                    issues.append(issue)
        else:
            for a, b in pairwise(stages):
                if issue := _check_stage_regression(timeline, keys, a, b):
                    issues.append(issue)

    return issues
