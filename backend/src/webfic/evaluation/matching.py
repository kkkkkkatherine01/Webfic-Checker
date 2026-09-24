"""Compare what one pipeline run produced (an Observation) with the golden answer.
Pure functions, no I/O."""

from dataclasses import dataclass, field

from webfic.checkers.types import Confidence
from webfic.evaluation.golden import AllowedIssue, ExpectedIssue, Story

VALUE_TOLERANCE = 0.01


# --- what a run produced ---------------------------------------------------------------


@dataclass
class ModelCharacter:
    id: str
    names: set[str]  # canonical name + aliases


@dataclass
class ModelAge:
    character_id: str
    chapter: int
    start: int
    end: int
    type: str
    value: float | None
    life_stage: str | None
    flashback: bool
    years_before_present: float | None
    value_max: float | None = None
    speculative: bool = False


@dataclass
class ModelElapsed:
    chapter: int
    start: int
    end: int
    kind: str
    years: float | None
    flashback: bool


@dataclass
class ModelIssue:
    subjects: list[str]
    chapters: set[int]
    confidence: Confidence
    description: str


@dataclass
class Observation:
    characters: list[ModelCharacter]
    ages: list[ModelAge]
    elapsed: list[ModelElapsed]
    issues: list[ModelIssue]
    dropped: int = 0


# --- scores ----------------------------------------------------------------------------


@dataclass
class IssueScore:
    expected: int = 0
    matched: dict[str, bool] = field(default_factory=dict)  # expected id -> confidence ok
    missed: list[str] = field(default_factory=list)
    false_positives: list[str] = field(default_factory=list)  # descriptions
    confidence_errors: list[str] = field(default_factory=list)
    allowed: int = 0
    insufficient: int = 0  # unmatched insufficient_info reports; not counted as errors


@dataclass
class FactScore:
    ages_expected: int = 0
    ages_found: int = 0
    ages_correct: int = 0  # found and every labelled attribute right
    age_errors: list[str] = field(default_factory=list)
    elapsed_expected: int = 0
    elapsed_found: int = 0
    elapsed_kind_correct: int = 0
    elapsed_correct: int = 0
    elapsed_errors: list[str] = field(default_factory=list)
    traps: int = 0
    traps_passed: int = 0
    trap_errors: list[str] = field(default_factory=list)
    merges: list[str] = field(default_factory=list)  # one model character = several real ones
    splits: list[str] = field(default_factory=list)  # one real character = several model ones
    dropped: int = 0


@dataclass
class StoryScore:
    story: str
    issues: IssueScore
    facts: FactScore


# --- characters ------------------------------------------------------------------------


def map_characters(story: Story, characters: list[ModelCharacter]) -> dict[str, set[str]]:
    """Model character id -> standard names it corresponds to (empty if unknown)."""
    golden = story.golden
    return {
        c.id: {std for std in golden.characters if c.names & golden.all_names(std)}
        for c in characters
    }


def _character_problems(
    story: Story, characters: list[ModelCharacter], mapping: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    by_id = {c.id: c for c in characters}
    merges = [
        f"「{'、'.join(sorted(by_id[cid].names))}」被当成同一人，实际是 {'、'.join(sorted(stds))}"
        for cid, stds in mapping.items()
        if len(stds) > 1
    ]
    splits = []
    for std in story.golden.characters:
        ids = [cid for cid, stds in mapping.items() if std in stds]
        if len(ids) > 1:
            parts = " / ".join("、".join(sorted(by_id[cid].names)) for cid in ids)
            splits.append(f"{std} 被拆成 {len(ids)} 个角色：{parts}")
    return merges, splits


# --- issues ----------------------------------------------------------------------------


def _covers(
    issue: ModelIssue, spec: ExpectedIssue | AllowedIssue, mapping: dict[str, set[str]]
) -> bool:
    """Same character, and the report's two ends are in exactly the expected chapters.
    Evidence in between (elapsed-time quotes) does not matter."""
    standard_names = set().union(*(mapping.get(s, set()) for s in issue.subjects))
    if spec.character not in standard_names or not issue.chapters:
        return False
    return (min(issue.chapters), max(issue.chapters)) in spec.endpoint_options()


def match_issues(
    story: Story, issues: list[ModelIssue], mapping: dict[str, set[str]]
) -> IssueScore:
    score = IssueScore(expected=len(story.golden.issues))
    unmatched = list(issues)

    for spec in story.golden.issues:
        candidates = [i for i in unmatched if _covers(i, spec, mapping)]
        if not candidates:
            score.missed.append(spec.id)
            continue
        best = candidates[0]
        unmatched.remove(best)
        ok = spec.confidence is None or best.confidence == spec.confidence
        score.matched[spec.id] = ok
        if not ok:
            score.confidence_errors.append(
                f"{spec.id}：应为 {spec.confidence}，实际 {best.confidence}（{best.description}）"
            )

    for issue in unmatched:
        if any(_covers(issue, a, mapping) for a in story.golden.allowed):
            score.allowed += 1
        elif issue.confidence == Confidence.INSUFFICIENT_INFO:
            score.insufficient += 1
        else:
            score.false_positives.append(issue.description)
    return score


# --- facts -----------------------------------------------------------------------------


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= VALUE_TOLERANCE


def _range(low: float | None, high: float | None) -> str:
    if low is None:
        return "空"
    return f"{low:g}" if high is None or _same(low, high) else f"{low:g}–{high:g}"


def _ranges_overlap(lo1: float | None, hi1: float | None, lo2: float, hi2: float | None) -> bool:
    if lo1 is None:
        return False
    hi1 = lo1 if hi1 is None else hi1
    hi2 = lo2 if hi2 is None else hi2
    return lo1 <= hi2 + VALUE_TOLERANCE and lo2 <= hi1 + VALUE_TOLERANCE


def match_facts(story: Story, obs: Observation, mapping: dict[str, set[str]]) -> FactScore:
    facts = story.golden.facts
    score = FactScore(dropped=obs.dropped)
    score.merges, score.splits = _character_problems(story, obs.characters, mapping)

    score.ages_expected = len(facts.ages)
    for exp in facts.ages:
        span = story.span(exp.chapter, exp.quote)
        hits = [
            a for a in obs.ages if a.type == exp.type and span.overlaps(a.chapter, a.start, a.end)
        ]
        if not hits:
            score.age_errors.append(f"第 {exp.chapter} 章未抽到：{exp.quote}")
            continue
        score.ages_found += 1
        # Prefer the hit attributed to the right character.
        hit = next((a for a in hits if exp.character in mapping.get(a.character_id, ())), hits[0])
        wrong = []
        if exp.character not in mapping.get(hit.character_id, ()):
            wrong.append("角色")
        if exp.value is not None:
            if exp.value_max is None:  # exact: the value must match and not be a range
                exact_hit = hit.value_max is None or _same(hit.value_max, hit.value)
                if not _same(hit.value, exp.value) or not exact_hit:
                    wrong.append(f"数值 {_range(hit.value, hit.value_max)}≠{exp.value:g}")
            elif not _ranges_overlap(hit.value, hit.value_max, exp.value, exp.value_max) or (
                hit.value_max is None or _same(hit.value_max, hit.value)
            ):  # approximate: must be given as a range overlapping the expected one
                wrong.append(
                    f"约数区间 {_range(hit.value, hit.value_max)}≠{exp.value:g}–{exp.value_max:g}"
                )
        if exp.life_stage is not None and hit.life_stage != exp.life_stage:
            wrong.append(f"人生阶段 {hit.life_stage}≠{exp.life_stage}")
        if hit.flashback != exp.flashback:
            wrong.append("回忆标记")
        if hit.speculative != exp.speculative:
            wrong.append("推测标记")
        if (
            exp.flashback
            and exp.years_before_present is not None
            and not _same(hit.years_before_present, exp.years_before_present)
        ):
            wrong.append(f"距今年数 {hit.years_before_present}≠{exp.years_before_present:g}")
        if exp.no_years_before_present and hit.years_before_present is not None:
            wrong.append(f"距今年数应为空（原文没写），实际 {hit.years_before_present:g}")
        if wrong:
            score.age_errors.append(f"第 {exp.chapter} 章「{exp.quote}」：{'、'.join(wrong)}错误")
        else:
            score.ages_correct += 1

    score.elapsed_expected = len(facts.elapsed)
    for exp_e in facts.elapsed:
        span = story.span(exp_e.chapter, exp_e.quote)
        hit_e = next((e for e in obs.elapsed if span.overlaps(e.chapter, e.start, e.end)), None)
        if hit_e is None:
            score.elapsed_errors.append(f"第 {exp_e.chapter} 章未抽到：{exp_e.quote}")
            continue
        score.elapsed_found += 1
        wrong = []
        if hit_e.kind == exp_e.kind:
            score.elapsed_kind_correct += 1
        else:
            wrong.append(f"类型 {hit_e.kind}≠{exp_e.kind}")
        if exp_e.years is not None and not _same(hit_e.years, exp_e.years):
            wrong.append(f"年数 {hit_e.years}≠{exp_e.years:g}")
        if hit_e.flashback != exp_e.flashback:
            wrong.append("回忆标记")
        if wrong:
            score.elapsed_errors.append(
                f"第 {exp_e.chapter} 章「{exp_e.quote}」：{'、'.join(wrong)}错误"
            )
        else:
            score.elapsed_correct += 1

    score.traps = len(facts.not_ages) + len(facts.not_elapsed) + len(story.golden.not_aliases)
    all_names = {n for c in obs.characters for n in c.names}
    for name in story.golden.not_aliases:
        if name in all_names:
            score.trap_errors.append(f"误作角色名或别名：{name}")
        else:
            score.traps_passed += 1
    for trap in facts.not_ages:
        span = story.span(trap.chapter, trap.quote)
        if any(span.overlaps(a.chapter, a.start, a.end) for a in obs.ages):
            score.trap_errors.append(f"第 {trap.chapter} 章误抽为年龄：{trap.quote}")
        else:
            score.traps_passed += 1
    for trap in facts.not_elapsed:
        span = story.span(trap.chapter, trap.quote)
        advances = [e for e in obs.elapsed if e.kind == "advance"]
        if any(span.overlaps(e.chapter, e.start, e.end) for e in advances):
            score.trap_errors.append(f"第 {trap.chapter} 章误抽为时间推进：{trap.quote}")
        else:
            score.traps_passed += 1
    return score


def score_story(story: Story, obs: Observation) -> StoryScore:
    mapping = map_characters(story, obs.characters)
    return StoryScore(
        story=story.id,
        issues=match_issues(story, obs.issues, mapping),
        facts=match_facts(story, obs, mapping),
    )
