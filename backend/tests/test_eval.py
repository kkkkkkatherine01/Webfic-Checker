from decimal import Decimal
from pathlib import Path

import pytest

from tests.fakes import FakeBackend
from tests.story01_ideal import STORY01, needs_golden, respond
from webfic.checkers.types import Confidence
from webfic.config import Settings
from webfic.evaluation.golden import Golden, GoldenError, Story, load_story
from webfic.evaluation.matching import (
    ModelAge,
    ModelCharacter,
    ModelElapsed,
    ModelIssue,
    Observation,
    score_story,
)
from webfic.evaluation.metrics import SampleResult, aggregate_story, combine
from webfic.evaluation.runner import open_cache, run_story
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

TEXT = (
    "第一章 甲\n林远今年十八岁。二十岁的年轻人都这样。\n"
    "第二章 乙\n两年后，林远二十岁。这些年他一直在外。"
)


def make_story(**golden) -> Story:
    data = {
        "title": "t",
        "characters": {"林远": ["林少爷"], "苏晚晴": []},
        **golden,
    }
    return Story("storyX", TEXT, split_chapters(TEXT).chapters, Golden.model_validate(data))


def span_of(chapter: int, quote: str) -> tuple[int, int]:
    content = split_chapters(TEXT).chapters[chapter - 1].content
    start = content.index(quote)
    return start, start + len(quote)


def age(cid, chapter, quote, value, type_="absolute_age", **kw):
    start, end = span_of(chapter, quote)
    return ModelAge(cid, chapter, start, end, type_, value, kw.get("stage"),
                    kw.get("flashback", False), kw.get("ybp"))  # fmt: skip


def elapsed(chapter, quote, kind, years):
    start, end = span_of(chapter, quote)
    return ModelElapsed(chapter, start, end, kind, years, False)


def issue(subjects, chapters, confidence=Confidence.CONFIRMED, desc="d"):
    return ModelIssue(subjects, set(chapters), confidence, desc)


LIN = ModelCharacter("c1", {"林远", "林少爷"})
EXPECTED = [{"id": "lin", "character": "林远", "chapters": [1, 2], "confidence": "confirmed"}]


# --- golden --------------------------------------------------------------------------


def test_golden_rejects_a_name_shared_by_two_characters():
    with pytest.raises(ValueError, match="同时属于"):
        make_story(characters={"林远": ["小林"], "林小": ["小林"]})


def test_golden_rejects_unknown_character_reference():
    with pytest.raises(ValueError, match="不在 characters"):
        make_story(issues=[{"id": "x", "character": "路人", "chapters": [1]}])


def test_load_story_reports_quotes_not_in_text(tmp_path: Path):
    (tmp_path / "text.txt").write_text(TEXT, "utf-8")
    (tmp_path / "expected.yaml").write_text(
        "title: t\ncharacters: {林远: []}\nfacts:\n  not_ages:\n"
        "    - {chapter: 1, quote: 不存在的句子}\n    - {chapter: 9, quote: x}\n",
        "utf-8",
    )
    with pytest.raises(GoldenError) as exc:
        load_story(tmp_path)
    assert "找不到引文" in str(exc.value) and "第 9 章不存在" in str(exc.value)


@needs_golden
def test_real_golden_set_is_valid():
    story = load_story(STORY01)
    assert len(story.chapters) == 5 and story.golden.issues


# --- issue matching ------------------------------------------------------------------


def test_issue_matched_through_alias_and_confidence_checked():
    story = make_story(issues=EXPECTED)
    obs = Observation([ModelCharacter("c1", {"林少爷"})], [], [], [issue(["c1"], [1, 2])])
    score = score_story(story, obs).issues
    assert score.matched == {"lin": True} and not score.false_positives


def test_wrong_confidence_still_counts_as_found():
    story = make_story(issues=EXPECTED)
    obs = Observation([LIN], [], [], [issue(["c1"], [1, 2], Confidence.SUSPECTED_REVIEW)])
    assert score_story(story, obs).issues.matched == {"lin": False}


def test_issue_on_wrong_character_or_chapters_is_missed_and_false_positive():
    su = ModelCharacter("c2", {"苏晚晴"})
    story = make_story(issues=EXPECTED)
    obs = Observation([LIN, su], [], [], [issue(["c2"], [1, 2]), issue(["c1"], [2])])
    score = score_story(story, obs).issues
    assert score.missed == ["lin"] and len(score.false_positives) == 2


def test_each_expected_issue_matches_one_report_and_allowed_is_not_an_error():
    story = make_story(issues=EXPECTED, allowed=[{"character": "林远", "chapters": [2]}])
    reports = [issue(["c1"], [1, 2]), issue(["c1"], [1, 2]), issue(["c1"], [2])]
    score = score_story(story, Observation([LIN], [], [], reports)).issues
    assert score.matched == {"lin": True}
    assert score.allowed == 1
    assert len(score.false_positives) == 1  # the duplicate report


def test_issue_ends_must_match_exactly():
    # Expected both ends in chapter 1; a cascade reaching into chapter 2 is not it.
    story = make_story(issues=[{"id": "x", "character": "林远", "chapters": [1]}])
    score = score_story(story, Observation([LIN], [], [], [issue(["c1"], [1, 2])])).issues
    assert score.missed == ["x"]


def test_issue_endpoint_alternatives():
    story = make_story(issues=[{"id": "x", "character": "林远", "chapters": [[1, 2], [2, 2]]}])
    for chapters in ([1, 2], [2]):
        obs = Observation([LIN], [], [], [issue(["c1"], chapters)])
        assert score_story(story, obs).issues.matched == {"x": True}


def test_unmatched_insufficient_info_is_not_a_false_positive():
    story = make_story()
    obs = Observation([LIN], [], [], [issue(["c1"], [1], Confidence.INSUFFICIENT_INFO)])
    score = score_story(story, obs).issues
    assert score.insufficient == 1 and score.false_positives == []


# --- fact matching -------------------------------------------------------------------


def test_fact_scoring():
    story = make_story(
        facts={
            "ages": [
                {"chapter": 1, "quote": "林远今年十八岁", "character": "林远",
                 "type": "absolute_age", "value": 18},
                {"chapter": 2, "quote": "林远二十岁", "character": "林远",
                 "type": "absolute_age", "value": 20},
            ],
            "elapsed": [
                {"chapter": 2, "quote": "两年后", "kind": "advance", "years": 2},
                {"chapter": 2, "quote": "这些年他一直在外", "kind": "retrospective"},
            ],
            "not_ages": [{"chapter": 1, "quote": "二十岁的年轻人"}],
            "not_elapsed": [{"chapter": 2, "quote": "两年后"}, {"chapter": 2, "quote": "这些年"}],
        }
    )  # fmt: skip
    obs = Observation(
        [LIN],
        ages=[
            age("c1", 1, "十八岁", 18),  # partial quote still overlaps
            age("c1", 2, "林远二十岁", 21),  # wrong value
            age("c1", 1, "二十岁的年轻人", 20),  # trap
        ],
        elapsed=[elapsed(2, "两年后", "advance", 2), elapsed(2, "这些年", "advance", None)],
        issues=[],
        dropped=1,
    )
    f = score_story(story, obs).facts
    assert (f.ages_expected, f.ages_found, f.ages_correct) == (2, 2, 1)
    assert "数值" in f.age_errors[0]
    assert (f.elapsed_found, f.elapsed_kind_correct, f.elapsed_correct) == (2, 1, 1)
    # 年龄陷阱失败；「两年后」被抽成推进 → 失败；「这些年」也被抽成推进 → 失败
    assert (f.traps, f.traps_passed) == (3, 0)
    assert f.dropped == 1


def test_merges_and_splits():
    story = make_story()
    merged = ModelCharacter("m", {"林远", "苏晚晴"})
    split_a, split_b = ModelCharacter("a", {"林远"}), ModelCharacter("b", {"林少爷"})
    f1 = score_story(story, Observation([merged], [], [], [])).facts
    f2 = score_story(story, Observation([split_a, split_b], [], [], [])).facts
    assert len(f1.merges) == 1 and not f1.splits
    assert len(f2.splits) == 1 and not f2.merges


# --- metrics -------------------------------------------------------------------------


def sample(score):
    return SampleResult(score, Decimal("0.01"), 100, 10, 2, 0, 1.0)


def test_hit_rates_across_samples_and_combine():
    story = make_story(issues=EXPECTED)
    found = score_story(story, Observation([LIN], [], [], [issue(["c1"], [1, 2])]))
    missed = score_story(story, Observation([LIN], [], [], []))
    m = aggregate_story("storyX", 100, [sample(found), sample(missed)])

    assert m.issue_hits == {"storyX/lin": 1}
    assert m.unstable_issues() == {"storyX/lin": 1}
    assert str(m.issue_recall) == "50% (1/2)"
    assert m.issue_precision.value == 1.0
    assert m.cost_usd == Decimal("0.02")
    assert m.cost_per_1k_chars == Decimal("0.1")

    total = combine([m, aggregate_story("storyY", 100, [sample(found), sample(found)])])
    assert total.samples == 2 and total.chars == 200
    assert total.issue_recall.num == 3 and total.issue_recall.den == 4


def test_failed_chapters_are_counted():
    story = make_story(issues=EXPECTED)
    score = score_story(story, Observation([LIN], [], [], []))
    failed = SampleResult(score, Decimal(0), 0, 0, 0, 0, 1.0, failed_chapters=2)
    m = aggregate_story("storyX", 100, [failed, sample(score)])
    assert m.failed_chapters == 2
    assert combine([m, m]).failed_chapters == 4


# --- runner --------------------------------------------------------------------------


@needs_golden
async def test_runner_with_ideal_extraction_scores_perfectly(tmp_path):
    story = load_story(STORY01)
    backend = FakeBackend(respond)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    cache = await open_cache(tmp_path / "cache.sqlite")

    def make_client(store):
        return JsonLLMClient(backend, tiers, store=store)

    results = await run_story(story, Settings(), make_client, cache, samples=1, fresh=False)
    m = aggregate_story(story.id, len(story.text), results)
    for ratio in (
        m.issue_recall,
        m.issue_precision,
        m.confidence_accuracy,
        m.age_recall,
        m.age_accuracy,
        m.elapsed_recall,
        m.elapsed_kind_accuracy,
        m.trap_pass,
    ):
        assert ratio.value == 1.0, (ratio, m.fact_errors, m.false_positives)  # fmt: skip
    assert (m.merges, m.splits, m.dropped) == (0, 0, 0)
    assert len(backend.calls) == 5

    # Second run is served from the eval cache; a fresh run calls the model again.
    await run_story(story, Settings(), make_client, cache, samples=1, fresh=False)
    assert len(backend.calls) == 5
    fresh = await run_story(story, Settings(), make_client, cache, samples=2, fresh=True)
    assert len(backend.calls) == 15
    assert aggregate_story(story.id, 1, fresh).issue_hits == {
        "story01/su-age-jump": 2, "story01/zhao-elderly-vs-30": 2, "story01/lin-age-backwards": 2,
    }  # fmt: skip
