import dataclasses
import itertools
import uuid

from webfic.checkers.age import AgeFact, ElapsedFact, check_ages
from webfic.checkers.types import Confidence
from webfic.extraction.schemas import LifeStage

LIN = uuid.uuid4()
SU = uuid.uuid4()
_offsets = itertools.count(0, 10)


def age(chapter, value, *, who=LIN, name="林远", flashback=False, ybp=None, text=None):
    return AgeFact(
        id=uuid.uuid4(),
        character_id=who,
        character_name=name,
        raw_text=text or f"{value}岁",
        statement_type="absolute_age",
        value=value,
        life_stage=None,
        is_flashback=flashback,
        years_before_present=ybp,
        chapter_number=chapter,
        char_start=(start := next(_offsets)),
        char_end=start + 5,
    )


def stage(chapter, life_stage, *, who=LIN, name="林远", flashback=False, ybp=None):
    return AgeFact(
        id=uuid.uuid4(),
        character_id=who,
        character_name=name,
        raw_text=str(life_stage),
        statement_type="life_stage",
        value=None,
        life_stage=life_stage,
        is_flashback=flashback,
        years_before_present=ybp,
        chapter_number=chapter,
        char_start=(start := next(_offsets)),
        char_end=start + 5,
    )


def elapsed(chapter, years, *, flashback=False):
    return ElapsedFact(
        id=uuid.uuid4(),
        raw_text=f"{years}年后",
        estimated_years=years,
        is_flashback=flashback,
        chapter_number=chapter,
        char_start=(start := next(_offsets)),
        char_end=start + 4,
    )


def only(issues):
    assert len(issues) == 1, issues
    return issues[0]


# --- absolute ages -------------------------------------------------------------------


def test_same_age_without_elapsed_time_is_fine():
    assert check_ages([age(1, 18), age(5, 18)], []) == []


def test_age_goes_backwards_without_elapsed_time_is_confirmed():
    issue = only(check_ages([age(3, 18), age(7, 16)], []))
    assert issue.confidence == Confidence.CONFIRMED
    assert "年龄倒退" in issue.description
    assert [e.chapter_number for e in issue.evidence] == [3, 7]


def test_age_goes_backwards_by_one_is_still_confirmed():
    issue = only(check_ages([age(3, 18), age(4, 17)], []))
    assert issue.confidence == Confidence.CONFIRMED


def test_one_year_older_without_elapsed_time_is_tolerated():
    assert check_ages([age(1, 18), age(9, 19)], []) == []


def test_much_older_without_elapsed_time_is_only_suspected():
    issue = only(check_ages([age(1, 18), age(9, 21)], []))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW
    assert "未写明的时间跨度" in issue.description


def test_elapsed_time_matching_age_change_is_fine():
    facts = [age(1, 18), age(3, 21)]
    assert check_ages(facts, [elapsed(2, 3)]) == []


def test_elapsed_time_not_matching_age_change_is_suspected():
    issue = only(check_ages([age(1, 18), age(3, 19)], [elapsed(2, 5)]))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW
    assert "约过去 5 年" in issue.description
    assert [e.chapter_number for e in issue.evidence] == [1, 2, 3]


def test_age_backwards_is_confirmed_even_with_elapsed_time_between():
    issue = only(check_ages([age(1, 18), age(3, 16)], [elapsed(2, 2)]))
    assert issue.confidence == Confidence.CONFIRMED
    assert "没有任何时间流逝" not in issue.description


def test_age_backwards_is_confirmed_even_across_unquantified_time():
    vague = ElapsedFact(uuid.uuid4(), "多年以后", None, False, 2, 0, 4)
    issue = only(check_ages([age(1, 18), age(3, 16)], [vague]))
    assert issue.confidence == Confidence.CONFIRMED


def test_short_time_spans_are_ignored():
    days = ElapsedFact(uuid.uuid4(), "第八天", None, False, 2, 0, 3, kind="short")
    issue = only(check_ages([age(1, 18), age(3, 21)], [days]))
    # Treated as if nothing passed: 3 years older with no time jump -> suspected, not skipped
    assert issue.confidence == Confidence.SUSPECTED_REVIEW


def test_unquantifiable_elapsed_time_suppresses_comparison():
    vague = ElapsedFact(uuid.uuid4(), "多年以后", None, False, 2, 0, 4)
    assert check_ages([age(1, 18), age(3, 40)], [vague]) == []


def test_elapsed_time_inside_flashback_does_not_advance_story():
    facts = [age(1, 18), age(3, 21)]
    issue = only(check_ages(facts, [elapsed(2, 3, flashback=True)]))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW


def test_retrospective_time_does_not_advance_story():
    retro = ElapsedFact(uuid.uuid4(), "这两年她奔波在外", 2, False, 2, 0, 8, kind="retrospective")
    assert check_ages([age(1, 18), age(3, 18)], [retro]) == []


def test_unquantifiable_retrospective_time_does_not_suppress_comparison():
    vague = ElapsedFact(uuid.uuid4(), "已经来了好些年", None, False, 2, 0, 7, kind="retrospective")
    issue = only(check_ages([age(1, 18), age(3, 16)], [vague]))
    assert issue.confidence == Confidence.CONFIRMED


def test_elapsed_time_outside_the_pair_is_ignored():
    facts = [age(2, 18), age(3, 18)]
    assert check_ages(facts, [elapsed(1, 10), elapsed(4, 10)]) == []


# --- flashbacks ----------------------------------------------------------------------


def test_flashback_with_known_offset_is_consistent():
    # Present: 18. Flashback "ten years ago he was 8".
    facts = [age(1, 18), age(2, 8, flashback=True, ybp=10)]
    assert check_ages(facts, []) == []


def test_flashback_with_known_offset_can_be_suspected_but_never_confirmed():
    facts = [age(1, 18), age(2, 3, flashback=True, ybp=10)]
    issue = only(check_ages(facts, []))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW
    assert "回忆" in issue.description


def test_flashback_with_unknown_offset_is_not_compared():
    facts = [age(1, 18), age(2, 8, flashback=True), age(3, 18)]
    assert check_ages(facts, []) == []


# --- characters are independent; only adjacent pairs are compared ------------------------


def test_characters_are_checked_independently():
    facts = [age(1, 18), age(2, 30, who=SU, name="苏晚晴"), age(3, 18)]
    assert check_ages(facts, []) == []


def test_one_wrong_statement_reports_against_neighbours_only():
    facts = [age(1, 18), age(2, 18), age(3, 15), age(4, 18), age(5, 18)]
    issues = check_ages(facts, [])
    # 18 -> 15 (backwards, confirmed) and 15 -> 18 (older without time, suspected)
    assert sorted(i.confidence for i in issues) == [
        Confidence.CONFIRMED,
        Confidence.SUSPECTED_REVIEW,
    ]


def test_fingerprint_is_stable_for_the_same_pair():
    facts = [age(3, 18), age(7, 16)]
    assert check_ages(facts, [])[0].fingerprint == check_ages(facts, [])[0].fingerprint


def test_fingerprint_survives_re_extraction_and_renumbering():
    # Re-extracting gives facts new ids; deleting an earlier chapter shifts the numbers.
    ch_a, ch_b = uuid.uuid4(), uuid.uuid4()
    before = [dataclasses.replace(age(3, 18), chapter_id=ch_a),
              dataclasses.replace(age(7, 16), chapter_id=ch_b)]  # fmt: skip
    after = [
        dataclasses.replace(f, id=uuid.uuid4(), chapter_number=f.chapter_number - 1) for f in before
    ]
    assert check_ages(before, [])[0].fingerprint == check_ages(after, [])[0].fingerprint


def test_fingerprint_tells_repeated_quotes_apart():
    ch = uuid.uuid4()
    first, second, third = (
        dataclasses.replace(age(1, v, text="他今年十八岁"), chapter_id=ch) for v in (18, 18, 15)
    )
    issues = check_ages([first, second, third], [])
    assert len(issues) == 1  # only 18 -> 15, reported against the second quote
    other = check_ages([first, third], [])  # a different pair with the same quotes
    assert issues[0].fingerprint != other[0].fingerprint


# --- life stages ---------------------------------------------------------------------


def test_life_stage_compatible_with_age_is_fine():
    assert check_ages([age(1, 16), stage(2, LifeStage.TEEN)], []) == []


def test_life_stage_incompatible_with_age_is_suspected():
    issue = only(check_ages([age(1, 18), stage(2, LifeStage.ELDERLY)], []))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW
    assert "老年" in issue.description


def test_life_stage_accounts_for_elapsed_time():
    facts = [age(1, 18), stage(3, LifeStage.MIDDLE_AGED)]
    assert check_ages(facts, [elapsed(2, 25)]) == []


def test_life_stage_before_first_absolute_age_uses_it_as_reference():
    facts = [stage(1, LifeStage.INFANT), age(2, 30)]
    issue = only(check_ages(facts, []))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW


def test_life_stage_regression_without_ages_is_insufficient_info():
    issue = only(check_ages([stage(1, LifeStage.ELDERLY), stage(5, LifeStage.TEEN)], []))
    assert issue.confidence == Confidence.INSUFFICIENT_INFO


def test_overlapping_life_stages_are_not_flagged():
    assert check_ages([stage(1, LifeStage.YOUNG_ADULT), stage(2, LifeStage.TEEN)], []) == []


def test_life_stage_regression_in_flashback_is_not_flagged():
    facts = [stage(1, LifeStage.ELDERLY), stage(2, LifeStage.CHILD, flashback=True, ybp=60)]
    assert check_ages(facts, []) == []


# --- approximate ages and speculation (step 2.7) ----------------------------------------


def approx(chapter, low, high, **kw):
    fact = age(chapter, low, **kw)
    return dataclasses.replace(fact, value_max=high)


def test_approximate_ages_that_overlap_are_consistent():
    # 三十来岁 (30–34) ... "现在" 12 years later nobody said anything; 二十出头 in flashback
    facts = [approx(1, 30, 34), approx(2, 20, 23, flashback=True, ybp=10)]
    assert check_ages(facts, []) == []


def test_approximate_ages_that_cannot_be_reconciled_are_suspected():
    # 五十出头 (50–53), two years later 年近七十 (66–69)
    issue = only(check_ages([approx(1, 50, 53), approx(3, 66, 69)], [elapsed(2, 2)]))
    assert issue.confidence == Confidence.SUSPECTED_REVIEW
    assert "50–53" in issue.description and "66–69" in issue.description


def test_approximate_backwards_is_confirmed_only_when_ranges_do_not_overlap():
    assert check_ages([approx(1, 30, 34), age(2, 32)], []) == []
    issue = only(check_ages([approx(1, 30, 34), approx(2, 20, 23)], []))
    assert issue.confidence == Confidence.CONFIRMED


def test_speculative_ages_are_ignored():
    guess = dataclasses.replace(age(2, 20), speculative=True)
    assert check_ages([age(1, 25), guess, age(3, 25)], []) == []


def test_life_stage_against_approximate_age():
    assert check_ages([approx(1, 12, 13), stage(2, LifeStage.TEEN)], []) == []
