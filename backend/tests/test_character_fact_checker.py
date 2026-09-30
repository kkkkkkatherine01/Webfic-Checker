"""CharacterFactChecker (step 5-1): candidates from rules only."""

import uuid

from webfic.checkers.character_facts import CharFact, check_character_facts
from webfic.checkers.types import IssueType

LIN, ZHEN, SU = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
NAMES = {"林远": LIN, "远儿": LIN, "林震": ZHEN, "苏晴": SU}
WHO = {LIN: "林远", ZHEN: "林震", SU: "苏晴"}


def fact(chapter, who, category, attribute, value=None, *, at=0, quote=None, **kw):
    return CharFact(
        id=uuid.uuid4(), character_id=who, character_name=WHO[who], category=category,
        attribute=attribute, value_text=value, is_flashback=kw.pop("flashback", False),
        is_speculative=kw.pop("guess", False), chapter_number=chapter, char_start=at,
        char_end=at + 4, raw_text=quote or f"{WHO[who]}{attribute}{value}{chapter}",
        chapter_id=uuid.UUID(int=chapter), qualifiers=kw.pop("qualifiers", {}),
    )  # fmt: skip


def looks(chapter, value, attribute="eye_color", who=LIN, **kw):
    return fact(chapter, who, "appearance", attribute, value, **kw)


def rel(chapter, who, relation, other, *, address=False, **kw):
    q = {"other_name": other, "other_mention": other, **({"address": True} if address else {})}
    return fact(chapter, who, "kinship", relation, qualifiers=q, **kw)


def kinds(issues):
    return [(i.issue_type, sorted(e.chapter_number for e in i.evidence)) for i in issues]


def test_a_feature_that_changes_is_a_candidate_once_against_its_neighbour():
    facts = [looks(1, "黑"), looks(2, "黑"), looks(3, "白")]
    assert kinds(check_character_facts(facts, NAMES)) == [(IssueType.FACT_APPEARANCE, [2, 3])]


def test_disguise_guesses_and_the_past_are_not_compared():
    facts = [
        looks(1, "黑"),
        looks(2, "蓝", guess=True),
        looks(3, "金", qualifiers={"disguised": True}),
        looks(4, "黄", flashback=True),
        looks(5, "黑"),
    ]
    assert check_character_facts(facts, NAMES) == []


def test_a_mark_that_moves_is_a_candidate_but_a_different_mark_is_not():
    facts = [
        looks(1, "左脸·刀疤", "mark"),
        looks(2, "右手·胎记", "mark"),
        looks(3, "右脸·刀疤", "mark"),
    ]
    assert kinds(check_character_facts(facts, NAMES)) == [(IssueType.FACT_APPEARANCE, [1, 3])]


def test_kinship_read_from_either_side_and_through_aliases():
    agree = [rel(1, ZHEN, "father", "林远"), rel(3, LIN, "son", "林震")]
    assert check_character_facts(agree, NAMES) == []
    through_alias = [rel(1, ZHEN, "father", "远儿"), rel(4, LIN, "elder_brother", "林震")]
    issues = check_character_facts(through_alias, NAMES)
    assert kinds(issues) == [(IssueType.CHARACTER_KINSHIP, [1, 4])]
    assert "辈分不同" in issues[0].description and sorted(issues[0].subjects) == sorted(
        [str(ZHEN), str(LIN)]
    )


def test_uncles_on_different_sides_disagree_and_guesses_do_not_count():
    facts = [rel(1, ZHEN, "paternal_uncle", "苏晴"), rel(2, ZHEN, "maternal_uncle", "苏晴")]
    assert kinds(check_character_facts(facts, NAMES)) == [(IssueType.CHARACTER_KINSHIP, [1, 2])]
    rumour = [rel(1, ZHEN, "paternal_uncle", "苏晴"), rel(2, ZHEN, "father", "苏晴", guess=True)]
    assert check_character_facts(rumour, NAMES) == []


def test_appearing_after_dying_is_a_candidate_once():
    life = [
        fact(1, ZHEN, "life", "present"),
        fact(2, ZHEN, "life", "died", at=10),
        fact(2, ZHEN, "life", "present", at=50),  # same chapter: too loose to judge
        fact(4, ZHEN, "life", "present"),
        fact(6, ZHEN, "life", "present"),
    ]
    assert kinds(check_character_facts(life, NAMES)) == [(IssueType.TIMELINE_REVIVAL, [2, 4])]
    rumour = [fact(2, ZHEN, "life", "died", guess=True), fact(4, ZHEN, "life", "present")]
    assert check_character_facts(rumour, NAMES) == []


def test_fingerprints_do_not_depend_on_row_ids():
    def make():
        return [looks(1, "黑", quote="黑眼睛"), looks(3, "白", quote="白眼睛")]

    assert [i.fingerprint for i in check_character_facts(make(), NAMES)] == [
        i.fingerprint for i in check_character_facts(make(), NAMES)
    ]


# --- step 5-1d ------------------------------------------------------------------------------


def test_passing_states_are_not_compared():
    facts = [looks(1, "黑"), looks(2, "红", qualifiers={"temporary": True}), looks(3, "黑")]
    assert check_character_facts(facts, NAMES) == []


def test_marks_are_compared_only_when_both_sides_are_written():
    loose = [looks(1, "脸·伤疤", "mark"), looks(2, "右脸·伤疤", "mark")]
    assert check_character_facts(loose, NAMES) == []
    same = [looks(1, "右脸·伤疤", "mark"), looks(2, "右脸·伤疤", "mark")]
    assert check_character_facts(same, NAMES) == []
    moved = [looks(1, "右脸·伤疤", "mark"), looks(2, "左侧脸·伤疤", "mark")]
    assert kinds(check_character_facts(moved, NAMES)) == [(IssueType.FACT_APPEARANCE, [1, 2])]


def test_a_form_of_address_is_not_compared_by_cousinhood():
    cousin = [
        rel(1, ZHEN, "tang_cousin", "苏晴"),
        rel(2, ZHEN, "elder_brother", "苏晴", address=True),
    ]
    assert check_character_facts(cousin, NAMES) == []
    father = [
        rel(1, ZHEN, "elder_brother", "苏晴"),
        rel(2, ZHEN, "father", "苏晴", address=True),
    ]
    assert kinds(check_character_facts(father, NAMES)) == [(IssueType.CHARACTER_KINSHIP, [1, 2])]
    order = [rel(1, ZHEN, "elder_brother", "苏晴"), rel(2, ZHEN, "younger_brother", "苏晴",
             address=True)]  # fmt: skip
    assert kinds(check_character_facts(order, NAMES)) == [(IssueType.CHARACTER_KINSHIP, [1, 2])]
    sides = [rel(1, ZHEN, "maternal_uncle", "苏晴"), rel(2, ZHEN, "paternal_uncle", "苏晴",
             address=True)]  # fmt: skip
    assert kinds(check_character_facts(sides, NAMES)) == [(IssueType.CHARACTER_KINSHIP, [1, 2])]
    uncle = [rel(1, ZHEN, "father", "苏晴", address=True), rel(2, ZHEN, "paternal_uncle", "苏晴")]
    assert kinds(check_character_facts(uncle, NAMES)) == [(IssueType.CHARACTER_KINSHIP, [1, 2])]
