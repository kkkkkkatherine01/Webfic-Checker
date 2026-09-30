"""Error injection for character facts (step 5-1c): where and what is injected."""

import random
import uuid

from webfic.checkers.character_facts import CharFact
from webfic.evaluation import inject as ij

LIN, ZHEN, SU = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
CHAPTERS = {
    1: "林远有一双绿色的眼睛。\n林震是林远的父亲。\n苏晴走了过来。\n林震坐了下来。",
    2: "林震病故了。\n林远守在灵前。\n苏晴递来一碗茶。",
    3: "三日后，林远出了门。\n苏晴跟在后面。",
}
NAMES = {LIN: ["林远"], ZHEN: ["林震"], SU: ["苏晴"]}


def fact(who, chapter, quote, category, attribute, value=None, **qualifiers):
    start = CHAPTERS[chapter].index(quote)
    return CharFact(
        id=uuid.uuid4(), character_id=who, character_name=NAMES[who][0], category=category,
        attribute=attribute, value_text=value, is_flashback=False, is_speculative=False,
        chapter_number=chapter, char_start=start, char_end=start + len(quote), raw_text=quote,
        qualifiers=qualifiers,
    )  # fmt: skip


VIEW = ij.BookView(
    "b", uuid.uuid4(), CHAPTERS, NAMES, [], [], [],
    [
        fact(LIN, 1, "一双绿色的眼睛", "appearance", "eye_color", "绿"),
        fact(ZHEN, 1, "林震是林远的父亲", "kinship", "father", other_name="林远",
             other_mention="林远"),
        fact(SU, 1, "苏晴走了过来", "life", "present"),
        fact(ZHEN, 2, "林震病故了", "life", "died"),
    ],
    {"林远": LIN, "林震": ZHEN, "苏晴": SU},
    NAMES,
)  # fmt: skip


def made(kind):
    return ij.candidates(VIEW, kind, random.Random(1))


def test_detection_kinds_inject_what_contradicts_a_stated_fact():
    (trait,) = made("fact_trait")
    assert "林远眨了眨那双黑色的眼睛。" in trait.new and trait.expect is not None
    assert trait.checker == "character_facts" and trait.ref[0] == 1
    (revival,) = made("fact_revival")
    assert revival.chapter == 3 and "林震推门走了进来" in revival.new  # after the death's chapter
    (kin,) = made("fact_kinship")
    assert "林震是林远的叔叔。" in kin.new and set(kin.names) == {"林震", "林远"}
    assert kin.chapter == 1  # before 林震's death in chapter 2: no revival by accident


def test_controls_inject_what_must_not_be_reported():
    for kind in ("fact_control_trait", "fact_control_disguise", "fact_control_dream"):
        assert made(kind) and all(i.expect is None for i in made(kind)), kind
    assert "绿色的眼睛" in made("fact_control_trait")[0].new
    assert "易容" in made("fact_control_disguise")[0].new
    assert "梦见林震" in made("fact_control_dream")[0].new
    polite = made("fact_control_polite")
    assert polite and "“大哥”" in polite[0].new


def test_the_age_kinds_come_first_so_their_seeded_plan_is_unchanged():
    kinds = list(ij.QUOTAS)
    assert kinds.index("control_future_new") < kinds.index("fact_trait")
