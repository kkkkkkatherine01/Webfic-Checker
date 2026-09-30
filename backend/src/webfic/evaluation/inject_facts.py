"""Error injection for character facts (step 5-1c): the same method as for ages
(webfic.evaluation.inject) — edit the real text by a patch, recompute, see what the
checker reports — with contradictions to detect and harmless edits that must not be
reported.

Detect: a different eye or hair colour after a stated one; the character appearing in
person after a certain death; a relation that cannot hold beside a stated one.
Controls: the same colour again; a disguised look; the dead seen in a dream; "大哥" said
to someone who is no relative.
"""

import random
import uuid

from webfic.checkers.character_facts import CHECKER_NAME, CharFact
from webfic.checkers.types import Confidence
from webfic.facts import kinship
from webfic.facts.registry import own_look

COLOR_WORD = {
    "黑": "黑色", "白": "白色", "银": "银色", "灰": "灰色", "金": "金色", "黄": "黄色",
    "红": "红色", "棕": "棕色", "蓝": "蓝色", "绿": "绿色", "紫": "紫色", "粉": "粉色",
}  # fmt: skip
# A colour clearly unlike each one, for the injected contradiction.
OTHER_COLOR = {
    "黑": "金", "白": "黑", "银": "黑", "灰": "红", "金": "黑", "黄": "蓝", "红": "蓝",
    "棕": "蓝", "蓝": "黑", "绿": "黑", "紫": "黑", "粉": "黑",
}  # fmt: skip
# A relation that cannot hold beside each one (webfic.facts.kinship.conflict), and the
# word for it.
CONFLICTING = {
    "father": ("paternal_uncle", "叔叔"),
    "mother": ("maternal_aunt", "姨妈"),
    "son": ("nephew", "侄子"),
    "daughter": ("niece", "侄女"),
    "elder_brother": ("younger_brother", "弟弟"),
    "younger_brother": ("elder_brother", "哥哥"),
    "elder_sister": ("younger_sister", "妹妹"),
    "younger_sister": ("elder_sister", "姐姐"),
    "paternal_uncle": ("maternal_uncle", "舅舅"),
    "maternal_uncle": ("paternal_uncle", "叔叔"),
    "paternal_aunt": ("maternal_aunt", "姨妈"),
    "maternal_aunt": ("paternal_aunt", "姑姑"),
    "paternal_grandfather": ("father", "父亲"),
    "maternal_grandfather": ("father", "父亲"),
}
assert all(kinship.conflict(kinship.kin(a), kinship.kin(b)) for a, (b, _) in CONFLICTING.items())

KINDS = {
    "fact_trait": "人物·改外貌",
    "fact_revival": "人物·死后出场",
    "fact_kinship": "人物·亲属冲突",
    "fact_control_trait": "人物对照·外貌一致",
    "fact_control_disguise": "人物对照·易容",
    "fact_control_dream": "人物对照·梦见死者",
    "fact_control_polite": "人物对照·称呼大哥",
}
DETECT = ("fact_trait", "fact_revival", "fact_kinship")
# The kind of issue each detection kind should produce.
EXPECTED_TYPE = {
    "fact_trait": "fact.appearance",
    "fact_revival": "timeline.revival",
    "fact_kinship": "character.kinship",
}
QUOTAS = {
    "fact_trait": 25, "fact_revival": 25, "fact_kinship": 25, "fact_control_trait": 20,
    "fact_control_disguise": 20, "fact_control_dream": 20, "fact_control_polite": 20,
}  # fmt: skip


def _comparable(f: CharFact) -> bool:
    return not (f.is_flashback or f.is_speculative) and own_look(f.qualifiers)


def candidates(view, kind: str, rng: random.Random) -> list:
    """Every place in the book where an injection of `kind` can be made (`view` is an
    inject.BookView)."""
    from webfic.evaluation.inject import Injection, _insertion_points

    found: list[Injection] = []

    def add(fact: CharFact, point, sentence: str, names: list[str], detail: str, expect):
        number, start, line, _ = point
        found.append(
            Injection(
                id=f"{view.name}/{kind}/{len(found)}", book=view.name, kind=kind,
                character=names[0], names=names, chapter=number, old=line,
                new=line + sentence,
                span=(start + len(line), start + len(line) + len(sentence)),
                ref=(fact.chapter_number, fact.char_start, fact.char_end),
                expect=expect, distance=number - fact.chapter_number, detail=detail,
                checker=CHECKER_NAME,
            )
        )  # fmt: skip

    facts = view.char_facts
    by: dict[tuple, list[CharFact]] = {}
    for f in facts:
        by.setdefault((f.character_id, f.category, f.attribute), []).append(f)
    for group in by.values():
        group.sort(key=lambda f: f.pos)

    if kind in ("fact_trait", "fact_control_trait", "fact_control_disguise"):
        for (cid, category, attribute), group in by.items():
            if category != "appearance" or attribute not in ("eye_color", "hair_color"):
                continue
            names = view.all_names.get(cid, [])
            usable = [f for f in group if _comparable(f) and f.value_text in COLOR_WORD]
            if not names or not usable:
                continue
            a = usable[0]
            later = [f for f in group if _comparable(f) and f.pos > a.pos]
            points = _insertion_points(view, names, _At(a), _At(later[0]) if later else None)
            points = _alive(view, cid, points)  # a dead character's look would be a revival
            if not points:
                continue
            point = rng.choice(points)
            name = point[3]
            color = a.value_text
            if kind == "fact_trait":
                color = OTHER_COLOR[a.value_text]
            part = "眼睛" if attribute == "eye_color" else "头发"
            if kind == "fact_control_disguise":
                other = COLOR_WORD[OTHER_COLOR[a.value_text]]
                sentence = f"{name}戴上易容面具，转眼成了一个{other}{part}的陌生人。"
                detail = f"前文{part}{a.value_text}，插入易容后的{other}"
                expect = None
            else:
                if attribute == "eye_color":
                    sentence = f"{name}眨了眨那双{COLOR_WORD[color]}的眼睛。"
                else:
                    sentence = f"{name}拢了拢那一头{COLOR_WORD[color]}的头发。"
                detail = f"前文{part}{a.value_text}，插入{color}"
                expect = Confidence.SUSPECTED_REVIEW if kind == "fact_trait" else None
            add(a, point, sentence, names, detail, expect)

    elif kind in ("fact_revival", "fact_control_dream"):
        for (cid, category, attribute), group in by.items():
            if category != "life" or attribute != "died":
                continue
            death = next((f for f in group if not f.is_speculative), None)
            names = view.all_names.get(cid, [])
            if death is None or not names:
                continue
            # A later chapter only: the checker leaves the death's own chapter alone.
            points = [
                p
                for p in _insertion_points(view, names, _At(death), None, any_line=True)
                if p[0] > death.chapter_number
            ]
            if not points:
                continue
            point = rng.choice(points)
            name = names[0]
            if kind == "fact_revival":
                sentence = f"{name}推门走了进来，在桌边坐下。"
                detail, expect = (
                    f"第 {death.chapter_number} 章已死，插入亲自出场",
                    (Confidence.SUSPECTED_REVIEW),
                )
            else:
                sentence = f"那天夜里，有人梦见{name}站在门口，朝屋里笑了笑。"
                detail, expect = f"第 {death.chapter_number} 章已死，插入梦境", None
            add(death, point, sentence, names, detail, expect)

    elif kind == "fact_kinship":
        for (cid, category, attribute), group in by.items():
            if category != "kinship" or attribute not in CONFLICTING:
                continue
            f = next((g for g in group if not g.is_speculative), None)
            other = view.char_names.get(f.qualifiers.get("other_name", "")) if f else None
            names = view.all_names.get(cid, [])
            other_names = view.all_names.get(other, []) if other else []
            if f is None or not names or not other_names:
                continue
            # A relation to "姨父" or "爹娘" is itself a misreading, not a sound anchor.
            if _looks_like_kin(names[0]) or _looks_like_kin(other_names[0]):
                continue
            points = _alive(view, cid, _insertion_points(view, names, _At(f), None))
            if not points:
                continue
            point = rng.choice(points)
            # The checker compares with the statement about the pair just before.
            f = _last_before(facts, {cid, other}, point, view.char_names) or f
            _, word = CONFLICTING[attribute]
            sentence = f"{names[0]}是{other_names[0]}的{word}。"
            detail = f"前文「{names[0]}是{other_names[0]}的{kinship.label(attribute)}」，插入{word}"
            add(f, point, sentence, [*names, *other_names], detail, Confidence.SUSPECTED_REVIEW)

    elif kind == "fact_control_polite":
        present = [f for f in facts if f.category == "life" and f.attribute == "present"]
        related = {
            frozenset((f.character_id, view.char_names.get(f.qualifiers.get("other_name", ""))))
            for f in facts
            if f.category == "kinship"
        }
        for f in present:
            names = view.all_names.get(f.character_id, [])
            others = [
                cid
                for cid in view.all_names
                if cid != f.character_id and frozenset((cid, f.character_id)) not in related
            ]
            if not names or not others:
                continue
            points = _alive(view, f.character_id, _insertion_points(view, names, _At(f), None))
            if not points:
                continue
            point = rng.choice(points)
            # The speaker must be alive there too, or the line would be a revival.
            speakers = [cid for cid in sorted(others, key=str) if _alive(view, cid, [point])]
            if not speakers:
                continue
            speaker = view.all_names[rng.choice(speakers)][0]
            sentence = f"{speaker}笑着冲{point[3]}喊了一声“大哥”。"
            add(f, point, sentence, names, f"{speaker}称{names[0]}为大哥（不是亲属）", None)
    return found


def _looks_like_kin(name: str) -> bool:
    """A kinship word taken for a name ("姨父", "爹娘")."""
    from webfic.extraction.character_facts import KIN_TERMS

    return any(term in name for term in KIN_TERMS)


def _alive(view, character_id, points: list) -> list:
    """The insertion points before the character's first certain death: injected text
    that has a dead character act would be a revival, whatever it was meant to test."""
    deaths = [
        f.pos
        for f in view.char_facts
        if f.character_id == character_id
        and f.category == "life"
        and f.attribute == "died"
        and not f.is_speculative
    ]
    first = min(deaths, default=None)
    return [p for p in points if first is None or (p[0], p[1]) < first]


def _last_before(
    facts: list[CharFact], pair: set, point, names: dict[str, uuid.UUID]
) -> CharFact | None:
    """The last certain statement about a pair (either way round) before a point."""
    number, start, line, _ = point
    before = [
        f
        for f in facts
        if f.category == "kinship"
        and not f.is_speculative
        and f.character_id in pair
        and names.get(f.qualifiers.get("other_name", "")) in pair - {f.character_id}
        and f.pos < (number, start + len(line))
    ]
    return max(before, key=lambda f: f.pos, default=None)


class _At:
    """What inject._insertion_points needs of a fact: where it is."""

    def __init__(self, f: CharFact):
        self.chapter_number, self.char_start, self.char_end = (
            f.chapter_number, f.char_start, f.char_end,
        )  # fmt: skip


def character_names(names: dict[uuid.UUID, list[str]]) -> dict[str, uuid.UUID]:
    return {n: cid for cid, ns in names.items() for n in ns}
