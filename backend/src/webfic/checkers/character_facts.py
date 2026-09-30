"""CharacterFactChecker (step 5-1): fixed features of appearance that change, blood
relations that cannot both hold, and characters who appear in person after dying. Pure
rules, no LLM: they find candidates; whether the text explains them (dye, disguise, a
ghost, a feigned death) or the two values are really the same is for the verify agent.

Like ages, a feature is compared with the one stated just before it, so one slip is
reported against its neighbours rather than against every other statement.
"""

import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from webfic.checkers.types import (
    Confidence,
    ConsistencyIssue,
    Evidence,
    IssueType,
    make_fingerprint,
)
from webfic.facts import kinship
from webfic.facts.registry import own_look

CHECKER_NAME = "character_facts"

_TRAIT = {"eye_color": "瞳色", "hair_color": "发色", "mark": "身体标记"}


@dataclass(frozen=True)
class CharFact:
    id: uuid.UUID
    character_id: uuid.UUID
    character_name: str
    category: str  # appearance / kinship / life
    attribute: str
    value_text: str | None
    is_flashback: bool
    is_speculative: bool
    chapter_number: int
    char_start: int
    char_end: int
    raw_text: str
    chapter_id: uuid.UUID | None = None
    mention: str = ""
    qualifiers: dict[str, Any] = field(default_factory=dict)

    @property
    def pos(self) -> tuple[int, int]:
        return (self.chapter_number, self.char_start)


def _evidence(f: CharFact) -> Evidence:
    return Evidence(
        chapter_number=f.chapter_number, quote=f.raw_text, char_start=f.char_start,
        char_end=f.char_end, chapter_id=f.chapter_id,
    )  # fmt: skip


class _Keys:
    """Content-based keys for fingerprints (as for ages, webfic.checkers.age): the
    chapter, the quote and which occurrence of it, never row or character ids, so the
    author's status survives recomputation and renumbering."""

    def __init__(self, facts: list[CharFact]):
        seen: dict[tuple[str, str], int] = {}
        self._keys: dict[uuid.UUID, str] = {}

        def order(f: CharFact) -> tuple:
            return (*f.pos, f.char_end, f.category, f.attribute, f.mention, f.value_text or "")

        for f in sorted(facts, key=order):
            chapter = str(f.chapter_id) if f.chapter_id else f"#{f.chapter_number}"
            n = seen.get((chapter, f.raw_text), 0)
            seen[(chapter, f.raw_text)] = n + 1
            self._keys[f.id] = f"{chapter}|{f.raw_text}|{n}"

    def fingerprint(self, kind: str, *facts: CharFact) -> str:
        return make_fingerprint(CHECKER_NAME, kind, *(self._keys[f.id] for f in facts))


def _mark(value: str | None) -> tuple[str, str, str] | None:
    """ "左脸·刀疤" -> ("左", "脸", "刀疤"): side, part, kind. Only marks whose side is
    written can be compared (step 5-1d): "布满伤疤的脸" beside "右脸的疤", or "背后"
    beside "背上", are the same mark told loosely, not a mark that moved."""
    if not value or "·" not in value:
        return None
    where, what = (x.strip() for x in value.split("·", 1))
    side = next((c for c in "左右" if c in where), None)
    if side is None:
        return None
    part = where.replace(side, "")
    for word in ("半边", "边", "侧"):  # "右半边脸", "左侧脸": the face
        part = part.replace(word, "")
    return side, part, what


def _appearance(facts: list[CharFact], keys: _Keys) -> list[ConsistencyIssue]:
    groups: dict[tuple, list[CharFact]] = {}
    for f in facts:
        if f.is_flashback or f.is_speculative or not own_look(f.qualifiers):
            continue
        if f.attribute == "mark":
            mark = _mark(f.value_text)
            if mark is None:
                continue
            # Marks of different kinds (a scar and a mole) can both be there; the same
            # kind on the other side of the same part is the candidate ("左脸刀疤" ->
            # "右脸刀疤").
            _side, part, kind = mark
            groups.setdefault((f.character_id, "mark", part, kind), []).append(f)
        elif f.value_text:
            groups.setdefault((f.character_id, f.attribute), []).append(f)

    issues = []
    for (_, attribute, *_), group in groups.items():
        group.sort(key=lambda f: f.pos)
        for a, b in pairwise(group):
            if attribute == "mark":
                differs = _mark(a.value_text)[0] != _mark(b.value_text)[0]  # type: ignore[index]  # sides
            else:
                differs = a.value_text != b.value_text
            if not differs:
                continue
            issues.append(
                ConsistencyIssue(
                    checker=CHECKER_NAME, issue_type=IssueType.FACT_APPEARANCE,
                    confidence=Confidence.SUSPECTED_REVIEW,
                    description=(
                        f"{b.character_name}的{_TRAIT.get(attribute, attribute)}前后不一："
                        f"第 {a.chapter_number} 章为「{a.value_text}」，"
                        f"第 {b.chapter_number} 章为「{b.value_text}」。"
                        "如果原文交代了变化（染发、易容、变身等），可以忽略。"
                    ),
                    evidence=[_evidence(a), _evidence(b)],
                    fingerprint=keys.fingerprint("appearance", a, b),
                    subjects=[str(b.character_id)],
                )
            )  # fmt: skip
    return issues


def _kinship(
    facts: list[CharFact], names: dict[str, uuid.UUID], keys: _Keys
) -> list[ConsistencyIssue]:
    """Statements about one pair, read in the same direction, compared in order."""
    pairs: dict[tuple[uuid.UUID, uuid.UUID], list[tuple[CharFact, kinship.Kin]]] = {}
    who: dict[uuid.UUID, str] = {}
    for f in facts:
        if f.is_speculative or f.attribute not in kinship.RELATIONS:
            continue
        other = names.get(f.qualifiers.get("other_name", "")) or names.get(
            f.qualifiers.get("other_mention", "")
        )
        if other is None or other == f.character_id:
            continue
        who[f.character_id] = f.character_name
        who.setdefault(other, f.qualifiers.get("other_name") or f.qualifiers["other_mention"])
        described = kinship.kin(f.attribute)
        if str(f.character_id) <= str(other):
            key = (f.character_id, other)
        else:
            key, described = (other, f.character_id), described.inverse()
        pairs.setdefault(key, []).append((f, described))

    issues = []
    for (a_id, b_id), statements in pairs.items():
        statements.sort(key=lambda s: s[0].pos)
        for (fa, ka), (fb, kb) in pairwise(statements):
            why = kinship.conflict(ka, kb)
            if why is None or (_loose(fa, fb) and _loose_about(why, ka)):
                continue
            issues.append(
                ConsistencyIssue(
                    checker=CHECKER_NAME, issue_type=IssueType.CHARACTER_KINSHIP,
                    confidence=Confidence.SUSPECTED_REVIEW,
                    description=(
                        f"{who[a_id]}与{who[b_id]}的亲属关系前后矛盾（{why}）："
                        f"第 {fa.chapter_number} 章说{_says(fa)}，"
                        f"第 {fb.chapter_number} 章说{_says(fb)}。"
                    ),
                    evidence=[_evidence(fa), _evidence(fb)],
                    fingerprint=keys.fingerprint("kinship", fa, fb),
                    subjects=[str(a_id), str(b_id)],
                )
            )  # fmt: skip
    return issues


def _loose(*facts: CharFact) -> bool:
    return any(f.qualifiers.get("address") for f in facts)


def _loose_about(why: str, k: kinship.Kin) -> bool:
    """What a form of address may blur (step 5-1d): among the same generation, sibling
    or cousin ("哥哥" said to a cousin), and which kind of cousin. Not who is elder
    ("哥哥" is said to an elder), the generation, the side or the sex, and not a parent
    against an uncle: "爹" is said to a father."""
    return why == "堂 / 表不同" or (why == "关系类型不同" and k.generations == 0)


def _says(f: CharFact) -> str:
    other = f.qualifiers.get("other_name") or f.qualifiers.get("other_mention") or "?"
    return f"{f.character_name}是{other}的{kinship.label(f.attribute).split(' / ')[0]}"


def _revival(facts: list[CharFact], keys: _Keys) -> list[ConsistencyIssue]:
    """The first time someone appears in person in a chapter after the one where they
    died; one issue per character. Within the death's own chapter the order of events is
    too loose to judge ("临死前他说……")."""
    deaths: dict[uuid.UUID, CharFact] = {}
    for f in sorted(facts, key=lambda f: f.pos):
        if f.attribute == "died" and not f.is_speculative:
            deaths.setdefault(f.character_id, f)
    issues = []
    for character_id, death in deaths.items():
        later = sorted(
            (
                f
                for f in facts
                if f.character_id == character_id
                and f.attribute == "present"
                and f.chapter_number > death.chapter_number
            ),
            key=lambda f: f.pos,
        )
        if not later:
            continue
        seen = later[0]
        issues.append(
            ConsistencyIssue(
                checker=CHECKER_NAME, issue_type=IssueType.TIMELINE_REVIVAL,
                confidence=Confidence.SUSPECTED_REVIEW,
                description=(
                    f"{seen.character_name}在第 {death.chapter_number} 章已经死亡，"
                    f"第 {seen.chapter_number} 章又亲自出场。"
                    "如果原文交代了复活、假死或鬼魂等，可以忽略。"
                ),
                evidence=[_evidence(death), _evidence(seen)],
                fingerprint=keys.fingerprint("revival", death, seen),
                subjects=[str(character_id)],
            )
        )  # fmt: skip
    return issues


def check_character_facts(
    facts: list[CharFact], names: dict[str, uuid.UUID]
) -> list[ConsistencyIssue]:
    """`names`: every canonical name and alias of the book's characters -> the character,
    to find the other side of a relation."""
    keys = _Keys(facts)
    by_category: dict[str, list[CharFact]] = {}
    for f in facts:
        by_category.setdefault(f.category, []).append(f)
    return [
        *_appearance(by_category.get("appearance", []), keys),
        *_kinship(by_category.get("kinship", []), names, keys),
        *_revival(by_category.get("life", []), keys),
    ]
