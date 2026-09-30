"""Character facts (step 5-1): fixed features of appearance, blood relations, deaths, and
who appears in person — a second kind of extraction beside ages, with its own prompt
(`character_facts_v2`; v1 and the failed v3 are kept for the record), model calls and
stored reading.

As with ages, the model labels (flashback, guess, disguise, temporary, generic, a form
of address, a word for two people) rather than being told to leave things out, and code
guards what has a clear form: every quote must be found in the text; a relation comes
from the word the text uses, looked up in a table, not from the model (step 5-1d: the
model got who-is-whose wrong); a colour comes from the colour word in the quote; the
quote must name the body part; a corpse is nobody appearing in person.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import resources
from typing import Any, Literal

from pydantic import BaseModel, Field

from webfic.extraction.locator import locate, normalized
from webfic.facts.kinship import COLLECTIVE, RANK, RELATIONS, term_relation
from webfic.ingest.chunker import chunk_text
from webfic.llm.base import LLMClient, LLMError, ProviderError, Tier, Usage

log = logging.getLogger(__name__)

# v3 (step 5-1e: who speaks to whom in dialogue) was a failed trial and is kept for the
# record: it split chained terms ("母亲的亲哥哥") more often, for little gain.
PROMPT_VERSION = "character_facts_v2"
# The code guards applied to a reply before it is stored (_normalised, _valid): a stored
# reading holds what they kept, so changing them must change the version too, or stored
# readings would be reused unguarded (step 5-1d). Bump on every change.
GUARDS_VERSION = 4
# A piece whose reply is cut off is read in halves, down to this size (step 5-1c).
MIN_SPLIT = 1000
SPLIT_OVERLAP = 200

TraitAttribute = Literal["eye_color", "hair_color", "mark"]
# The colour words the model is asked to use, so "碧色" and "绿" compare as the same.
COLORS = ["黑", "白", "银", "灰", "金", "黄", "红", "棕", "蓝", "绿", "紫", "粉", "异色"]


class TraitStatement(BaseModel):
    mention: str = Field(description="原文中对该角色的称呼")
    resolved_name: str | None = None
    raw_text: str
    attribute: TraitAttribute
    # A colour from COLORS for eyes and hair; "部位·类型" for a mark ("左脸·刀疤").
    value: str
    is_flashback: bool = False
    speculative: bool = False  # a guess, a misreading ("灯下看着像是蓝眼睛"), hearsay
    disguised: bool = False  # disguise, transformation, illusion: not their own looks
    # A passing state: eyes glowing with a technique, reddened with rage, a battle form
    # (step 5-1d). Not compared.
    temporary: bool = False
    generic: bool = False  # a group or a rule, not one character


class KinshipStatement(BaseModel):
    """`mention` is the `relation` of `other_mention`."""

    mention: str
    resolved_name: str | None = None
    other_mention: str
    other_resolved_name: str | None = None
    relation: str  # a key of webfic.facts.kinship.RELATIONS
    raw_text: str
    speculative: bool = False
    # A polite or figurative address ("大哥" for a friend), sworn or adoptive kin (义父,
    # 干娘, 师父 as "父"): not blood. Labelled, then left out.
    not_blood: bool = False
    # Said to someone's face in dialogue ("哥哥！"), not stated: loose about cousins and
    # order, so only generations are compared (step 5-1d).
    address: bool = False


class KinshipMention(BaseModel):
    """What the model is asked for (step 5-1d): the word the text uses, who is called so
    and whose it is. The relation is looked up from the word, in code."""

    term: str = Field(description="原文里的称谓词，逐字")
    person: str  # who is called so ("爹" in "我爹" is the speaker's father)
    person_resolved: str | None = None
    of: str  # whose it is
    of_resolved: str | None = None
    raw_text: str
    speculative: bool = False
    not_blood: bool = False
    address: bool = False  # said to the person's face
    collective: bool = False  # a word for both at once ("父子俩", "姐弟")

    def statement(self) -> "KinshipStatement | None":
        """The relation it states, or None: a word for two people, a word the table
        does not know, a word the quote does not contain, or the end of a chain taken
        alone ("哥哥" out of "罗鸣母亲的哥哥": not 罗鸣's brother)."""
        term = self.term.strip()
        relation = term_relation(term)
        if self.collective or relation is None or term not in self.raw_text:
            return None
        # "母亲的亲哥哥" taken as "哥哥": the rank and emphasis in front go with it.
        before = self.raw_text[: self.raw_text.index(term)].rstrip(RANK)
        if before.endswith("的") and _kin_at_end(before[:-1]):
            return None
        return KinshipStatement(
            mention=self.person, resolved_name=self.person_resolved, other_mention=self.of,
            other_resolved_name=self.of_resolved, relation=relation, raw_text=self.raw_text,
            speculative=self.speculative, not_blood=self.not_blood, address=self.address,
        )  # fmt: skip


def _kin_at_end(text: str) -> bool:
    """Whether a word for a relative ends the text ("罗鸣母亲", "我娘"). One character
    counts only after a pronoun: the last one of "小姐" or "师父" is no relative."""
    words = [text[-n:] for n in range(2, min(len(text), 4) + 1)]
    if len(text) == 1 or text[-2:-1] in ("我", "你", "他", "她", "其", "咱", "俺"):
        words.append(text[-1:])
    return any(term_relation(w) for w in words)


class DeathStatement(BaseModel):
    mention: str
    resolved_name: str | None = None
    raw_text: str
    speculative: bool = False  # rumour, presumed dead, fate unknown


class PresenceStatement(BaseModel):
    """The character is there in person, now, and acts or speaks."""

    mention: str
    resolved_name: str | None = None
    raw_text: str


class CharacterFacts(BaseModel):
    """The model's reply."""

    traits: list[TraitStatement] = []
    kinship: list[KinshipMention] = []
    deaths: list[DeathStatement] = []
    presence: list[PresenceStatement] = []


Statement = TraitStatement | KinshipStatement | DeathStatement | PresenceStatement
Section = Literal["traits", "kinship", "deaths", "presence"]
_MODELS: dict[Section, type[BaseModel]] = {
    "traits": TraitStatement,
    "kinship": KinshipStatement,
    "deaths": DeathStatement,
    "presence": PresenceStatement,
}


@dataclass
class Located:
    section: Section
    statement: Any  # one of the statement models
    char_start: int
    char_end: int


@dataclass
class FactsReading:
    """One chapter's character facts, positions in chapter coordinates."""

    facts: list[Located] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost_usd: Decimal = Decimal(0)
    llm_calls: int = 0
    cache_hits: int = 0
    reused: bool = False
    revealed_names: list = field(default_factory=list)  # none: ages carry revelations


def load_prompt(name: str = PROMPT_VERSION) -> str:
    return resources.files("webfic.extraction.prompts").joinpath(f"{name}.txt").read_text("utf-8")


def version(system_prompt: str, chunk_size: int, chunk_overlap: int) -> str:
    key = f"{PROMPT_VERSION}|g{GUARDS_VERSION}|{chunk_size}|{chunk_overlap}|{system_prompt}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


# A blood relation must be stated in words: its quote carries a kinship term. A relation
# the model inferred from the context ("拿刀抵着知寒的脖子" read as "his elder brother")
# has none and is dropped (step 5-1).
KIN_TERMS = (
    "父", "母", "爹", "娘", "爸", "妈", "兄", "弟", "姐", "妹", "哥", "叔", "伯", "舅",
    "姑", "姨", "爷", "奶", "祖", "孙", "甥", "侄", "姥", "外公", "外婆", "胞",
    "儿子", "女儿", "之子", "之女", "长子", "次子", "幼子", "独子", "长女", "次女", "幼女",
)  # fmt: skip
# Single characters such as 子 / 女 / 儿 / 亲 / 表 / 堂 are left out: they occur in
# ordinary words (脖子, 女人, 这儿, 亲眼, 表情, 祠堂) and would let inferred relations pass.


# The quote of a colour must name what is coloured: "素白色的长袍" is no hair colour,
# "通体漆黑" no eye colour (step 5-1d).
BODY_PART = {"eye_color": ("眼", "瞳", "眸"), "hair_color": ("发", "鬓", "须", "头", "丝")}
# Colour words in a quote -> the colour: the same word always gives the same colour, so
# "琥珀色" is never yellow once and brown the next time (step 5-1d). Longer words first.
COLOR_WORDS = [
    ("琥珀", "金"), ("碧蓝", "蓝"), ("湛蓝", "蓝"), ("蔚蓝", "蓝"), ("银白", "银"),
    ("雪白", "白"), ("霜白", "白"), ("花白", "白"), ("乌黑", "黑"), ("漆黑", "黑"), ("墨", "黑"),
    ("碧", "绿"), ("翠", "绿"), ("绿", "绿"), ("黑", "黑"), ("白", "白"), ("银", "银"),
    ("灰", "灰"), ("金", "金"), ("黄", "黄"), ("赤", "红"), ("红", "红"), ("血", "红"),
    ("棕", "棕"), ("褐", "棕"), ("栗", "棕"), ("蓝", "蓝"), ("紫", "紫"), ("粉", "粉"),
    # Single characters of the literary register ("乌发", "玄瞳", "雪发"); "青丝" names
    # hair, not a colour, and is left out.
    ("乌", "黑"), ("玄", "黑"), ("雪", "白"), ("霜", "白"), ("靛", "蓝"), ("茶", "棕"),
]  # fmt: skip
CORPSE = ("尸体", "尸首", "尸身", "遗体", "遗骸", "灵柩", "棺")
# What else a colour word can describe: the face, the skin, clothes (step 5-1e).
_OTHER_PARTS = ("脸", "面", "肤", "皮", "衣", "袍", "裙", "衫", "甲", "袖")


def colours_in(quote: str) -> set[str]:
    """The colours a quote names ("银发金瞳": silver and gold)."""
    if "异色" in quote:
        return {"异色"}
    found, rest = set(), quote
    for word, colour in COLOR_WORDS:
        if word in rest:
            found.add(colour)
            rest = rest.replace(word, "")
    return found


def describes(quote: str, attribute: str) -> bool:
    """Whether some colour word in the quote belongs to this body part: the body-part
    word nearest to it is one of the part's (step 5-1e: in "银发男人的瞳孔" the silver is
    the hair's, not the eyes'; in "漆黑的脸，阴冷的眸子" the black is the face's)."""
    parts = {c: a for a, cs in BODY_PART.items() for c in cs} | dict.fromkeys(_OTHER_PARTS, "")
    where = [(i, parts[c]) for i, c in enumerate(quote) if c in parts]
    rest = quote
    for word, _ in COLOR_WORDS:
        start = rest.find(word)
        while start != -1:
            end = start + len(word)
            # Nearest first; at equal distance the part after it ("黑发黑眼": the second
            # black is the eyes'), as a colour usually comes before what it describes.
            near = min(
                where,
                key=lambda w: (w[0] - end + 1 if w[0] >= end else start - w[0], w[0] < start),
                default=None,
            )
            if near is not None and near[1] == attribute:
                return True
            rest = rest[:start] + " " * len(word) + rest[end:]
            start = rest.find(word)
    return False


def colour_in(quote: str) -> str | None:
    """The one colour the quote names, or None (none, or several: "由黑转红")."""
    found = colours_in(quote)
    return found.pop() if len(found) == 1 else None


def _looks_like_kin(name: str | None) -> bool:
    """A word for a relative taken for a person ("爹娘", "父亲")."""
    return bool(name) and (term_relation(name) is not None or any(c in name for c in COLLECTIVE))


def _valid(section: Section, s: Any) -> bool:
    """Code guards on what has a clear form (the prompt only asks)."""
    match section:
        case "traits":
            if s.generic or not s.value.strip():
                return False
            if s.attribute in ("eye_color", "hair_color"):
                return s.value.strip() in COLORS and (
                    "异色" in s.raw_text or describes(s.raw_text, s.attribute)
                )
            return "·" in s.value  # 部位·类型
        case "kinship":
            # Nobody is "the father of 爹娘": both sides must be people, unless the model
            # said who the word means.
            unnamed = (_looks_like_kin(s.mention) and not s.resolved_name) or (
                _looks_like_kin(s.other_mention) and not s.other_resolved_name
            )
            return (
                s.relation in RELATIONS
                and not s.not_blood
                and not unnamed
                and normalized(s.mention) != normalized(s.other_mention)
                and any(term in s.raw_text for term in KIN_TERMS)
            )
        case "presence":
            return not any(word in s.raw_text for word in CORPSE)
        case _:
            return True


def _normalised(section: Section, s: Any) -> Any:
    """A reply's statement as stored: a relation looked up from its word, a colour from
    the colour word in the quote (None: dropped)."""
    if section == "kinship":
        return s.statement()
    if section == "traits" and s.attribute in ("eye_color", "hair_color"):
        # A colour the quote does not name was guessed ("精致的眸子" read as black):
        # dropped. Of several named ("银发金瞳"), the model says which is whose.
        found = colours_in(s.raw_text)
        if len(found) == 1:
            return s.model_copy(update={"value": found.pop()})
        return s if s.value in found else None
    return s


def _person(section: Section, s: Any) -> str:
    who = (s.resolved_name or s.mention).strip()
    if section == "kinship":
        who += "|" + (s.other_resolved_name or s.other_mention).strip() + "|" + s.relation
    if section == "traits":
        who += "|" + s.attribute
    return who


async def extract(
    llm: LLMClient,
    *,
    chapter_number: int,
    text: str,
    known_characters: str,
    system_prompt: str,
    chunk_size: int,
    chunk_overlap: int,
    offset: int = 0,
) -> FactsReading:
    """Extract from `text` (the chapter's story, after its author's notes are set
    aside); positions are `offset` + position in `text`."""
    result = FactsReading()
    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    seen: set[tuple[int, int, str, str]] = set()
    # (text, where it starts in `text`, how it is labelled for the model)
    work = [
        (c.text, c.start, f"，第 {c.index + 1}/{len(chunks)} 段" if len(chunks) > 1 else "")
        for c in chunks
    ]
    while work:
        piece, piece_start, part = work.pop(0)
        try:
            reply = await llm.generate_json(
                tier=Tier.EXTRACT,
                system=system_prompt,
                user=f"已知角色：\n{known_characters}\n\n正文（第 {chapter_number} 章{part}）：\n"
                f"{piece}",
                schema=CharacterFacts,
                purpose="extract.character_facts",
            )
        except ProviderError:
            raise
        except LLMError:
            # A chapter crowded with people can need more output than one reply holds:
            # the JSON is cut off (step 5-1c). Its halves are read instead.
            if len(piece) < MIN_SPLIT:
                raise
            halves = chunk_text(piece, size=len(piece) // 2 + SPLIT_OVERLAP, overlap=SPLIT_OVERLAP)
            log.info("chapter %s: character facts cut off, reading %d parts", chapter_number,
                     len(halves))  # fmt: skip
            work[:0] = [
                (h.text, piece_start + h.start, f"{part}（第 {h.index + 1}/{len(halves)} 小段）")
                for h in halves
            ]
            continue
        result.usage += reply.usage
        result.cost_usd += reply.cost_usd
        result.llm_calls += 1
        result.cache_hits += int(reply.cache_hit)
        for section in _MODELS:
            last_end: dict[str, int] = {}
            for said in getattr(reply.data, section):
                st = _normalised(section, said)
                if st is None:
                    result.dropped.append(said.raw_text)
                    continue
                span = locate(piece, st.raw_text, start_from=last_end.get(st.raw_text, 0))
                if span is not None:
                    last_end[st.raw_text] = span[1]
                if span is None or not _valid(section, st):
                    result.dropped.append(st.raw_text)
                    continue
                start = span[0] + piece_start + offset
                end = span[1] + piece_start + offset
                key = (start, end, section, _person(section, st))
                if key in seen:  # listed twice, or read again in the overlap
                    continue
                seen.add(key)
                result.facts.append(Located(section, st, start, end))
    result.facts.sort(key=lambda f: (f.char_start, f.section))
    if result.dropped:
        log.info("chapter %s: dropped %d character facts", chapter_number, len(result.dropped))
    return result


def dump(reading: FactsReading, offset: int = 0) -> dict[str, Any]:
    """Stored in story coordinates, like ages (step 4.6)."""
    return {
        "coords": "story",
        "facts": [
            {
                "section": f.section,
                "statement": f.statement.model_dump(mode="json"),
                "start": f.char_start - offset,
                "end": f.char_end - offset,
            }
            for f in reading.facts
        ],
        "dropped": list(reading.dropped),
    }


def load(data: dict[str, Any], offset: int = 0) -> FactsReading:
    return FactsReading(
        facts=[
            Located(
                f["section"],
                _MODELS[f["section"]].model_validate(f["statement"]),
                f["start"] + offset,
                f["end"] + offset,
            )
            for f in data["facts"]
        ],
        dropped=list(data["dropped"]),
    )


def positions_hold(reading: FactsReading, content: str) -> bool:
    return all(
        0 <= f.char_start <= f.char_end <= len(content)
        and normalized(content[f.char_start : f.char_end]) == normalized(f.statement.raw_text)
        for f in reading.facts
    )
