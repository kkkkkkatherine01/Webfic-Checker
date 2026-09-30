"""Character facts (step 5-1): the prompt's examples, the code guards, the rows written,
and each kind of extraction being shown only the characters it named."""

import json
import re
import uuid

import pytest
from sqlalchemy import select

from tests.fakes import FakeBackend, MemoryCallStore
from tests.story01_ideal import GOLDEN_DIR, needs_golden
from tests.test_chapters import USER, World, respond
from webfic.db.models import Character, FactRow
from webfic.extraction.character_facts import CharacterFacts, load_prompt
from webfic.extraction.resolver import CharacterIndex, KnownCharacter
from webfic.facts.kinship import conflict, kin, term_relation
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig
from webfic.memory import core

PROMPT = load_prompt()
EXAMPLES = re.findall(r"## 示例 (\d+)\n.*?正文：\n(.*?)\n\n输出：\n(\{.*?\]\})", PROMPT, re.S)


def test_the_prompt_has_examples():
    assert len(EXAMPLES) >= 3


@pytest.mark.parametrize(("number", "text", "output"), EXAMPLES)
def test_examples_are_valid_quote_their_own_text_and_use_known_terms(number, text, output):
    from webfic.extraction.character_facts import _normalised, _valid

    data = CharacterFacts.model_validate(json.loads(output))
    quotes = [s.raw_text for part in (data.traits, data.kinship, data.deaths, data.presence)
              for s in part]  # fmt: skip
    assert [q for q in quotes if q not in text] == [], f"示例 {number}"
    # Every term of a blood relation is in the table; a word for two is not.
    for k in data.kinship:
        assert k.term in k.raw_text
        assert (term_relation(k.term) is None) == k.collective, k.term
    # The code guards keep what the examples show as kept (step 5-1d): what an example
    # shows must not be dropped by code, or the model learns to say what is thrown away.
    for t in data.traits:
        if not t.generic:
            assert _valid("traits", _normalised("traits", t)), t.raw_text
            assert _normalised("traits", t).value == t.value, t.raw_text
    for k in data.kinship:
        s = _normalised("kinship", k)
        assert (s is not None and _valid("kinship", s)) == (not k.collective and not k.not_blood)


def test_example_relations_are_read_the_right_way_round():
    output = next(o for n, _, o in EXAMPLES if "外孙女" in o)
    data = CharacterFacts.model_validate(json.loads(output))
    said = {
        (s.resolved_name, s.relation, s.other_resolved_name)
        for k in data.kinship
        if (s := k.statement()) is not None and not s.not_blood
    }
    assert said == {
        ("岑远山", "father", "岑小满"),
        ("钱守业", "maternal_grandfather", "岑小满"),
        ("钱柏", "maternal_uncle", "岑小满"),
        ("岑小满", "daughters_daughter", "钱守业"),
        ("钱菱", "maternal_aunt", "岑小满"),
    }


@needs_golden
def test_no_golden_story_text_leaks_into_the_prompt():
    fragments = set(re.findall(r"[一-鿿]{6,}", PROMPT))
    leaks = []
    for story in sorted(GOLDEN_DIR.glob("story*/text.txt")):
        text = story.read_text("utf-8-sig")
        leaks += [(story.parent.name, f) for f in fragments if f in text]
    assert leaks == []


# --- kinship --------------------------------------------------------------------------------


def test_a_relation_and_its_inverse_agree_and_contradictions_are_named():
    father = kin("father")  # A is B's father
    assert conflict(father, kin("father")) is None
    assert conflict(kin("son").inverse(), kin("father")) is None  # B is A's son
    assert conflict(kin("elder_brother").inverse(), father) == "辈分不同"
    assert conflict(kin("paternal_uncle"), kin("maternal_uncle")) == "父系 / 母系不同"
    assert conflict(kin("brother"), kin("elder_brother")) is None
    assert conflict(kin("elder_brother"), kin("younger_brother")) == "长幼不同"
    assert conflict(kin("father"), kin("mother")) == "性别不同"
    assert conflict(kin("tang_cousin"), kin("biao_cousin")) == "堂 / 表不同"


# --- the kind in the pipeline -----------------------------------------------------------------

BOOK = """第1章 起
林远今年18岁。林远有一双绿眼睛，他的父亲林震走了进来。
第2章 承
林远今年19岁。林震去年已经病故。
"""


def facts(messages):
    text = messages[1].content
    out = {"traits": [], "kinship": [], "deaths": [], "presence": []}
    if "绿眼睛" in text:
        out["traits"] = [
            {"mention": "林远", "resolved_name": "林远", "raw_text": "一双绿眼睛",
             "attribute": "eye_color", "value": "绿"},
            {"mention": "林远", "resolved_name": "林远", "raw_text": "林远有一双绿眼睛",
             "attribute": "hair_color", "value": "绿"},  # names no hair: dropped
        ]  # fmt: skip
        out["kinship"] = [
            {"term": "父亲", "person": "林震", "person_resolved": "林震", "of": "他",
             "of_resolved": "林远", "raw_text": "他的父亲林震"},
            {"term": "义父", "person": "林震", "of": "林远",
             "raw_text": "他的父亲林震"},  # the word is not in the quote: dropped
            {"term": "父子", "person": "林震", "of": "林远", "raw_text": "他的父亲林震",
             "collective": True},  # a word for both: dropped
        ]  # fmt: skip
        out["presence"] = [{"mention": "林震", "resolved_name": "林震", "raw_text": "林震走了进来"}]
    else:
        out["deaths"] = [
            {"mention": "林震", "resolved_name": "林震", "raw_text": "林震去年已经病故"}
        ]
    return out


async def test_character_facts_become_rows_and_new_characters_stay_out_of_the_age_prompt(
    factory,
):
    world = World(factory)
    world.backend = FakeBackend(respond, facts=facts)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    world.llm = JsonLLMClient(world.backend, tiers, store=MemoryCallStore())
    await world.load(BOOK)
    async with factory() as session:
        rows = (
            await session.execute(
                select(FactRow.chapter_number, FactRow.category, FactRow.attribute,
                       FactRow.value_text, FactRow.qualifiers, Character.canonical_name)
                .join(Character, Character.id == FactRow.character_id)
                .where(FactRow.category != "age")
                .order_by(FactRow.chapter_number, FactRow.char_start)
            )
        ).all()  # fmt: skip
        kinds = dict(
            (await session.execute(select(Character.canonical_name, Character.kind))).all()
        )
    assert [(*r[:4], r[5]) for r in rows] == [
        (1, "appearance", "eye_color", "绿", "林远"),
        (1, "kinship", "father", None, "林震"),
        (1, "life", "present", None, "林震"),
        (2, "life", "died", None, "林震"),
    ]
    assert rows[1][4] == {"other_name": "林远", "other_mention": "他"}
    assert kinds == {"林远": "age", "林震": "character_facts"}
    # 林震 was named by character facts in chapter 1: chapter 2's age prompt does not
    # list him (its requests stay what they were before character facts existed); the
    # character-facts prompt, coming after ages, lists both.
    age_prompt = world.backend.calls[1][1].content
    assert "林远" in age_prompt and "林震" not in age_prompt.split("正文")[0]
    facts_prompt = world.backend.facts_calls[1][1].content.split("正文")[0]
    assert "林震" in facts_prompt and "林远" in facts_prompt
    # Core keeps the latest stated features and the death.
    async with factory() as session:
        lin = await core.get_character(session, user_id=USER, book_id=world.book_id, name="林远")
        zhen = await core.get_character(session, user_id=USER, book_id=world.book_id, name="林震")
    assert lin.traits["eye_color"].value == "绿" and lin.died_chapter is None
    assert (zhen.died_chapter, zhen.died_quote) == (2, "林震去年已经病故")


def test_each_kind_is_shown_what_it_and_earlier_kinds_named_and_the_authors_aliases():
    index = CharacterIndex(
        [
            KnownCharacter(
                uuid.uuid4(),
                "林远",
                ["远儿", "小远"],
                "age",
                {"远儿": "age", "小远": "user"},
            )
        ]
    )
    index.kind = "character_facts"
    index.resolve("林震", "林震")
    index.resolve("远哥", "林远")  # an alias named by character facts
    assert index.for_prompt({"age"}) == "- 林远：小远、远儿"
    # A later kind is shown what the earlier ones named too.
    assert index.for_prompt({"age", "character_facts"}) == "- 林远：小远、远儿、远哥\n- 林震"
    assert index.for_prompt() == "- 林远：小远、远儿、远哥\n- 林震"


def test_a_relation_must_be_stated_with_a_kinship_term():
    from webfic.extraction.character_facts import KinshipStatement, _valid

    def relation(quote):
        return KinshipStatement(
            mention="林知寒", other_mention="林知白", relation="elder_brother", raw_text=quote
        )

    assert _valid("kinship", relation("哥哥林知寒比他大三岁"))
    assert not _valid("kinship", relation("拿刀抵着林知寒的脖子"))  # inferred, not stated


async def test_a_reply_cut_off_is_read_again_in_halves():
    # Step 5-1c: a chapter crowded with people overflowed one reply (the JSON was cut off).
    from webfic.extraction.character_facts import extract

    text = ("林远走进院子。" * 200) + "林远有一双绿色的眼睛。" + ("苏晴在院中练剑。" * 200)
    shown = []

    def answer(messages):
        piece = messages[1].content.rsplit("）：\n", 1)[1]
        shown.append(len(piece))
        if len(piece) > 3000:
            return '{"traits": [{"mention": "林远"'  # cut off
        if "绿色的眼睛" not in piece:
            return {}
        return {
            "traits": [
                {
                    "mention": "林远",
                    "resolved_name": "林远",
                    "raw_text": "一双绿色的眼睛",
                    "attribute": "eye_color",
                    "value": "绿",
                }
            ]
        }

    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash")}
    llm = JsonLLMClient(FakeBackend(respond, facts=answer), tiers, store=MemoryCallStore())
    reading = await extract(
        llm, chapter_number=1, text=text, known_characters="", system_prompt=load_prompt(),
        chunk_size=8000, chunk_overlap=500,
    )  # fmt: skip
    assert [f.statement.value for f in reading.facts] == ["绿"]
    assert text[reading.facts[0].char_start : reading.facts[0].char_end] == "一双绿色的眼睛"
    assert max(shown) > 3000 and min(shown) < 3000


# --- adding a kind never changes what the earlier ones are asked (step 5-1c) -----------------

LATER_BOOK = """第1章 起
林远今年18岁。他的父亲林震走了进来。
第2章 承
林震今年45岁。林远今年19岁。
第3章 转
林震今年46岁。震叔今年46岁。
"""


def father_named_first(messages):
    """Character facts name 林震 (and his alias 震叔) before any age mentions him."""
    text = messages[1].content.rsplit("）：\n", 1)[1]
    out = {"traits": [], "kinship": [], "deaths": [], "presence": []}
    if "父亲林震" in text:
        out["kinship"] = [
            {"term": "父亲", "person": "林震", "person_resolved": "林震", "of": "他",
             "of_resolved": "林远", "raw_text": "他的父亲林震"},
        ]  # fmt: skip
        out["presence"] = [{"mention": "震叔", "resolved_name": "林震", "raw_text": "林震走了进来"}]
    return out


def age_requests(backend):
    return [m[1].content for m in backend.calls]


async def test_age_requests_are_the_same_with_or_without_character_facts(factory, monkeypatch):
    from webfic.extraction.kinds import AgeKind
    from webfic.services import imports

    with_facts = World(factory)
    with_facts.backend = FakeBackend(respond, facts=father_named_first)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    with_facts.llm = JsonLLMClient(with_facts.backend, tiers, store=MemoryCallStore())
    await with_facts.load(LATER_BOOK)

    monkeypatch.setattr(imports, "KINDS", [AgeKind()])
    ages_only = World(factory)
    await ages_only.load(LATER_BOOK)
    assert age_requests(with_facts.backend) == age_requests(ages_only.backend)
    # 林震 was named by character facts, then used by ages: credited to ages.
    async with factory() as session:
        kinds = dict(
            (
                await session.execute(
                    select(Character.canonical_name, Character.kind).where(
                        Character.book_id == with_facts.book_id
                    )
                )
            ).all()
        )
    assert kinds["林震"] == "age"


async def test_undoing_a_chapter_undoes_its_promotions(factory):
    from webfic.memory import events

    world = World(factory)
    world.backend = FakeBackend(respond, facts=father_named_first)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    world.llm = JsonLLMClient(world.backend, tiers, store=MemoryCallStore())
    await world.load(LATER_BOOK)
    async with factory() as session:
        await events.undo_from_chapter(
            session, user_id=USER, book_id=world.book_id, chapter_number=2
        )
        kind = await session.scalar(
            select(Character.kind).where(
                Character.book_id == world.book_id, Character.canonical_name == "林震"
            )
        )
        await session.rollback()
    assert kind == "character_facts"  # as after chapter 1


# --- code guards (step 5-1d) ------------------------------------------------------------------


def test_kinship_terms_are_looked_up_with_their_ranks_and_words_for_two_are_not():
    assert term_relation("五舅舅") == "maternal_uncle"
    assert term_relation("外甥女") == "sisters_daughter"
    assert term_relation("二叔") == "paternal_uncle"
    assert term_relation("堂兄") == "tang_cousin"
    # A chain is composed (step 5-1d); one the table cannot name is not.
    assert term_relation("母亲的亲哥哥") == "maternal_uncle"
    assert term_relation("父亲的父亲") == "paternal_grandfather"
    assert term_relation("大伯的女儿") == "tang_cousin"
    assert term_relation("哥哥的父亲") is None
    for word in ("父子俩", "姐弟", "三小姐", "老子", "夫妇"):
        assert term_relation(word) is None, word


def test_a_mention_is_read_as_the_relation_of_its_word():
    from webfic.extraction.character_facts import KinshipMention

    said = KinshipMention(term="我爹", person="岑远山", of="我", of_resolved="岑小满",
                          raw_text="这是我爹")  # fmt: skip
    assert said.statement() is None  # "我爹" is not a term: the table does not know it
    said = said.model_copy(update={"term": "爹"})
    s = said.statement()
    assert (s.mention, s.relation, s.other_resolved_name) == ("岑远山", "father", "岑小满")
    assert said.model_copy(update={"term": "娘"}).statement() is None  # not in the quote
    # The end of a chain taken alone is not the relation.
    chain = KinshipMention(term="哥哥", person="罗晟", of="罗鸣", raw_text="他是罗鸣母亲的哥哥")
    assert chain.statement() is None
    assert chain.model_copy(update={"raw_text": "他是罗鸣母亲的亲哥哥"}).statement() is None
    s = chain.model_copy(update={"term": "母亲的哥哥"}).statement()
    assert (s.mention, s.relation, s.other_mention) == ("罗晟", "maternal_uncle", "罗鸣")
    assert KinshipMention(term="姑姑", person="苗秀", of="她", raw_text="她的姑姑苗秀").statement()


def test_a_colour_comes_from_the_quote_and_the_quote_names_the_body_part():
    from webfic.extraction.character_facts import TraitStatement, _normalised, _valid, colour_in

    assert colour_in("一双琥珀色的眼睛") == "金"
    assert colour_in("碧色的眸子") == "绿"
    assert colour_in("眼睛由黑转红") is None  # two colours
    assert colour_in("一头乌发") == "黑" and colour_in("三千青丝") is None
    guessed = TraitStatement(mention="林远", raw_text="精致的眸子", attribute="eye_color",
                             value="黑")  # fmt: skip
    assert _normalised("traits", guessed) is None  # no colour in the quote
    both = guessed.model_copy(update={"raw_text": "银发金瞳", "value": "金"})
    assert _normalised("traits", both).value == "金"
    assert _normalised("traits", both.model_copy(update={"value": "黑"})) is None
    hair = TraitStatement(mention="林远", raw_text="素白色的长袍", attribute="hair_color",
                          value="白")  # fmt: skip
    assert not _valid("traits", hair)
    eyes = hair.model_copy(update={"raw_text": "一双琥珀色的眼睛", "attribute": "eye_color",
                                   "value": "黄"})  # fmt: skip
    read = _normalised("traits", eyes)
    assert read.value == "金" and _valid("traits", read)


def test_a_corpse_is_nobody_appearing_in_person():
    from webfic.extraction.character_facts import PresenceStatement, _valid

    assert not _valid("presence", PresenceStatement(mention="邢烈", raw_text="邢烈的尸首被抬回"))
    assert _valid("presence", PresenceStatement(mention="邢烈", raw_text="邢烈推门进来"))


def test_changing_the_code_guards_changes_the_version(monkeypatch):
    # Stored readings hold what the guards kept: new guards must not reuse them.
    from webfic.extraction import character_facts as cf

    before = cf.version("prompt", 8000, 500)
    monkeypatch.setattr(cf, "GUARDS_VERSION", cf.GUARDS_VERSION + 1)
    assert cf.version("prompt", 8000, 500) != before


def test_a_colour_belongs_to_the_nearest_body_part():
    # Step 5-1e: "银发男人的瞳孔" was read as silver eyes.
    from webfic.extraction.character_facts import describes

    assert not describes("银发男人的瞳孔", "eye_color") and describes(
        "银发男人的瞳孔", "hair_color"
    )
    assert describes("黑发黑眼的脸", "eye_color") and describes("黑发黑眼的脸", "hair_color")
    assert not describes("漆黑的脸，阴冷的眸子", "eye_color")  # the face is black
    assert describes("眸色碧绿", "eye_color") and describes("头发金黄", "hair_color")
