import uuid

from webfic.extraction.resolver import CharacterIndex, KnownCharacter, is_generic_mention

LIN = uuid.uuid4()


def index():
    return CharacterIndex([KnownCharacter(LIN, "林远", ["远儿"])])


def test_resolved_name_matches_canonical_and_mention_becomes_alias():
    idx = index()
    assert idx.resolve("林少爷", "林远") == LIN
    assert [a.alias for a in idx.new_aliases] == ["林少爷"]
    assert idx.resolve("林少爷", None) == LIN  # alias now known


def test_known_alias_resolves_without_model_help():
    idx = index()
    assert idx.resolve("远儿", None) == LIN
    assert idx.new_aliases == []


def test_generic_mention_resolved_by_model_is_not_stored_as_alias():
    idx = index()
    assert idx.resolve("那少年", "林远") == LIN
    assert idx.new_aliases == []


def test_unknown_name_creates_character():
    idx = index()
    new_id = idx.resolve("苏晚晴", None)
    assert new_id not in (None, LIN)
    assert [c.canonical_name for c in idx.new_characters] == ["苏晚晴"]
    assert idx.resolve("苏晚晴", None) == new_id  # no duplicate


def test_pronoun_with_unknown_resolved_name_creates_character_from_that_name():
    idx = index()
    new_id = idx.resolve("他", "赵无极")
    assert idx.name_of(new_id) == "赵无极"
    assert idx.new_aliases == []


def test_new_character_is_named_by_bare_name_and_titled_mention_becomes_alias():
    idx = index()
    new_id = idx.resolve("赵无极老先生", "赵无极")
    assert idx.name_of(new_id) == "赵无极"
    assert [a.alias for a in idx.new_aliases] == ["赵无极老先生"]
    assert idx.resolve("赵无极", None) == new_id


def test_descriptive_mention_resolves_but_is_not_an_alias():
    idx = index()
    zhao = idx.resolve("赵无极", "赵无极")
    assert idx.resolve("白发老者", "赵无极") == zhao
    assert idx.new_aliases == []
    assert idx.resolve("白发老者", None) is None


def test_unattributable_generic_mention_is_dropped():
    idx = index()
    assert idx.resolve("那少年", None) is None
    assert idx.new_characters == []


def test_reveal_renames_epithet_to_real_name():
    idx = index()
    blade = idx.resolve("疤脸刀客", None)  # created this chapter under the epithet
    idx.reveal("疤脸刀客", "沈砚")
    assert idx.name_of(blade) == "沈砚"
    assert idx.resolve("沈砚", None) == blade
    assert idx.resolve("疤脸刀客", None) == blade
    assert "- 沈砚：疤脸刀客" in idx.for_prompt()


def test_reveal_merges_a_character_wrongly_created_under_the_real_name():
    blade, shen = uuid.uuid4(), uuid.uuid4()
    idx = CharacterIndex(
        [KnownCharacter(blade, "疤脸刀客", ["刀客"]), KnownCharacter(shen, "沈砚", ["沈大哥"])]
    )
    idx.reveal("疤脸刀客", "沈砚")
    assert [(m.from_id, m.into_id) for m in idx.merges] == [(shen, blade)]
    assert [(r.character_id, r.new_name) for r in idx.renames] == [(blade, "沈砚")]
    for name in ("沈砚", "沈大哥", "疤脸刀客", "刀客"):
        assert idx.resolve(name, None) == blade
    # 沈砚 is canonical now; only the epithet that stopped being canonical needs a row.
    assert [a.alias for a in idx.new_aliases] == ["疤脸刀客"]


def test_reveal_ignores_unknown_epithet_and_generic_names():
    idx = index()
    idx.reveal("神秘人", "沈砚")
    idx.reveal("林远", "那少年")
    assert idx.renames == [] and idx.merges == [] and idx.name_of(LIN) == "林远"


def test_prompt_listing_is_sorted_and_stable():
    a = CharacterIndex(
        [KnownCharacter(uuid.uuid4(), "苏晚晴", ["苏师姐", "晚晴"]), KnownCharacter(LIN, "林远")]
    )
    listing = a.for_prompt()
    assert listing.splitlines()[0].startswith("- 林远")
    assert "晚晴、苏师姐" in listing
    assert CharacterIndex([]).for_prompt() == "（暂无）"


def test_generic_mentions():
    assert is_generic_mention("他")
    assert is_generic_mention("那个老者")
    assert is_generic_mention("  ")
    assert not is_generic_mention("林远")


def test_generic_mentions_from_real_text():
    for mention in ["您", "他们", "她们", "大家", "老人家", "老家伙", "他俩",
                    "两个十二三岁的小孩", "五火和带土", "师父与师娘",
                    "十二岁的旗木卡卡西", "我六岁的表弟", "十四岁宇智波",
                    "一个特别特别长的奇怪外号"]:  # fmt: skip
        assert is_generic_mention(mention), mention
    for name in ["旗木卡卡西", "带土", "五火", "Ilyana Osipova", "Maxim", "陈师傅", "许总",
                 "和珅", "林同光", "司徒和", "何及", "欧阳和平"]:  # fmt: skip
        assert not is_generic_mention(name), name
    assert is_generic_mention("Ilya Igorevich Osipov Junior")  # more than 3 words


def test_generic_mention_resolved_to_known_character_is_not_stored():
    idx = index()
    assert idx.resolve("五火和带土", "林远") == LIN  # the model decides who...
    assert idx.resolve("十二岁的林远", "林远") == LIN
    assert idx.new_aliases == [] and idx.new_characters == []  # ...but no alias is kept


def test_descriptive_phrase_does_not_create_a_character():
    idx = index()
    assert idx.resolve("我六岁的表弟", None) is None
    assert idx.new_characters == []
