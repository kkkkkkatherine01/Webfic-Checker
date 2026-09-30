"""Map how the text refers to someone (mention) to a character of the book."""

import re
import uuid
from dataclasses import dataclass, field

from webfic.facts.kinship import term_relation

# Mentions that can refer to different people in different places. They may be resolved
# by the model in context, but must never be stored as a character's alias or used to
# create a character: a wrong alias silently misattributes every later chapter.
_GENERIC_MENTIONS = {
    "他", "她", "它", "我", "你", "您", "自己", "本人", "此人", "那人", "这人", "对方",
    "人家", "在下", "本座", "老夫", "大家", "众人",
    "少年", "少女", "青年", "老者", "老人", "男子", "女子", "孩子", "婴儿", "年轻人",
    "师兄", "师姐", "师弟", "师妹", "师父", "师傅", "父亲", "母亲", "大哥", "小弟",
    # Roles and kin: whoever fits them in the scene at hand (step 3.5 found "老头子"
    # merging two old men into one character, and "老伴", "凶犯" made characters).
    "老头", "老头子", "老头儿", "老汉", "老妪", "老太", "老太太", "老太婆", "老伴", "老婆",
    "老公", "丈夫", "妻子", "儿子", "女儿", "爷爷", "奶奶", "外公", "外婆", "姥姥", "孙子",
    "孙女", "表哥", "表弟", "表姐", "表妹", "堂哥", "堂弟", "堂姐", "堂妹", "女孩", "男孩",
    "小孩", "凶手", "凶犯", "犯人", "嫌犯",
    # Step 5-1e: descriptions and roles found bound as aliases or made characters in the
    # evaluation base texts ("朋友" as 梅西, "管家" as 阮伯通, "皇帝" for two emperors).
    "朋友", "队友", "管家", "店家", "来人", "末将", "朕", "奴婢", "小徒", "女尼", "小童",
    "刀客", "举子", "道人", "和尚", "小和尚", "老先生", "老爷子", "老太爷", "老奶奶",
    "女鬼", "皇上", "皇帝", "陛下", "太子", "殿下", "公主", "小公主", "皇太孙", "小皇帝",
    "城主", "百夫长", "主公", "老师",
}  # fmt: skip
_DEMONSTRATIVE = re.compile(r"^(那|这|那个|这个|此|该)")
# Descriptions such as "白发老者" or "青衣少女" can describe other people too.
_GENERIC_SUFFIXES = ("老者", "老人", "老人家", "老家伙", "小家伙", "少年", "少女", "男子",
                     "女子", "青年", "孩子", "小孩", "姑娘", "年轻人", "老头", "老头子",
                     "老汉", "老妪", "老太", "女孩", "男孩",
                     # step 5-1e
                     "中年人", "妇人", "男人", "女人", "妇女", "死者", "嫌疑人", "强者",
                     "大汉", "壮汉", "家伙", "小子", "小伙", "小伙子", "毛头小伙", "人影",
                     "老外", "帅哥", "美女", "女孩儿", "小女孩儿", "小女娃", "小屁孩",
                     "大学生", "工作人员", "工作员", "老太监", "小朋友")  # fmt: skip
# Groups ("他们", "两个小孩") and pairs ("五火和带土") are not one person.
_GROUP_MARKERS = ("们", "两个", "几个", "二人", "两人", "俩", "诸位", "各位", "夫妇", "夫妻",
                  "兄弟", "姐妹", "众人")  # fmt: skip
# Counted or picked out of a group (step 5-1e): "一个女生", "四个儿子", "两名死者",
# "三人", "其中一个", "另一个公子", "有个人", "为首一人". "头" is left out as a measure
# word: "九头蛟龙" and "三头犬" are names, as are "十方魔尊" and "三皇子".
_NUMERAL = "一二两三四五六七八九十百千几数多"
_COUNTED = re.compile(
    rf"^(有|其中|另|另外|其他|其余)?[{_NUMERAL}]+[个名位只条批群伙些员]"
    rf"|^[{_NUMERAL}]+人$|^(有个|有人|其中|另一|其他|其余|某|为首)"
)
# A pronoun and a word for a relative: "他父亲", "我娘", "你的舅舅" (step 5-1e).
_PRONOUN_KIN = re.compile(r"^(他|她|我|你|您|其|咱|俺|它)(的)?(.+)$")
_COLLOQUIAL_KIN = ("老爹", "老娘", "爸", "妈", "爹", "娘", "爸爸", "妈妈", "哥", "姐", "师父")
# Names contain these characters too (和珅, 林同光, 司徒和), so a mention is a pair only
# when both sides of the conjunction are at least two characters long.
_PAIR = re.compile(r"^.{2,}[和与跟及同].{2,}$")
# Descriptive phrases ("十二岁的旗木卡卡西", "我六岁的表弟") are not names.
_AGE_PHRASE = re.compile(r"[\d零〇一二两三四五六七八九十百]+岁")
_MAX_CJK_NAME = 8
_MAX_LATIN_WORDS = 3  # Latin names are often long in letters ("Ilyana Osipova")


def is_generic_mention(mention: str) -> bool:
    """True if the mention cannot serve as a name: fine for the model to resolve in
    context, but never stored as an alias or used to create a character."""
    mention = mention.strip()
    if not mention:
        return True
    if (
        mention in _GENERIC_MENTIONS
        or _DEMONSTRATIVE.match(mention)
        or mention.endswith(_GENERIC_SUFFIXES)
        or any(m in mention for m in _GROUP_MARKERS)
        or _PAIR.match(mention)
        or "的" in mention
        or _AGE_PHRASE.search(mention)
        or _COUNTED.search(mention)
        or _kin_word(mention)
    ):
        return True
    if re.search(r"[A-Za-z]", mention):
        return len(mention.split()) > _MAX_LATIN_WORDS
    return len(mention) > _MAX_CJK_NAME


def _kin_word(mention: str) -> bool:
    """A word for a relative, alone or after a pronoun ("表弟", "他父亲", "我娘")."""
    if term_relation(mention):
        return True
    m = _PRONOUN_KIN.match(mention)
    return bool(m) and (term_relation(m.group(3)) is not None or m.group(3) in _COLLOQUIAL_KIN)


# Characters too common in names and titles to show that two names are one person's.
_NOT_SHARED = set("小老大阿之儿子人王神圣殿主道长师兄弟姐妹公皇帝太后爷娘哥叔伯先生女男头上下"
                  "一二三四五六七八九十")  # fmt: skip


# What may follow a surname in a name that is not a given name: a title, a term of
# address, a nickname marker ("陆长老", "赵总旗", "刘哥", "老阮", "江丫头").
_TITLES = (
    "哥", "姐", "弟", "妹", "兄", "叔", "伯", "爷", "总", "老", "小", "阿", "嫂", "婶",
    "老大", "老弟", "老哥", "大哥", "大姐", "先生", "女士", "夫人", "太太", "小姐", "公子",
    "少爷", "少主", "老爷", "老太爷", "老太太", "大人", "长老", "掌门", "宗主", "堂主",
    "城主", "院长", "将军", "队长", "主任", "副主任", "总旗", "师兄", "师姐", "师弟",
    "师妹", "师叔", "师伯", "道长", "真人", "上人", "丫头", "同学", "老师", "捕头",
    "掌柜", "员外", "司空", "丞相", "太守", "校尉", "都督", "王爷", "公公", "姑娘",
)  # fmt: skip


_TITLE_ENDINGS = set("哥姐弟妹兄叔伯爷姑姨舅嫂婶帝王公侯君翁婆总")


def shares_name(alias: str, name: str) -> bool:
    """Whether two names are one person's by their form: one contains the other ("水门" /
    "波风水门"), they share a character of the given name ("天河真人" / "天河子", "景儿" /
    "罗景"), or the alias is the surname with a title ("陆长老" / "陆辰照"). Two full names
    that share only the surname are two people, often of one family ("许思安" / "许采文",
    step 5-1e)."""
    if alias in name or name in alias:
        return True
    shared = (set(alias) & set(name)) - _NOT_SHARED
    surname = name[:1]
    if shared - {surname}:
        return True
    if surname and surname in alias and "一" <= surname <= "鿿":  # a Chinese surname
        rest = alias.replace(surname, "", 1)
        return (
            rest in _TITLES
            or any(len(t) > 1 and rest.endswith(t) for t in _TITLES)
            or (len(rest) <= 2 and rest[-1:] in _TITLE_ENDINGS)  # "武老伯", "玉帝"
        )
    return False


# Aliases the author added are shown to every kind of extraction.
USER = "user"


@dataclass
class KnownCharacter:
    id: uuid.UUID
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    kind: str = "age"  # the kind of extraction that named the character (step 5-1)
    alias_kinds: dict[str, str] = field(default_factory=dict)  # alias -> kind (or USER)


@dataclass
class NewAlias:
    character_id: uuid.UUID
    alias: str
    kind: str = "age"


@dataclass
class Rename:
    character_id: uuid.UUID
    new_name: str
    old_name: str  # kept so the rename can be undone


@dataclass
class Promotion:
    """A character (alias None) or one of its aliases, named by a later kind of
    extraction, used by an earlier one: credited to the earlier kind from now on, as if
    it had named it (step 5-1c)."""

    character_id: uuid.UUID
    alias: str | None
    old_kind: str
    new_kind: str


@dataclass
class Merge:
    """`from_id` turned out to be the same person as `into_id`."""

    from_id: uuid.UUID
    into_id: uuid.UUID
    from_name: str  # kept so the merge can be undone
    from_kind: str = "age"


class CharacterIndex:
    """In-memory view of a book's characters. `reveal` and `resolve` may create, rename
    and merge characters; callers persist `new_characters`, `renames`, `merges` and
    `new_aliases` afterwards (in that order). Characters and aliases created are
    credited to `kind`, the kind of extraction whose statements are being resolved."""

    def __init__(self, characters: list[KnownCharacter], kinds: list[str] | None = None):
        # The kinds of extraction in the order they read a chapter: a kind is shown what
        # it and the kinds before it named (see `for_prompt`).
        self._rank = {k: n for n, k in enumerate(kinds or ["age"])}
        self._by_id = {c.id: c for c in characters}
        self._by_name: dict[str, uuid.UUID] = {}
        for c in characters:
            self._by_name[c.canonical_name] = c.id
            for alias in c.aliases:
                self._by_name.setdefault(alias, c.id)
        self.new_characters: list[KnownCharacter] = []
        self.new_aliases: list[NewAlias] = []
        self.renames: list[Rename] = []
        self.merges: list[Merge] = []
        self.promotions: list[Promotion] = []
        self.kind = "age"

    def characters(self) -> list[KnownCharacter]:
        return list(self._by_id.values())

    def name_of(self, character_id: uuid.UUID) -> str:
        return self._by_id[character_id].canonical_name

    def for_prompt(self, kinds: set[str] | None = None) -> str:
        """Stable, sorted listing so identical inputs hash identically. With `kinds`,
        only the characters and aliases those kinds of extraction named (and the
        author's aliases). A kind is shown what it and the kinds before it named, so
        adding a kind never changes what the earlier ones are asked (step 5-1)."""
        shown = [c for c in self._by_id.values() if kinds is None or c.kind in kinds]
        if not shown:
            return "（暂无）"
        lines = []
        for c in sorted(shown, key=lambda c: c.canonical_name):
            names = [
                a
                for a in c.aliases
                if kinds is None or c.alias_kinds.get(a, c.kind) in {*kinds, USER}
            ]
            aliases = "、".join(sorted(set(names) - {c.canonical_name}))
            lines.append(f"- {c.canonical_name}：{aliases}" if aliases else f"- {c.canonical_name}")
        return "\n".join(lines)

    def resolve(self, mention: str, resolved_name: str | None, quote: str = "") -> uuid.UUID | None:
        """Return the character id, or None if the mention cannot be attributed. `quote`
        is the statement's text: a name written beside the character's own there
        ("衡王朱允熞") is evidence enough to keep it as an alias."""
        mention = mention.strip()
        resolved_name = resolved_name.strip() if resolved_name else None
        target = self._by_name.get(resolved_name) if resolved_name else None

        if target is not None:
            self._used(target, resolved_name)
        if target is None:
            target = self._by_name.get(mention)
        if target is None:
            # New character: name it by the model's bare name ("赵无极"), not by a
            # mention carrying a title ("赵无极老先生"), which is kept as an alias below.
            if resolved_name and not is_generic_mention(resolved_name):
                target = self._create(resolved_name)
            elif not is_generic_mention(mention):
                return self._create(mention)
            else:
                return None

        if (
            mention
            and mention not in self._by_name
            and not is_generic_mention(mention)
            and self._evident_alias(target, mention, quote)
        ):
            self._add_alias(target, mention)
        elif self._by_name.get(mention) == target:
            self._used(target, mention)
        return target

    def _evident_alias(self, character_id: uuid.UUID, alias: str, quote: str) -> bool:
        """Whether a new name for a character can be kept as an alias (step 5-1e). The
        model's reading of one statement is used for that statement, but an alias is kept
        for every later chapter, so it needs evidence: it shares the character's name
        ("水门" / "波风水门", "陆长老" / "陆辰照"), or the quote writes both ("衡王朱允熞").
        Other names ("雪娘" read as "雄雕") are not kept; real names revealed later are
        kept by `reveal`, and the author can add aliases."""
        c = self._by_id[character_id]
        names = [c.canonical_name, *c.aliases]
        if any(shares_name(alias, name) for name in names):
            return True
        return bool(quote) and alias in quote and any(n in quote for n in names)

    def _later(self, kind: str) -> bool:
        """Whether `kind` comes after the kind being resolved (so it is not shown to it)."""
        if kind == USER:
            return False
        return self._rank.get(kind, len(self._rank)) > self._rank.get(self.kind, 0)

    def _used(self, character_id: uuid.UUID, name: str | None) -> None:
        """The kind being resolved used this character, by this name. Had the later kind
        not named it first, this kind would have: credit it with the character, and with
        the alias, so what it is shown later stays what it would have been."""
        c = self._by_id[character_id]
        if c in self.new_characters:
            return
        if self._later(c.kind):
            self.promotions.append(Promotion(character_id, None, c.kind, self.kind))
            c.kind = self.kind
        alias = name if name and name != c.canonical_name else None
        if alias in c.alias_kinds and self._later(c.alias_kinds[alias]):
            self.promotions.append(Promotion(character_id, alias, c.alias_kinds[alias], self.kind))
            c.alias_kinds[alias] = self.kind

    def reveal(self, known_as: str, real_name: str) -> None:
        """The character known as `known_as` (e.g. an epithet, "疤脸刀客") is really
        `real_name`. Make the real name the canonical one and keep the epithet as an
        alias; if `real_name` was wrongly created as a separate character, merge it in.
        Call before `resolve` for the same chapter."""
        known_as, real_name = known_as.strip(), real_name.strip()
        target = self._by_name.get(known_as)
        if target is None or not real_name or is_generic_mention(real_name):
            return
        self._used(target, known_as)

        other = self._by_name.get(real_name)
        if other is not None and other != target:
            self._merge(other, into=target)

        character = self._by_id[target]
        if character.canonical_name == real_name:
            return
        old_name = character.canonical_name
        character.canonical_name = real_name
        self._by_name[real_name] = target
        self.renames.append(Rename(target, real_name, old_name))
        # The real name is canonical now; a pending alias row for it would be redundant.
        self.new_aliases = [
            a for a in self.new_aliases if not (a.character_id == target and a.alias == real_name)
        ]
        if old_name not in character.aliases:
            self._add_alias(target, old_name)

    def _merge(self, from_id: uuid.UUID, *, into: uuid.UUID) -> None:
        source = self._by_id.pop(from_id)
        persisted = source not in self.new_characters
        if not persisted:
            self.new_characters.remove(source)
        target = self._by_id[into]
        for name in [source.canonical_name, *source.aliases]:
            self._by_name[name] = into
            if name not in target.aliases:
                target.aliases.append(name)
                target.alias_kinds[name] = source.alias_kinds.get(name, source.kind)
        # The source's alias rows move with the merge; only its canonical name needs a
        # new alias row.
        self.new_aliases = [
            NewAlias(into, a.alias, a.kind) if a.character_id == from_id else a
            for a in self.new_aliases
        ]
        self.new_aliases.append(NewAlias(into, source.canonical_name, source.kind))
        if persisted:
            self.merges.append(Merge(from_id, into, source.canonical_name, source.kind))

    def _create(self, name: str) -> uuid.UUID:
        character = KnownCharacter(id=uuid.uuid4(), canonical_name=name, kind=self.kind)
        self._by_id[character.id] = character
        self._by_name[name] = character.id
        self.new_characters.append(character)
        return character.id

    def _add_alias(self, character_id: uuid.UUID, alias: str) -> None:
        self._by_id[character_id].aliases.append(alias)
        self._by_id[character_id].alias_kinds[alias] = self.kind
        self._by_name[alias] = character_id
        self.new_aliases.append(NewAlias(character_id, alias, self.kind))
