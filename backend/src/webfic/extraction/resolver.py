"""Map how the text refers to someone (mention) to a character of the book."""

import re
import uuid
from dataclasses import dataclass, field

# Mentions that can refer to different people in different places. They may be resolved
# by the model in context, but must never be stored as a character's alias or used to
# create a character: a wrong alias silently misattributes every later chapter.
_GENERIC_MENTIONS = {
    "他", "她", "它", "我", "你", "您", "自己", "本人", "此人", "那人", "这人", "对方",
    "人家", "在下", "本座", "老夫", "大家", "众人",
    "少年", "少女", "青年", "老者", "老人", "男子", "女子", "孩子", "婴儿", "年轻人",
    "师兄", "师姐", "师弟", "师妹", "师父", "师傅", "父亲", "母亲", "大哥", "小弟",
}  # fmt: skip
_DEMONSTRATIVE = re.compile(r"^(那|这|那个|这个|此|该)")
# Descriptions such as "白发老者" or "青衣少女" can describe other people too.
_GENERIC_SUFFIXES = ("老者", "老人", "老人家", "老家伙", "小家伙", "少年", "少女", "男子",
                     "女子", "青年", "孩子", "小孩", "姑娘", "年轻人")  # fmt: skip
# Groups ("他们", "两个小孩") and pairs ("五火和带土") are not one person.
_GROUP_MARKERS = ("们", "两个", "几个", "二人", "两人", "俩", "诸位", "各位")
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
    ):
        return True
    if re.search(r"[A-Za-z]", mention):
        return len(mention.split()) > _MAX_LATIN_WORDS
    return len(mention) > _MAX_CJK_NAME


@dataclass
class KnownCharacter:
    id: uuid.UUID
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class NewAlias:
    character_id: uuid.UUID
    alias: str


@dataclass
class Rename:
    character_id: uuid.UUID
    new_name: str
    old_name: str  # kept so the rename can be undone


@dataclass
class Merge:
    """`from_id` turned out to be the same person as `into_id`."""

    from_id: uuid.UUID
    into_id: uuid.UUID
    from_name: str  # kept so the merge can be undone


class CharacterIndex:
    """In-memory view of a book's characters. `reveal` and `resolve` may create, rename
    and merge characters; callers persist `new_characters`, `renames`, `merges` and
    `new_aliases` afterwards (in that order)."""

    def __init__(self, characters: list[KnownCharacter]):
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

    def characters(self) -> list[KnownCharacter]:
        return list(self._by_id.values())

    def name_of(self, character_id: uuid.UUID) -> str:
        return self._by_id[character_id].canonical_name

    def for_prompt(self) -> str:
        """Stable, sorted listing so identical inputs hash identically."""
        if not self._by_id:
            return "（暂无）"
        lines = []
        for c in sorted(self._by_id.values(), key=lambda c: c.canonical_name):
            aliases = "、".join(sorted(set(c.aliases) - {c.canonical_name}))
            lines.append(f"- {c.canonical_name}：{aliases}" if aliases else f"- {c.canonical_name}")
        return "\n".join(lines)

    def resolve(self, mention: str, resolved_name: str | None) -> uuid.UUID | None:
        """Return the character id, or None if the mention cannot be attributed."""
        mention = mention.strip()
        resolved_name = resolved_name.strip() if resolved_name else None
        target = self._by_name.get(resolved_name) if resolved_name else None

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

        if mention and mention not in self._by_name and not is_generic_mention(mention):
            self._add_alias(target, mention)
        return target

    def reveal(self, known_as: str, real_name: str) -> None:
        """The character known as `known_as` (e.g. an epithet, "疤脸刀客") is really
        `real_name`. Make the real name the canonical one and keep the epithet as an
        alias; if `real_name` was wrongly created as a separate character, merge it in.
        Call before `resolve` for the same chapter."""
        known_as, real_name = known_as.strip(), real_name.strip()
        target = self._by_name.get(known_as)
        if target is None or not real_name or is_generic_mention(real_name):
            return

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
        for name in [source.canonical_name, *source.aliases]:
            self._by_name[name] = into
            if name not in self._by_id[into].aliases:
                self._by_id[into].aliases.append(name)
        # The source's alias rows move with the merge; only its canonical name needs a
        # new alias row.
        self.new_aliases = [
            NewAlias(into, a.alias) if a.character_id == from_id else a for a in self.new_aliases
        ]
        self.new_aliases.append(NewAlias(into, source.canonical_name))
        if persisted:
            self.merges.append(Merge(from_id, into, source.canonical_name))

    def _create(self, name: str) -> uuid.UUID:
        character = KnownCharacter(id=uuid.uuid4(), canonical_name=name)
        self._by_id[character.id] = character
        self._by_name[name] = character.id
        self.new_characters.append(character)
        return character.id

    def _add_alias(self, character_id: uuid.UUID, alias: str) -> None:
        self._by_id[character_id].aliases.append(alias)
        self._by_name[alias] = character_id
        self.new_aliases.append(NewAlias(character_id, alias))
