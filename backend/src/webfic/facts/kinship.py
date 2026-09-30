"""Blood relations between two characters, and when two statements about the same pair
cannot both be true (step 5-1).

A statement reads "A is the <relation> of B" ("林远是林震的儿子": A 林远, B 林震, son).
Each relation is described by what it says about the pair: how many generations A is
above B, the kind of link (parent and child, siblings, uncle and nephew, cousins), which
side of the family, A's and B's sex where the word tells it, and who is older among
siblings. Seen from B the same facts read the other way round, so a statement and its
inverse compare directly: "A is B's father" and later "B is A's elder brother" put A one
generation above B and then on the same one — a contradiction.

Marriage is left out: it changes (remarriage), so it is not a fixed fact.
"""

from dataclasses import dataclass, replace
from typing import Literal

Sex = Literal["M", "F"] | None
Link = Literal["parent", "grand", "sibling", "uncle", "cousin"]


@dataclass(frozen=True)
class Kin:
    """What "A is <relation> of B" says about the pair."""

    generations: int  # A's generation above B's (father: 1, son: -1, brother: 0)
    link: Link
    side: str | None  # paternal / maternal line, or 堂 / 表 for cousins; None: not told
    sex_a: Sex
    sex_b: Sex = None
    a_older: bool | None = None  # siblings and cousins: is A the elder

    def inverse(self) -> "Kin":
        """The same facts read from B: "B is <?> of A"."""
        return replace(
            self,
            generations=-self.generations,
            sex_a=self.sex_b,
            sex_b=self.sex_a,
            a_older=None if self.a_older is None else not self.a_older,
        )


# relation -> (Chinese label, what it says). A is the relation of B.
RELATIONS: dict[str, tuple[str, Kin]] = {
    "father": ("父亲", Kin(1, "parent", None, "M")),
    "mother": ("母亲", Kin(1, "parent", None, "F")),
    "son": ("儿子", Kin(-1, "parent", None, "M")),
    "daughter": ("女儿", Kin(-1, "parent", None, "F")),
    "elder_brother": ("哥哥", Kin(0, "sibling", None, "M", a_older=True)),
    "younger_brother": ("弟弟", Kin(0, "sibling", None, "M", a_older=False)),
    "elder_sister": ("姐姐", Kin(0, "sibling", None, "F", a_older=True)),
    "younger_sister": ("妹妹", Kin(0, "sibling", None, "F", a_older=False)),
    "brother": ("兄弟（长幼不明）", Kin(0, "sibling", None, "M")),
    "sister": ("姐妹（长幼不明）", Kin(0, "sibling", None, "F")),
    "paternal_grandfather": ("祖父 / 爷爷", Kin(2, "grand", "paternal", "M")),
    "paternal_grandmother": ("祖母 / 奶奶", Kin(2, "grand", "paternal", "F")),
    "maternal_grandfather": ("外祖父 / 外公 / 姥爷", Kin(2, "grand", "maternal", "M")),
    "maternal_grandmother": ("外祖母 / 外婆 / 姥姥", Kin(2, "grand", "maternal", "F")),
    "grandson": ("孙子", Kin(-2, "grand", "paternal", "M")),
    "granddaughter": ("孙女", Kin(-2, "grand", "paternal", "F")),
    "daughters_son": ("外孙", Kin(-2, "grand", "maternal", "M")),
    "daughters_daughter": ("外孙女", Kin(-2, "grand", "maternal", "F")),
    # Uncles and aunts on the father's side call their siblings' children 侄; on the
    # mother's side, 外甥.
    "paternal_uncle": ("伯父 / 叔叔", Kin(1, "uncle", "paternal", "M")),
    "paternal_aunt": ("姑姑", Kin(1, "uncle", "paternal", "F")),
    "maternal_uncle": ("舅舅", Kin(1, "uncle", "maternal", "M")),
    "maternal_aunt": ("姨妈 / 姨母", Kin(1, "uncle", "maternal", "F")),
    "nephew": ("侄子", Kin(-1, "uncle", "paternal", "M")),
    "niece": ("侄女", Kin(-1, "uncle", "paternal", "F")),
    "sisters_son": ("外甥", Kin(-1, "uncle", "maternal", "M")),
    "sisters_daughter": ("外甥女", Kin(-1, "uncle", "maternal", "F")),
    "tang_cousin": ("堂兄弟姐妹", Kin(0, "cousin", "堂", None)),
    "biao_cousin": ("表兄弟姐妹", Kin(0, "cousin", "表", None)),
}


def label(relation: str) -> str:
    return RELATIONS[relation][0] if relation in RELATIONS else relation


def kin(relation: str) -> Kin:
    return RELATIONS[relation][1]


def conflict(first: Kin, second: Kin) -> str | None:
    """Why two statements about the same ordered pair (A, B) cannot both hold, or None.
    Only what both statements tell is compared: "brother" and "elder brother" agree."""
    if first.generations != second.generations:
        return "辈分不同"
    if first.link != second.link:
        return "关系类型不同"
    if first.side and second.side and first.side != second.side:
        return "父系 / 母系不同" if first.link != "cousin" else "堂 / 表不同"
    for a, b, what in (
        (first.sex_a, second.sex_a, "性别不同"),
        (first.sex_b, second.sex_b, "性别不同"),
        (first.a_older, second.a_older, "长幼不同"),
    ):
        if a is not None and b is not None and a != b:
            return what
    return None


# --- words for relations (step 5-1d) --------------------------------------------------------
#
# Extraction copies the word the text uses ("爹", "外甥女", "五舅舅") and says who is
# called so and whose it is; the relation is looked up here instead of being chosen by
# the model, which is where the direction and the kind of relation went wrong.

TERMS: dict[str, str] = {
    **dict.fromkeys(
        ["父亲", "父", "爹", "爹爹", "阿爹", "爸", "爸爸", "老爸", "家父", "生父", "先父",
         "令尊", "父王", "父皇", "爹地"], "father"),
    **dict.fromkeys(
        ["母亲", "母", "娘", "娘亲", "阿娘", "妈", "妈妈", "老妈", "家母", "生母", "先母",
         "令堂", "母后", "母妃", "额娘"], "mother"),
    **dict.fromkeys(["儿子", "儿", "犬子", "长子", "次子", "幼子", "独子", "令郎", "世子"], "son"),
    **dict.fromkeys(["女儿", "闺女", "长女", "次女", "幼女", "独女", "小女", "令爱", "千金"],
                    "daughter"),
    **dict.fromkeys(["哥哥", "哥", "兄长", "兄", "胞兄", "家兄", "王兄", "皇兄"], "elder_brother"),
    **dict.fromkeys(["弟弟", "弟", "胞弟", "舍弟", "王弟", "皇弟"], "younger_brother"),
    **dict.fromkeys(["姐姐", "姐", "胞姐", "家姐", "阿姐"], "elder_sister"),
    **dict.fromkeys(["妹妹", "妹", "胞妹", "舍妹"], "younger_sister"),
    **dict.fromkeys(["兄弟"], "brother"),
    **dict.fromkeys(["姐妹"], "sister"),
    **dict.fromkeys(["爷爷", "爷", "祖父", "家祖", "阿爷", "皇爷爷"], "paternal_grandfather"),
    **dict.fromkeys(["奶奶", "祖母", "阿奶"], "paternal_grandmother"),
    **dict.fromkeys(["外公", "外祖父", "姥爷", "外祖"], "maternal_grandfather"),
    **dict.fromkeys(["外婆", "外祖母", "姥姥"], "maternal_grandmother"),
    **dict.fromkeys(["孙子", "孙儿", "孙"], "grandson"),
    **dict.fromkeys(["孙女"], "granddaughter"),
    **dict.fromkeys(["外孙"], "daughters_son"),
    **dict.fromkeys(["外孙女"], "daughters_daughter"),
    **dict.fromkeys(["伯父", "伯伯", "伯", "叔叔", "叔父", "叔"], "paternal_uncle"),
    **dict.fromkeys(["姑姑", "姑妈", "姑母", "姑"], "paternal_aunt"),
    **dict.fromkeys(["舅舅", "舅父", "舅"], "maternal_uncle"),
    **dict.fromkeys(["姨妈", "姨母", "姨娘", "姨"], "maternal_aunt"),
    **dict.fromkeys(["侄子", "侄儿", "侄"], "nephew"),
    **dict.fromkeys(["侄女"], "niece"),
    **dict.fromkeys(["外甥"], "sisters_son"),
    **dict.fromkeys(["外甥女"], "sisters_daughter"),
    **dict.fromkeys(["堂兄", "堂哥", "堂弟", "堂姐", "堂妹", "堂兄弟", "堂姐妹"], "tang_cousin"),
    **dict.fromkeys(["表兄", "表哥", "表弟", "表姐", "表妹", "表兄弟", "表姐妹"], "biao_cousin"),
}  # fmt: skip

# Words naming both people at once ("父子俩" says nothing of who is the father).
COLLECTIVE = (
    "父子", "父女", "母子", "母女", "兄弟俩", "姐弟", "兄妹", "姐妹俩", "夫妇", "夫妻", "爹娘",
    "父母", "爸妈", "双亲", "祖孙", "叔侄", "舅甥", "兄弟们", "姐妹们",
)  # fmt: skip
# Words that look like kin but are not in everyday use: forms of address and titles
# ("小姐" is not a younger elder sister, "老子" is often "I").
NOT_KIN = (
    "小姐", "老子", "老弟", "老兄", "兄台", "贤弟", "贤兄", "大姐头",
    "老爷", "少爷", "王爷", "姑娘", "大娘",
)  # fmt: skip

# Order, rank and emphasis in front of a word: "五舅舅", "三姐", "二伯父", "亲舅舅", "皇叔".
RANK = "大二三四五六七八九十小老亲胞皇族幺"
_BARE_ONLY = ("儿", "孙")
_SUFFIX = ("大人", "们")


_BROTHERS = ("elder_brother", "younger_brother", "brother")
_SISTERS = ("elder_sister", "younger_sister", "sister")
_CHILDREN = {"son": 0, "daughter": 1}
# A relative's relative -> the relation ("母亲的哥哥" is a maternal uncle; step 5-1d).
# Chains the table cannot name (a brother's father) are left unnamed.
COMPOSE: dict[tuple[str, str], str] = {
    **{("father", b): "paternal_uncle" for b in _BROTHERS},
    **{("father", s): "paternal_aunt" for s in _SISTERS},
    **{("mother", b): "maternal_uncle" for b in _BROTHERS},
    **{("mother", s): "maternal_aunt" for s in _SISTERS},
    ("father", "father"): "paternal_grandfather",
    ("father", "mother"): "paternal_grandmother",
    ("mother", "father"): "maternal_grandfather",
    ("mother", "mother"): "maternal_grandmother",
    **{(b, c): ("nephew", "niece")[i] for b in _BROTHERS for c, i in _CHILDREN.items()},
    **{(s, c): ("sisters_son", "sisters_daughter")[i] for s in _SISTERS
       for c, i in _CHILDREN.items()},
    ("son", "son"): "grandson",
    ("son", "daughter"): "granddaughter",
    ("daughter", "son"): "daughters_son",
    ("daughter", "daughter"): "daughters_daughter",
    **{("paternal_uncle", c): "tang_cousin" for c in _CHILDREN},
    **{(a, c): "biao_cousin" for a in ("paternal_aunt", "maternal_uncle", "maternal_aunt")
       for c in _CHILDREN},
}  # fmt: skip


def term_relation(term: str) -> str | None:
    """The relation a word for a relative names, or None (not a word for a blood
    relative, a word for two people, or not in the table). A chain ("母亲的哥哥") is
    composed step by step."""
    word = term.strip()
    if not word or any(n in word for n in (*NOT_KIN, *COLLECTIVE)):
        return None
    if "的" in word:
        steps = [_one_term(part) for part in word.split("的")]
        relation = steps[0]
        for step in steps[1:]:
            relation = COMPOSE.get((relation, step)) if relation and step else None
        return relation
    return _one_term(word)


def _one_term(word: str) -> str | None:
    for suffix in _SUFFIX:
        word = word.removesuffix(suffix)
    if word in TERMS:
        return TERMS[word]
    stripped = word.lstrip(RANK)
    # "九儿", "小三儿", "老孙", "小孙" are names and nicknames, not a son or a grandson
    # (step 5-1e); these two count only on their own.
    if stripped in _BARE_ONLY and stripped != word:
        return None
    return TERMS.get(stripped) if stripped else None
