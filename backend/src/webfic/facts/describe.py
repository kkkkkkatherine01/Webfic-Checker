"""Facts in words, for the model (the verify agent reads what extraction made of a
statement) and for people. Each category registers its own `describe` in
`webfic.facts.registry`; time spans are described here too."""

from typing import Any, Protocol

from webfic.facts import kinship


class DescribedFact(Protocol):
    category: str
    attribute: str
    value_num: float | None
    value_max: float | None
    value_text: str | None
    is_flashback: bool
    years_before_present: float | None
    is_speculative: bool
    qualifiers: dict[str, Any]


_AGE_ATTRIBUTE = {
    "absolute_age": "年龄",
    "relative_age": "相对年龄",
    "birth_year": "出生年份",
    "life_stage": "人生阶段",
}
_SPAN_KIND = {
    "advance": "推进",
    "short": "短时推进（不计入年龄推算）",
    "retrospective": "回顾（不推进）",
    "future": "将来 / 计划（不推进）",
}


def describe_age(f: DescribedFact) -> str:
    if f.attribute == "life_stage":
        value = f.value_text or "?"
    elif f.value_num is None:
        value = "?"
    else:
        value = fmt_range(f.value_num, f.value_max) + (
            " 岁" if f.attribute == "absolute_age" else ""
        )
    return f"{_AGE_ATTRIBUTE.get(f.attribute, f.attribute)} {value}；{when_and_how_sure(f)}"


def when_and_how_sure(f: DescribedFact) -> str:
    """The flags every category shares: flashback (and how long ago), guess."""
    flags = []
    if f.is_flashback:
        ago = (
            f"，距今 {fmt_num(f.years_before_present)} 年"
            if f.years_before_present
            else "，距今年数未知"
        )
        flags.append("回忆" + ago)
    else:
        flags.append("现在时")
    if f.is_speculative:
        flags.append("推测")
    return "、".join(flags)


_TRAIT = {"eye_color": "瞳色", "hair_color": "发色", "mark": "身体标记"}


def describe_trait(f: DescribedFact) -> str:
    """Appearance (step 5-1): "瞳色 绿；现在时"."""
    flags = [when_and_how_sure(f)]
    if (f.qualifiers or {}).get("disguised"):
        flags.append("伪装 / 变身后的样子")
    if (f.qualifiers or {}).get("temporary"):
        flags.append("一时的状态")
    return f"{_TRAIT.get(f.attribute, f.attribute)} {f.value_text or '?'}；{'、'.join(flags)}"


def describe_kinship(f: DescribedFact) -> str:
    """A blood relation: "是「林震」的儿子"."""
    other = (f.qualifiers or {}).get("other_mention") or "?"
    sure = "；推测 / 传闻" if f.is_speculative else ""
    return f"亲属：是「{other}」的{kinship.label(f.attribute)}{sure}"


def describe_life(f: DescribedFact) -> str:
    """Death, or appearing in person."""
    if f.attribute == "died":
        return "死亡" + ("（传闻 / 不确定）" if f.is_speculative else "")
    return "在现在的场景里亲自出场"


def describe_span(kind: str, years: float | None, flashback: bool) -> str:
    amount = f"{fmt_num(years)} 年" if years is not None else "年数不明"
    return f"{_SPAN_KIND.get(kind, kind)}，{amount}" + ("；在回忆里" if flashback else "")


def fmt_num(x: float | None) -> str:
    if x is None:
        return "?"
    return str(int(x)) if float(x).is_integer() else f"{x:.3g}"


def fmt_range(low: float, high: float | None) -> str:
    if high is None or high == low:
        return fmt_num(low)
    return f"{fmt_num(low)}–{fmt_num(high)}"
