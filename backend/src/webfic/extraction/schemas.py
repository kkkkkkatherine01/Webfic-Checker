"""Shapes the LLM must return for extraction. Kept lenient: invalid combinations are
filtered after parsing instead of failing validation and triggering a paid retry."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

AGE_SCHEMA_VERSION = "age_v3"


class LifeStage(StrEnum):
    INFANT = "infant"
    CHILD = "child"
    TEEN = "teen"
    YOUNG_ADULT = "young_adult"
    MIDDLE_AGED = "middle_aged"
    ELDERLY = "elderly"


# Deliberately wide, overlapping ranges: missing a contradiction is cheaper for the
# author than a false alarm. Upper bound None means open-ended.
LIFE_STAGE_AGE_RANGE: dict[LifeStage, tuple[float, float | None]] = {
    LifeStage.INFANT: (0, 3),
    LifeStage.CHILD: (2, 13),
    LifeStage.TEEN: (10, 20),
    LifeStage.YOUNG_ADULT: (16, 35),
    LifeStage.MIDDLE_AGED: (30, 60),
    LifeStage.ELDERLY: (50, None),
}

StatementType = Literal["absolute_age", "relative_age", "birth_year", "life_stage"]


class AgeStatement(BaseModel):
    mention: str = Field(description="原文中对该角色的称呼")
    resolved_name: str | None = Field(
        default=None, description="若是已知角色，填其主名；判断为新角色则为 null"
    )
    raw_text: str = Field(description="逐字摘自原文的连续片段")
    statement_type: StatementType
    value: float | None = None
    # Approximate ages ("三十来岁") are ranges: value is the low end, value_max the high end.
    value_max: float | None = None
    life_stage: LifeStage | None = None
    is_flashback: bool = False
    years_before_present: float | None = None
    # The text that states how long ago it was ("十五年前"); without it the offset is
    # the model's own guess and is discarded.
    years_before_present_quote: str | None = None
    # A character's guess, a hypothetical or hearsay ("大概三十出头吧", "就算他四十岁").
    speculative: bool = False


ElapsedKind = Literal["advance", "short", "retrospective"]

# An "advance" shorter than this is really a "short" one.
SHORT_SPAN_YEARS = 1 / 12


class ElapsedTimeStatement(BaseModel):
    raw_text: str
    estimated_years: float | None = None
    # "advance": the story's present jumps forward by months or years ("三年后").
    # "short": it moves forward by days or weeks ("第二天", "三天之后"); too small to
    #   matter for ages, and often unquantified, so checkers ignore it.
    # "retrospective": a summary of time already passed ("这两年她奔波在外"); it does not
    #   move the present.
    # Models label more reliably than they omit, so all three are extracted.
    kind: ElapsedKind = "advance"
    is_flashback: bool = False


class RevealedName(BaseModel):
    """Someone known so far by an epithet ("疤脸刀客") turns out to be `real_name`."""

    known_as: str
    real_name: str


class AgeExtraction(BaseModel):
    age_statements: list[AgeStatement] = Field(default_factory=list)
    elapsed_time_statements: list[ElapsedTimeStatement] = Field(default_factory=list)
    revealed_names: list[RevealedName] = Field(default_factory=list)
