"""The categories of facts stored in the `facts` table, and how each attribute is allowed
to change over story time. Checkers use the change rule to decide which differences are
candidate contradictions; step 3a registered ages, step 5-1 appearance, blood relations
and life and death.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict

from webfic.extraction.schemas import StatementType
from webfic.facts.describe import (
    DescribedFact,
    describe_age,
    describe_kinship,
    describe_life,
    describe_trait,
)
from webfic.facts.kinship import RELATIONS


class ChangeRule(StrEnum):
    FIXED = "fixed"  # never changes (eye colour, birth year): any difference is a candidate
    TEMPORAL = "temporal"  # follows story time (age): compare after projecting
    MONOTONIC = "monotonic"  # only grows (cultivation realm): going back is a candidate
    MUTABLE = "mutable"  # may change (identity, whereabouts): needs a reason in the text


class NoQualifiers(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class CategorySpec:
    name: str
    attributes: dict[str, ChangeRule]
    # What extraction made of a statement, in words (for the verify agent).
    describe: Callable[[DescribedFact], str]
    qualifiers: type[BaseModel] = NoQualifiers


AGE = CategorySpec(
    name="age",
    attributes={statement_type: ChangeRule.TEMPORAL for statement_type in get_args(StatementType)},
    describe=describe_age,
)


class AppearanceQualifiers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disguised: bool = False  # disguise, transformation, illusion: not their own looks
    temporary: bool = False  # a passing state (a technique, rage, a battle form); 5-1d


def own_look(qualifiers: dict[str, Any] | None) -> bool:
    """Whether an appearance fact is the character's own, lasting look (not disguised,
    not a passing state): only those are compared."""
    q = qualifiers or {}
    return not (q.get("disguised") or q.get("temporary"))


class KinshipQualifiers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    address: bool = False  # said to the person's face: only generations compared (5-1d)
    # The character is the <attribute> of this one. Kept by name, not id: merges and
    # renames move names in the alias table, and the checker looks the name up there.
    other_name: str
    other_mention: str


# Step 5-1. Differences in values are candidates; whether the text explains them (dye,
# disguise, revival) is for the verify agent to find.
APPEARANCE = CategorySpec(
    name="appearance",
    attributes={a: ChangeRule.FIXED for a in ("eye_color", "hair_color", "mark")},
    describe=describe_trait,
    qualifiers=AppearanceQualifiers,
)
KINSHIP = CategorySpec(
    name="kinship",
    attributes={relation: ChangeRule.FIXED for relation in RELATIONS},
    describe=describe_kinship,
    qualifiers=KinshipQualifiers,
)
LIFE = CategorySpec(
    name="life",
    # Alive, then dead, never back: appearing in person after dying is a candidate.
    attributes={"died": ChangeRule.MONOTONIC, "present": ChangeRule.MUTABLE},
    describe=describe_life,
)

CATEGORIES: dict[str, CategorySpec] = {spec.name: spec for spec in (AGE, APPEARANCE, KINSHIP, LIFE)}


class UnknownFactKind(ValueError):
    pass


def change_rule(category: str, attribute: str) -> ChangeRule:
    spec = CATEGORIES.get(category)
    if spec is None or attribute not in spec.attributes:
        raise UnknownFactKind(f"unregistered fact kind: {category}.{attribute}")
    return spec.attributes[attribute]


def validate_qualifiers(category: str, qualifiers: dict[str, Any]) -> dict[str, Any]:
    """Check a fact's category-specific extras against its category's model."""
    spec = CATEGORIES.get(category)
    if spec is None:
        raise UnknownFactKind(f"unregistered fact category: {category}")
    return spec.qualifiers.model_validate(qualifiers).model_dump(exclude_none=True)


def describe_fact(fact: DescribedFact) -> str:
    """A fact in words, by its category's own description."""
    spec = CATEGORIES.get(fact.category)
    if spec is None:
        raise UnknownFactKind(f"unregistered fact category: {fact.category}")
    return spec.describe(fact)
