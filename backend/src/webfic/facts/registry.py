"""The categories of facts stored in the `facts` table, and how each attribute is allowed
to change over story time. Checkers use the change rule to decide which differences are
candidate contradictions; step 3a only registers ages, step 5 adds appearance, titles...
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict

from webfic.extraction.schemas import StatementType


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
    qualifiers: type[BaseModel] = NoQualifiers


AGE = CategorySpec(
    name="age",
    attributes={statement_type: ChangeRule.TEMPORAL for statement_type in get_args(StatementType)},
)

CATEGORIES: dict[str, CategorySpec] = {spec.name: spec for spec in (AGE,)}


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
