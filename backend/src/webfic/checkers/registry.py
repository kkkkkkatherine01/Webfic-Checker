"""The checkers a book is checked with. Each one reads the facts it needs and returns
the issues it finds; `services.checks.run_checks` runs them and keeps the issue table in
step with what they found (step 5-0: before, the age checker was the only one and was
called directly).

Adding a kind of check means adding a checker here (see docs/steps/05-more-checkers.md
A7 for the whole procedure).
"""

import uuid
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.checkers.age import CHECKER_NAME as AGE_CHECKER
from webfic.checkers.age import AgeFact, ElapsedFact, check_ages
from webfic.checkers.character_facts import (
    CHECKER_NAME as CHARACTER_FACTS_CHECKER,
)
from webfic.checkers.character_facts import CharFact, check_character_facts
from webfic.checkers.types import ConsistencyIssue
from webfic.db.models import Character, CharacterAlias, ElapsedTimeFactRow, FactRow
from webfic.extraction.schemas import LifeStage
from webfic.facts.registry import AGE, APPEARANCE, KINSHIP, LIFE


class Checker(Protocol):
    name: str  # stored in consistency_issues.checker

    async def check(
        self, session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID
    ) -> list[ConsistencyIssue]: ...


async def load_age_facts(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> tuple[list[AgeFact], list[ElapsedFact]]:
    rows = (
        await session.execute(
            select(FactRow, Character.canonical_name)
            .join(
                Character,
                (Character.id == FactRow.character_id) & (Character.user_id == user_id),
            )
            .where(
                FactRow.user_id == user_id,
                FactRow.book_id == book_id,
                FactRow.category == AGE.name,
            )
            .order_by(FactRow.chapter_number, FactRow.char_start, FactRow.id)
        )
    ).all()
    ages = [
        AgeFact(
            id=r.id,
            character_id=r.character_id,
            character_name=name,
            raw_text=r.raw_text,
            statement_type=r.attribute,
            value=r.value_num,
            life_stage=LifeStage(r.value_text) if r.value_text else None,
            is_flashback=r.is_flashback,
            years_before_present=r.years_before_present,
            value_max=r.value_max,
            speculative=r.is_speculative,
            chapter_number=r.chapter_number,
            char_start=r.char_start,
            char_end=r.char_end,
            chapter_id=r.chapter_id,
            mention=r.mention,
        )
        for r, name in rows
    ]
    elapsed_rows = (
        await session.scalars(
            select(ElapsedTimeFactRow).where(
                ElapsedTimeFactRow.user_id == user_id, ElapsedTimeFactRow.book_id == book_id
            )
        )
    ).all()
    elapsed = [
        ElapsedFact(
            id=r.id,
            raw_text=r.raw_text,
            estimated_years=r.estimated_years,
            kind=r.kind,
            is_flashback=r.is_flashback,
            chapter_number=r.chapter_number,
            char_start=r.char_start,
            char_end=r.char_end,
            chapter_id=r.chapter_id,
        )
        for r in elapsed_rows
    ]
    return ages, elapsed


class AgeChecker:
    """Do a character's stated ages agree with the story time between them?"""

    name = AGE_CHECKER

    async def check(
        self, session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID
    ) -> list[ConsistencyIssue]:
        ages, elapsed = await load_age_facts(session, user_id, book_id)
        return check_ages(ages, elapsed)


async def load_character_facts(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> tuple[list[CharFact], dict[str, uuid.UUID]]:
    """The book's appearance, kinship and life facts, and every name and alias of its
    characters (the other side of a relation is kept by name)."""
    rows = (
        await session.execute(
            select(FactRow, Character.canonical_name)
            .join(
                Character,
                (Character.id == FactRow.character_id) & (Character.user_id == user_id),
            )
            .where(
                FactRow.user_id == user_id,
                FactRow.book_id == book_id,
                FactRow.category.in_([APPEARANCE.name, KINSHIP.name, LIFE.name]),
            )
            .order_by(FactRow.chapter_number, FactRow.char_start, FactRow.id)
        )
    ).all()
    facts = [
        CharFact(
            id=r.id, character_id=r.character_id, character_name=name, category=r.category,
            attribute=r.attribute, value_text=r.value_text, is_flashback=r.is_flashback,
            is_speculative=r.is_speculative, chapter_number=r.chapter_number,
            char_start=r.char_start, char_end=r.char_end, raw_text=r.raw_text,
            chapter_id=r.chapter_id, mention=r.mention, qualifiers=r.qualifiers or {},
        )
        for r, name in rows
    ]  # fmt: skip
    names: dict[str, uuid.UUID] = {}
    for character_id, alias in await session.execute(
        select(CharacterAlias.character_id, CharacterAlias.alias).where(
            CharacterAlias.user_id == user_id, CharacterAlias.book_id == book_id
        )
    ):
        names[alias] = character_id
    for character_id, name in await session.execute(
        select(Character.id, Character.canonical_name).where(
            Character.user_id == user_id, Character.book_id == book_id
        )
    ):
        names[name] = character_id  # canonical names win over aliases
    return facts, names


class CharacterFactChecker:
    """Appearance, blood relations, and appearing in person after dying (step 5-1)."""

    name = CHARACTER_FACTS_CHECKER

    async def check(
        self, session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID
    ) -> list[ConsistencyIssue]:
        facts, names = await load_character_facts(session, user_id, book_id)
        return check_character_facts(facts, names)


def registered() -> list[Checker]:
    """Every checker, in the order they run."""
    return [AgeChecker(), CharacterFactChecker()]
