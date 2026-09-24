"""Recall layer queries: the structured facts extracted chapter by chapter.

Written to become agent tools in step 4: Pydantic in and out, short self-describing
fields, and every query scoped to one user's book.
"""

import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Select, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.db.models import Book, Character, CharacterAlias, ElapsedTimeFactRow, FactRow
from webfic.services.errors import NotFound

ChapterRange = tuple[int, int]  # inclusive: (3, 5) = chapters 3, 4 and 5


class FactView(BaseModel):
    character_id: uuid.UUID
    character_name: str
    chapter_number: int
    category: str
    attribute: str
    value_num: float | None
    value_max: float | None
    value_text: str | None
    is_flashback: bool
    years_before_present: float | None
    is_speculative: bool
    qualifiers: dict[str, Any]
    mention: str
    raw_text: str
    char_start: int
    char_end: int


class TimeSpanView(BaseModel):
    chapter_number: int
    raw_text: str
    estimated_years: float | None
    kind: str  # advance / short / retrospective
    is_flashback: bool
    char_start: int
    char_end: int


async def _check_book(session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID) -> None:
    owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id)
    if await session.scalar(owned) is None:
        raise NotFound(f"book {book_id}")


async def find_character(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, name: str
) -> Character:
    """The character called `name`, by canonical name or alias."""
    name = name.strip()
    alias_of = select(CharacterAlias.character_id).where(
        CharacterAlias.user_id == user_id,
        CharacterAlias.book_id == book_id,
        CharacterAlias.alias == name,
    )
    character = await session.scalar(
        select(Character)
        .where(
            Character.user_id == user_id,
            Character.book_id == book_id,
            or_(Character.canonical_name == name, Character.id.in_(alias_of)),
        )
        .limit(1)
    )
    if character is None:
        raise NotFound(f"character {name}")
    return character


def _in_chapters(query: Select[Any], column: Any, chapters: ChapterRange | None) -> Select[Any]:
    if chapters is None:
        return query
    first, last = chapters
    return query.where(column >= first, column <= last)


async def list_facts(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    character: str | None = None,
    category: str | None = None,
    chapters: ChapterRange | None = None,
) -> list[FactView]:
    """Facts in narrative order, optionally for one character (name or alias), one
    category and a chapter range."""
    await _check_book(session, user_id, book_id)
    query = (
        select(FactRow, Character.canonical_name)
        .join(Character, Character.id == FactRow.character_id)
        .where(FactRow.user_id == user_id, FactRow.book_id == book_id)
        .order_by(FactRow.chapter_number, FactRow.char_start)
    )
    if character is not None:
        found = await find_character(session, user_id=user_id, book_id=book_id, name=character)
        query = query.where(FactRow.character_id == found.id)
    if category is not None:
        query = query.where(FactRow.category == category)
    query = _in_chapters(query, FactRow.chapter_number, chapters)
    return [
        FactView(
            character_id=r.character_id,
            character_name=name,
            chapter_number=r.chapter_number,
            category=r.category,
            attribute=r.attribute,
            value_num=r.value_num,
            value_max=r.value_max,
            value_text=r.value_text,
            is_flashback=r.is_flashback,
            years_before_present=r.years_before_present,
            is_speculative=r.is_speculative,
            qualifiers=r.qualifiers or {},
            mention=r.mention,
            raw_text=r.raw_text,
            char_start=r.char_start,
            char_end=r.char_end,
        )
        for r, name in await session.execute(query)
    ]


async def list_time_spans(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapters: ChapterRange | None = None,
) -> list[TimeSpanView]:
    """Statements of time passing ("三年后", "第二天", "这两年"), in narrative order."""
    await _check_book(session, user_id, book_id)
    query = (
        select(ElapsedTimeFactRow)
        .where(ElapsedTimeFactRow.user_id == user_id, ElapsedTimeFactRow.book_id == book_id)
        .order_by(ElapsedTimeFactRow.chapter_number, ElapsedTimeFactRow.char_start)
    )
    query = _in_chapters(query, ElapsedTimeFactRow.chapter_number, chapters)
    return [
        TimeSpanView(
            chapter_number=r.chapter_number,
            raw_text=r.raw_text,
            estimated_years=r.estimated_years,
            kind=r.kind,
            is_flashback=r.is_flashback,
            char_start=r.char_start,
            char_end=r.char_end,
        )
        for r in await session.scalars(query)
    ]
