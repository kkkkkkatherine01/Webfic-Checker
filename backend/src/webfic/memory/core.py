"""Core layer: each character's latest known state, updated chapter by chapter, with a
snapshot of the whole book after every chapter.

`advance` is a pure function (previous state + one chapter's facts -> new state), so
updating chapter by chapter gives the same result as recomputing from all facts, and
rolling back to chapter N-1 is just loading that chapter's snapshot.

Story time is counted in years from the start. A jump the text does not quantify
("多年以后") starts a new time segment (`epoch`): ages are only projected forward within
the segment they were stated in, never across such a jump.
"""

import uuid
from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.db.models import (
    Chapter,
    Character,
    CharacterAlias,
    CharacterStateRow,
    CoreSnapshot,
    ElapsedTimeFactRow,
    FactRow,
)
from webfic.extraction.schemas import LifeStage
from webfic.facts.registry import AGE
from webfic.memory.recall import find_character
from webfic.services.errors import NotFound

# --- state -----------------------------------------------------------------------------


class CharacterState(BaseModel):
    character_id: uuid.UUID
    canonical_name: str
    aliases: list[str] = []
    # Latest present-time, non-speculative stated age (a range for approximate ages).
    age_low: float | None = None
    age_high: float | None = None
    age_chapter: int | None = None
    age_quote: str | None = None
    age_epoch: int | None = None
    age_story_time: float | None = None  # story time within `age_epoch`
    life_stage: LifeStage | None = None
    life_stage_chapter: int | None = None


class BookState(BaseModel):
    chapter_number: int = 0  # the state is as of the end of this chapter
    epoch: int = 0
    story_time: float = 0.0  # years since the start of the current epoch
    characters: dict[uuid.UUID, CharacterState] = {}


class CoreFact(Protocol):
    character_id: uuid.UUID
    category: str
    attribute: str
    value_num: float | None
    value_max: float | None
    value_text: str | None
    is_flashback: bool
    is_speculative: bool
    raw_text: str
    char_start: int


class CoreSpan(Protocol):
    estimated_years: float | None
    kind: str
    is_flashback: bool
    char_start: int


class CoreCharacter(Protocol):
    id: uuid.UUID
    canonical_name: str
    aliases: list[str]


def _clock(previous: BookState, spans: Sequence[CoreSpan]):
    """(epoch, story time) just before a given offset of the chapter."""
    advances = sorted(
        (s for s in spans if s.kind == "advance" and not s.is_flashback),
        key=lambda s: s.char_start,
    )

    def at(offset: float) -> tuple[int, float]:
        epoch, time = previous.epoch, previous.story_time
        for span in advances:
            if span.char_start >= offset:
                break
            if span.estimated_years is None:
                epoch, time = epoch + 1, 0.0
            else:
                time += span.estimated_years
        return epoch, time

    return at


def advance(
    previous: BookState,
    *,
    chapter_number: int,
    facts: Sequence[CoreFact],
    spans: Sequence[CoreSpan],
    characters: Iterable[CoreCharacter],
    merges: Sequence[tuple[uuid.UUID, uuid.UUID]] = (),
) -> BookState:
    """The state at the end of chapter `chapter_number`. `characters` is the book's
    character list after the chapter (names and aliases may have changed); `merges` are
    (from, into) pairs made in the chapter."""
    states = {cid: s.model_copy(deep=True) for cid, s in previous.characters.items()}
    for from_id, into_id in merges:
        source = states.pop(from_id, None)
        target = states.get(into_id)
        if source is not None and target is not None and target.age_low is None:
            for field in ("age_low", "age_high", "age_chapter", "age_quote", "age_epoch",
                          "age_story_time", "life_stage", "life_stage_chapter"):  # fmt: skip
                setattr(target, field, getattr(source, field))

    current: dict[uuid.UUID, CharacterState] = {}
    for c in characters:
        state = states.get(c.id) or CharacterState(character_id=c.id, canonical_name="")
        state.canonical_name = c.canonical_name
        state.aliases = sorted(set(c.aliases) - {c.canonical_name})
        current[c.id] = state

    clock = _clock(previous, spans)
    for fact in sorted(facts, key=lambda f: f.char_start):
        state = current.get(fact.character_id)
        if state is None or fact.category != AGE.name or fact.is_flashback or fact.is_speculative:
            continue
        if fact.attribute == "absolute_age" and fact.value_num is not None:
            state.age_low = fact.value_num
            state.age_high = max(fact.value_num, fact.value_max or fact.value_num)
            state.age_chapter, state.age_quote = chapter_number, fact.raw_text
            state.age_epoch, state.age_story_time = clock(fact.char_start)
        elif fact.attribute == "life_stage" and fact.value_text:
            state.life_stage = LifeStage(fact.value_text)
            state.life_stage_chapter = chapter_number

    epoch, story_time = clock(float("inf"))
    return BookState(
        chapter_number=chapter_number, epoch=epoch, story_time=story_time, characters=current
    )


def projected_age(state: CharacterState, book: BookState) -> tuple[float, float] | None:
    """The character's age at the book state's point in story time, if it can be worked
    out without guessing: the stated age must lie in the current time segment."""
    if (
        state.age_low is None
        or state.age_high is None
        or state.age_story_time is None
        or state.age_epoch != book.epoch
    ):
        return None
    passed = book.story_time - state.age_story_time
    return state.age_low + passed, state.age_high + passed


# --- persistence -----------------------------------------------------------------------


async def load_state(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    before_chapter: int | None = None,
    as_of_chapter: int | None = None,
) -> BookState:
    """The latest snapshot before chapter `before_chapter` (or at / before
    `as_of_chapter`, or the latest of all); an empty state if there is none."""
    query = select(CoreSnapshot.state).where(
        CoreSnapshot.user_id == user_id, CoreSnapshot.book_id == book_id
    )
    if before_chapter is not None:
        query = query.where(CoreSnapshot.chapter_number < before_chapter)
    if as_of_chapter is not None:
        query = query.where(CoreSnapshot.chapter_number <= as_of_chapter)
    state = await session.scalar(query.order_by(CoreSnapshot.chapter_number.desc()).limit(1))
    return BookState.model_validate(state) if state is not None else BookState()


async def _write_current(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, state: BookState
) -> None:
    await session.execute(
        delete(CharacterStateRow).where(
            CharacterStateRow.user_id == user_id, CharacterStateRow.book_id == book_id
        )
    )
    existing = set(
        await session.scalars(
            select(Character.id).where(Character.user_id == user_id, Character.book_id == book_id)
        )
    )
    for character_id, character_state in state.characters.items():
        if character_id in existing:
            session.add(
                CharacterStateRow(
                    user_id=user_id, book_id=book_id, character_id=character_id,
                    chapter_number=state.chapter_number,
                    state=character_state.model_dump(mode="json"),
                )
            )  # fmt: skip


async def save_state(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    chapter_id: uuid.UUID,
    state: BookState,
) -> None:
    """Store the state after one chapter: current states and that chapter's snapshot."""
    await _write_current(session, user_id, book_id, state)
    await session.execute(delete(CoreSnapshot).where(CoreSnapshot.chapter_id == chapter_id))
    session.add(
        CoreSnapshot(
            user_id=user_id, book_id=book_id, chapter_id=chapter_id,
            chapter_number=state.chapter_number, state=state.model_dump(mode="json"),
        )
    )  # fmt: skip


async def restore(
    session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID, chapter_number: int
) -> BookState:
    """Roll the Core back to the end of chapter `chapter_number - 1`: later snapshots are
    dropped and the current states come from that snapshot (empty for chapter 1)."""
    await session.execute(
        delete(CoreSnapshot).where(
            CoreSnapshot.user_id == user_id,
            CoreSnapshot.book_id == book_id,
            CoreSnapshot.chapter_number >= chapter_number,
        )
    )
    state = await load_state(
        session, user_id=user_id, book_id=book_id, before_chapter=chapter_number
    )
    await _write_current(session, user_id, book_id, state)
    await session.flush()
    return state


async def rebuild(session: AsyncSession, *, user_id: uuid.UUID, book_id: uuid.UUID) -> BookState:
    """Recompute the Core from all stored facts, chapter by chapter, and store it. For
    books extracted before the Core existed; also the reference that incremental updates
    must agree with. Uses today's names (renames are not replayed)."""
    chapters = (
        await session.execute(
            select(Chapter.id, Chapter.number)
            .where(
                Chapter.user_id == user_id,
                Chapter.book_id == book_id,
                Chapter.status == "extracted",
            )
            .order_by(Chapter.number)
        )
    ).all()
    characters = await _characters(session, user_id, book_id)
    state = BookState()
    for chapter_id, number in chapters:
        facts = (
            await session.scalars(select(FactRow).where(FactRow.chapter_id == chapter_id))
        ).all()
        spans = (
            await session.scalars(
                select(ElapsedTimeFactRow).where(ElapsedTimeFactRow.chapter_id == chapter_id)
            )
        ).all()
        state = advance(
            state, chapter_number=number, facts=facts, spans=spans, characters=characters
        )
        await save_state(
            session, user_id=user_id, book_id=book_id, chapter_id=chapter_id, state=state
        )
    await session.flush()
    return state


class _Named(BaseModel):
    id: uuid.UUID
    canonical_name: str
    aliases: list[str]


async def _characters(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> list[_Named]:
    rows = (
        await session.scalars(
            select(Character).where(Character.user_id == user_id, Character.book_id == book_id)
        )
    ).all()
    aliases: dict[uuid.UUID, list[str]] = {}
    for a in await session.scalars(
        select(CharacterAlias).where(
            CharacterAlias.user_id == user_id, CharacterAlias.book_id == book_id
        )
    ):
        aliases.setdefault(a.character_id, []).append(a.alias)
    return [
        _Named(id=c.id, canonical_name=c.canonical_name, aliases=aliases.get(c.id, []))
        for c in rows
    ]


# --- query (an agent tool from step 4) --------------------------------------------------


class CharacterView(BaseModel):
    character_id: uuid.UUID
    canonical_name: str
    aliases: list[str]
    as_of_chapter: int  # the state is as of the end of this chapter
    age_low: float | None  # last stated present-time age, with where it was stated
    age_high: float | None
    age_chapter: int | None
    age_quote: str | None
    # The stated age carried forward to `as_of_chapter` by the story time in between;
    # None when a jump the text does not quantify lies in between.
    estimated_age_low: float | None
    estimated_age_high: float | None
    life_stage: LifeStage | None
    life_stage_chapter: int | None


async def get_character(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    name: str,
    as_of_chapter: int | None = None,
) -> CharacterView:
    """What is known about a character (by name or alias), as of the latest chapter or
    the end of `as_of_chapter`."""
    character = await find_character(session, user_id=user_id, book_id=book_id, name=name)
    book = await load_state(session, user_id=user_id, book_id=book_id, as_of_chapter=as_of_chapter)
    if book.chapter_number == 0 and as_of_chapter is None:
        raise NotFound("这部作品还没有角色状态（在 Core 之前导入的作品需要重新导入或重建）")
    state = book.characters.get(character.id) or CharacterState(
        character_id=character.id, canonical_name=character.canonical_name
    )
    aliases = sorted(
        await session.scalars(
            select(CharacterAlias.alias).where(
                CharacterAlias.user_id == user_id, CharacterAlias.character_id == character.id
            )
        )
    )
    estimate = projected_age(state, book)
    return CharacterView(
        character_id=character.id,
        canonical_name=character.canonical_name,
        aliases=aliases,
        as_of_chapter=book.chapter_number,
        age_low=state.age_low,
        age_high=state.age_high,
        age_chapter=state.age_chapter,
        age_quote=state.age_quote,
        estimated_age_low=estimate[0] if estimate else None,
        estimated_age_high=estimate[1] if estimate else None,
        life_stage=state.life_stage,
        life_stage_chapter=state.life_stage_chapter,
    )
