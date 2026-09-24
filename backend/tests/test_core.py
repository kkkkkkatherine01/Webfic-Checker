"""Core layer: the pure state update, its storage, rollback and the character query."""

import uuid
from dataclasses import dataclass, field

import pytest
from sqlalchemy import delete, select

from tests.fakes import FakeBackend
from tests.test_pipeline import USER, import_book, respond
from webfic.db.models import CharacterStateRow, CoreSnapshot
from webfic.extraction.schemas import LifeStage
from webfic.memory import core
from webfic.memory.core import BookState, advance, projected_age
from webfic.services.errors import NotFound

LIN, SU = uuid.uuid4(), uuid.uuid4()


@dataclass
class Fact:
    character_id: uuid.UUID
    value_num: float | None
    char_start: int = 0
    attribute: str = "absolute_age"
    value_max: float | None = None
    value_text: str | None = None
    is_flashback: bool = False
    is_speculative: bool = False
    category: str = "age"
    raw_text: str = "……岁"


@dataclass
class Span:
    estimated_years: float | None
    char_start: int = 0
    kind: str = "advance"
    is_flashback: bool = False


@dataclass
class Person:
    id: uuid.UUID
    canonical_name: str
    aliases: list[str] = field(default_factory=list)


PEOPLE = [Person(LIN, "林远", ["林少爷"]), Person(SU, "苏晚晴")]


def run(chapters, people=PEOPLE):
    """chapters: list of (facts, spans); returns the state after each chapter."""
    state, states = BookState(), []
    for number, (facts, spans) in enumerate(chapters, start=1):
        state = advance(state, chapter_number=number, facts=facts, spans=spans, characters=people)
        states.append(state)
    return states


def age_now(state, who=LIN):
    return projected_age(state.characters[who], state)


# --- advance ---------------------------------------------------------------------------


def test_age_is_carried_forward_by_story_time():
    s1, s2 = run([([Fact(LIN, 18)], []), ([], [Span(3)])])
    assert age_now(s1) == (18, 18)
    assert (s2.story_time, age_now(s2)) == (3, (21, 21))
    assert s2.characters[LIN].age_chapter == 1  # the stated age and where it was stated


def test_time_before_a_statement_in_the_same_chapter_counts():
    (s1,) = run(
        [([Fact(LIN, 21, char_start=50)], [Span(3, char_start=10), Span(2, char_start=90)])]
    )
    assert s1.characters[LIN].age_story_time == 3
    assert age_now(s1) == (23, 23)


def test_unquantified_jump_starts_a_new_segment():
    _, s2, s3 = run([
        ([Fact(LIN, 18)], []),
        ([], [Span(None)]),  # 多年以后
        ([Fact(SU, 30, char_start=5)], [Span(2, char_start=9)]),
    ])  # fmt: skip
    assert age_now(s2) is None  # never projected across the jump
    assert s2.characters[LIN].age_low == 18  # but the stated age is still there
    assert age_now(s3, SU) == (32, 32)  # ages stated after the jump project again


def test_only_present_time_stated_ages_update_the_state():
    (s1,) = run([([
        Fact(LIN, 18, char_start=0),
        Fact(LIN, 12, char_start=10, is_flashback=True),
        Fact(LIN, 40, char_start=20, is_speculative=True),
        Fact(LIN, 3, char_start=30, attribute="relative_age"),
        Fact(LIN, 19, char_start=40),  # the last present-time age wins
    ], [])])  # fmt: skip
    assert s1.characters[LIN].age_low == 19


def test_approximate_age_and_life_stage():
    (s1,) = run([([
        Fact(SU, 30, value_max=39),
        Fact(LIN, None, attribute="life_stage", value_text="teen"),
    ], [])])  # fmt: skip
    assert age_now(s1, SU) == (30, 39)
    assert s1.characters[LIN].life_stage is LifeStage.TEEN


def test_flashback_and_other_kinds_of_time_do_not_move_the_present():
    (s1,) = run([([], [Span(5, is_flashback=True), Span(1, kind="retrospective"),
                       Span(0.01, kind="short")])])  # fmt: skip
    assert s1.story_time == 0


def test_names_follow_the_character_list_and_merges_keep_known_ages():
    shen, blade = uuid.uuid4(), uuid.uuid4()
    (s1,) = run([([Fact(shen, 30)], [])], people=[Person(shen, "沈砚"), Person(blade, "疤脸刀客")])
    s2 = advance(
        s1, chapter_number=2, facts=[], spans=[],
        characters=[Person(blade, "沈砚", ["疤脸刀客", "沈砚"])], merges=[(shen, blade)],
    )  # fmt: skip
    assert list(s2.characters) == [blade]
    merged = s2.characters[blade]
    assert (merged.canonical_name, merged.aliases, merged.age_low) == ("沈砚", ["疤脸刀客"], 30)


# --- storage, rollback and queries ------------------------------------------------------


@pytest.fixture
async def book(factory):
    job, _, _ = await import_book(factory, FakeBackend(respond))
    return job.book_id


async def test_import_stores_a_snapshot_per_chapter(factory, book):
    async with factory() as session:
        numbers = (await session.scalars(select(CoreSnapshot.chapter_number))).all()
        latest = await core.load_state(session, user_id=USER, book_id=book)
    assert sorted(numbers) == [1, 2, 3]
    lin = next(iter(latest.characters.values()))
    # 18 in ch.1, "三年后" 21 in ch.2, 16 in ch.3: the latest stated age wins.
    assert (lin.age_low, lin.age_chapter, latest.story_time) == (16, 3, 3)


async def test_incremental_state_equals_a_full_rebuild(factory, book):
    async with factory() as session:
        incremental = await core.load_state(session, user_id=USER, book_id=book)
        rebuilt = await core.rebuild(session, user_id=USER, book_id=book)
    assert rebuilt == incremental


async def test_restore_rolls_back_to_the_previous_chapter(factory, book):
    async with factory() as session:
        after_2 = await core.load_state(session, user_id=USER, book_id=book, as_of_chapter=2)
        restored = await core.restore(session, user_id=USER, book_id=book, chapter_number=3)
        await session.commit()
    assert restored == after_2
    async with factory() as session:
        numbers = (await session.scalars(select(CoreSnapshot.chapter_number))).all()
        current = (await session.scalars(select(CharacterStateRow))).all()
    assert sorted(numbers) == [1, 2]
    assert [(r.chapter_number, r.state["age_low"]) for r in current] == [(2, 21)]


async def test_get_character(factory, book):
    async with factory() as session:
        now = await core.get_character(session, user_id=USER, book_id=book, name="林少爷")
        then = await core.get_character(
            session, user_id=USER, book_id=book, name="林远", as_of_chapter=2
        )
    assert (now.canonical_name, now.aliases, now.as_of_chapter) == ("林远", ["林少爷"], 3)
    assert (now.age_low, now.age_chapter, now.age_quote) == (16, 3, "十六岁的林远")
    assert (now.estimated_age_low, now.estimated_age_high) == (16, 16)
    assert (then.age_low, then.age_chapter, then.as_of_chapter) == (21, 2, 2)


async def test_get_character_needs_a_core_and_the_owner(factory, book):
    async with factory() as session:
        with pytest.raises(NotFound):
            await core.get_character(session, user_id=uuid.uuid4(), book_id=book, name="林远")
        await session.execute(delete(CoreSnapshot))
        await session.commit()
        with pytest.raises(NotFound, match="重新导入"):
            await core.get_character(session, user_id=USER, book_id=book, name="林远")
