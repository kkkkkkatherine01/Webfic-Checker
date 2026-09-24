"""The character change log: every change made while extracting a chapter is logged,
and undoing chapters N.. restores the character table exactly as it was after N-1."""

import uuid

from sqlalchemy import select

from tests import fakes
from tests.fakes import FakeBackend
from webfic.config import Settings
from webfic.db.models import Character, CharacterAlias, CharacterEvent, FactRow
from webfic.memory import events
from webfic.services import imports

USER = uuid.uuid4()

BOOK = """第一章 刀客
疤脸刀客今年三十岁。
第二章 沈家
沈砚今年三十岁。
第三章 真相
原来疤脸刀客就是沈砚。林少爷十八岁。
"""


def age(mention, resolved, raw, value):
    return {"mention": mention, "resolved_name": resolved, "raw_text": raw,
            "statement_type": "absolute_age", "value": value}  # fmt: skip


def respond(messages):
    user = messages[1].content
    if "第 1 章" in user:
        return {"age_statements": [age("疤脸刀客", None, "疤脸刀客今年三十岁", 30)]}
    if "第 2 章" in user:
        return {"age_statements": [age("沈砚", None, "沈砚今年三十岁", 30)]}
    return {
        "age_statements": [age("林少爷", "林远", "林少爷十八岁", 18)],
        "revealed_names": [{"known_as": "疤脸刀客", "real_name": "沈砚"}],
    }


async def run(factory, backend, book_id):
    llm = fakes.make_llm(backend, factory, user_id=USER, book_id=book_id)
    return await imports.run_import_job(factory, llm, Settings(), user_id=USER, book_id=book_id)


async def table_state(factory, *, before_chapter=None):
    """Characters, aliases and who each fact belongs to (optionally before a chapter)."""
    async with factory() as session:
        characters = {(c.id, c.canonical_name) for c in await session.scalars(select(Character))}
        aliases = {(a.character_id, a.alias) for a in await session.scalars(select(CharacterAlias))}
        query = select(FactRow)
        if before_chapter is not None:
            query = query.where(FactRow.chapter_number < before_chapter)
        facts = {(f.id, f.character_id) for f in await session.scalars(query)}
    return characters, aliases, facts


async def test_undoing_a_chapter_restores_the_character_table(factory):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)

    # Chapter 3 fails first, so we can see the tables right after chapter 2.
    flaky = FakeBackend(lambda m: "garbage" if "第 3 章" in m[1].content else respond(m))
    first = await run(factory, flaky, job.book_id)
    assert (first.extracted, first.failed) == (2, 1)
    after_chapter_2 = await table_state(factory)
    assert {name for _, name in after_chapter_2[0]} == {"疤脸刀客", "沈砚"}

    # Chapter 3 reveals 疤脸刀客 = 沈砚: merge + rename, and creates 林远 with an alias.
    await run(factory, FakeBackend(respond), job.book_id)
    characters, aliases, _ = await table_state(factory)
    assert {name for _, name in characters} == {"沈砚", "林远"}
    assert {alias for _, alias in aliases} == {"疤脸刀客", "林少爷"}
    async with factory() as session:
        kinds = [e.kind for e in await session.scalars(
            select(CharacterEvent).where(CharacterEvent.chapter_number == 3)
            .order_by(CharacterEvent.seq)
        )]  # fmt: skip
    assert kinds == ["create", "rename", "merge", "alias", "alias"]

    async with factory() as session:
        undone = await events.undo_from_chapter(
            session, user_id=USER, book_id=job.book_id, chapter_number=3
        )
        await session.commit()
    assert undone == 5
    # Same characters (same ids and names), no aliases, facts back with their owners.
    assert await table_state(factory, before_chapter=3) == after_chapter_2
    async with factory() as session:
        left = (await session.scalars(select(CharacterEvent.chapter_number))).all()
    assert sorted(left) == [1, 2]


async def test_undo_leaves_earlier_chapters_and_other_books_alone(factory):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)
        other = await imports.create_import_job(session, user_id=USER, title="u", text=BOOK)
    await run(factory, FakeBackend(respond), job.book_id)
    await run(factory, FakeBackend(respond), other.book_id)
    before = await table_state(factory)

    async with factory() as session:
        assert await events.undo_from_chapter(
            session, user_id=USER, book_id=job.book_id, chapter_number=4
        ) == 0  # fmt: skip
        assert await events.undo_from_chapter(
            session, user_id=uuid.uuid4(), book_id=job.book_id, chapter_number=1
        ) == 0  # fmt: skip
        await session.commit()
    assert await table_state(factory) == before


async def test_undo_everything_empties_the_character_table(factory):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)
    await run(factory, FakeBackend(respond), job.book_id)
    async with factory() as session:
        await events.undo_from_chapter(session, user_id=USER, book_id=job.book_id, chapter_number=1)
        await session.commit()
    characters, aliases, facts = await table_state(factory)
    assert (characters, aliases, facts) == (set(), set(), set())
