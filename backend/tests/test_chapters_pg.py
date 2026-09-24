"""A dry run against the real Postgres database (skipped when it is not running or not
migrated): every row of the test user is unchanged afterwards, except the LLM usage
ledger, which keeps the calls that were really made. Cleans up after itself."""

import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import delete, func, select, text

from tests import test_chapters
from tests.test_chapters import World
from webfic.config import Settings
from webfic.db.models import Base, Book, LLMCallRow
from webfic.db.session import make_engine, make_session_factory
from webfic.llm.cache import DbCallStore
from webfic.llm.client import JsonLLMClient
from webfic.services import chapters

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture
async def pg_factory():
    url = Settings().database_url
    if not url.startswith("postgresql"):
        pytest.skip("WEBFIC_DATABASE_URL is not Postgres")
    engine = make_engine(url)
    head = ScriptDirectory.from_config(Config(str(BACKEND / "alembic.ini"))).get_current_head()
    try:
        async with engine.connect() as conn:
            current = await conn.scalar(text("SELECT version_num FROM alembic_version"))
    except Exception as exc:  # not running
        await engine.dispose()
        pytest.skip(f"Postgres not available: {type(exc).__name__}")
    if current != head:
        await engine.dispose()
        pytest.skip(f"dev database at {current}, code at {head}: run alembic upgrade head")
    yield make_session_factory(engine)
    await engine.dispose()


async def rows_of(factory, user_id):
    async with factory() as session:
        return {
            table.name: sorted(
                map(
                    repr,
                    (await session.execute(select(table).where(table.c.user_id == user_id))).all(),
                )
            )
            for table in Base.metadata.sorted_tables
            if table.name != "llm_calls" and "user_id" in table.c
        }


async def test_dry_run_on_postgres(pg_factory, monkeypatch):
    user = uuid.uuid4()
    monkeypatch.setattr(test_chapters, "USER", user)
    world = World(pg_factory)
    try:
        await world.load()
        world.llm = JsonLLMClient(
            world.backend, world.llm._tiers,
            store=DbCallStore(pg_factory, user_id=user, book_id=world.book_id),
        )  # fmt: skip
        async with pg_factory() as session:
            ledger = select(func.count()).where(LLMCallRow.user_id == user)
            calls_before = await session.scalar(ledger)
        before = await rows_of(pg_factory, user)

        tried = await world.do(
            chapters.patch_chapter, number=1, old="18岁", new="10岁", dry_run=True
        )

        assert tried.dry_run and [test_chapters.ends(i) for i in tried.issues_added] == [[1, 2]]
        assert await rows_of(pg_factory, user) == before
        async with pg_factory() as session:
            assert await session.scalar(ledger) > calls_before  # the real call is on record
    finally:
        async with pg_factory() as session:
            await session.execute(delete(Book).where(Book.user_id == user))  # cascades
            await session.execute(delete(LLMCallRow).where(LLMCallRow.user_id == user))
            await session.commit()
