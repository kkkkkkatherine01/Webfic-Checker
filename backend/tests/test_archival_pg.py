"""Search on the real Postgres database (pgvector + full-text search), which is the path
used in production and by the retrieval evaluation. Skipped when Postgres is not running
or not migrated to the latest revision; cleans up after itself."""

import uuid

from sqlalchemy import delete, select

from tests import test_chapters
from tests.fakes import make_archival
from tests.test_chapters import World
from tests.test_chapters_pg import pg_factory  # noqa: F401  (fixture)
from webfic.db.models import Book, LLMCallRow, PassageRow
from webfic.memory.archival import search_text


async def test_search_on_postgres(pg_factory, monkeypatch, tmp_path):  # noqa: F811
    user = uuid.uuid4()
    monkeypatch.setattr(test_chapters, "USER", user)
    world = World(pg_factory, make_archival(tmp_path, size=40, overlap=10))
    try:
        await world.load()
        async with pg_factory() as session:
            count = len(
                (
                    await session.scalars(select(PassageRow.id).where(PassageRow.user_id == user))
                ).all()
            )

            async def search(query, **filters):
                return await search_text(
                    session, world.archival, user_id=user, book_id=world.book_id, query=query,
                    **filters,
                )  # fmt: skip

            top = (await search("苏晚晴多大", k=1))[0]
            by_keyword = (await search("苏晚晴", k=1, mode="keyword"))[0]
            by_vector = (await search("苏晚晴今年", k=1, mode="vector"))[0]
            in_range = await search("今年多少岁", k=10, chapters=(2, 3))
            of_lin = await search("今年多少岁", k=10, character="林远")
            nothing = await search("完全无关的词语组合", k=5, mode="keyword")
        assert count >= 4
        assert top.chapter_number == 4 and set(top.matched_by) == {"vector", "keyword"}
        assert by_keyword.chapter_number == 4 and by_vector.chapter_number == 4
        assert in_range and {h.chapter_number for h in in_range} <= {2, 3}
        assert {h.chapter_number for h in of_lin} == {1, 2, 3}
        assert nothing == []
    finally:
        async with pg_factory() as session:
            await session.execute(delete(Book).where(Book.user_id == user))  # cascades
            await session.execute(delete(LLMCallRow).where(LLMCallRow.user_id == user))
            await session.commit()
