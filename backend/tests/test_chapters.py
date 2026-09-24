"""Chapter management: append / replace / delete / patch, recomputing from the changed
chapter, and dry runs that leave the database untouched."""

import re
import uuid

import pytest
from sqlalchemy import select, update

from tests.fakes import FakeBackend, MemoryCallStore
from webfic.checkers.types import IssueStatus
from webfic.config import Settings
from webfic.db.models import Base, Chapter, Character, FactRow, IssueRow
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig
from webfic.services import chapters, checks, imports
from webfic.services.errors import InvalidEdit, NotFound

USER = uuid.uuid4()

BOOK = """第1章 起
林远今年18岁。
第2章 承
3年后。
林远今年21岁。
第3章 转
林远今年16岁。
第4章 合
苏晚晴今年20岁。
"""

_AGE = re.compile(r"^(\S{2,4}?)今年(\d+)岁", re.MULTILINE)
_SPAN = re.compile(r"(\d+)年后")
_REVEAL = re.compile(r"原来(\S+?)就是(\S+?)。")


def respond(messages):
    """A fake model that reads the chapter text, so answers follow edits and renumbering."""
    text = messages[1].content.rsplit("）：\n", 1)[1]
    return {
        "age_statements": [
            {"mention": m[1], "resolved_name": None, "raw_text": m[0],
             "statement_type": "absolute_age", "value": int(m[2])}
            for m in _AGE.finditer(text)
        ],
        "elapsed_time_statements": [
            {"raw_text": m[0], "estimated_years": int(m[1]), "kind": "advance"}
            for m in _SPAN.finditer(text)
        ],
        "revealed_names": [{"known_as": m[1], "real_name": m[2]} for m in _REVEAL.finditer(text)],
    }  # fmt: skip


class World:
    """One book imported with the fake model; `llm` shares one response cache."""

    def __init__(self, factory):
        self.factory = factory
        self.backend = FakeBackend(respond)
        tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
        self.llm = JsonLLMClient(self.backend, tiers, store=MemoryCallStore())
        self.book_id: uuid.UUID

    async def load(self, text=BOOK):
        async with self.factory() as session:
            job = await imports.create_import_job(session, user_id=USER, title="t", text=text)
        self.book_id = job.book_id
        await imports.run_import_job(
            self.factory, self.llm, Settings(), user_id=USER, book_id=self.book_id
        )
        async with self.factory() as session:
            await checks.run_checks(session, user_id=USER, book_id=self.book_id)
        return self

    async def do(self, operation, **kwargs):
        return await operation(
            self.factory, self.llm, Settings(), user_id=USER, book_id=self.book_id, **kwargs
        )

    async def issues(self):
        async with self.factory() as session:
            rows = (await session.scalars(select(IssueRow))).all()
        return [(sorted({e["chapter_number"] for e in r.evidence}), r.status) for r in rows]


def ends(issue):
    return sorted({e.chapter_number for e in issue.evidence})


@pytest.fixture
async def world(factory):
    return await World(factory).load()


async def test_starting_point(world):
    assert await world.issues() == [([2, 3], IssueStatus.OPEN)]  # 21 -> 16: going backwards


async def test_replacing_a_chapter_fixes_a_contradiction(world):
    calls = len(world.backend.calls)
    result = await world.do(chapters.replace_chapter, number=3, content="林远今年21岁。")
    assert [ends(i) for i in result.issues_removed] == [[2, 3]]
    assert result.issues_added == []
    # Chapters 3 and 4 are recomputed; only the changed one is read by the model again,
    # the unchanged one reuses its stored extraction.
    assert (result.extracted, result.llm_calls, result.reused) == (2, 1, 1)
    assert len(world.backend.calls) == calls + 1
    assert await world.issues() == [([2, 3], IssueStatus.RESOLVED)]


async def test_patching_a_passage_introduces_a_contradiction(world):
    result = await world.do(chapters.patch_chapter, number=1, old="18岁", new="10岁")
    assert [ends(i) for i in result.issues_added] == [[1, 2]]  # 10 + 3 years is not 21
    assert result.issues_removed == []  # the 2 -> 3 issue is still the same issue


async def test_the_authors_status_survives_recomputation(world):
    async with world.factory() as session:
        await session.execute(update(IssueRow).values(status=IssueStatus.INTENTIONAL))
        await session.commit()
    result = await world.do(
        chapters.replace_chapter, number=1, content="林远今年18岁。\n天气很好。"
    )
    assert (result.issues_added, result.issues_removed) == ([], [])
    assert await world.issues() == [([2, 3], IssueStatus.INTENTIONAL)]


async def test_deleting_a_chapter_renumbers_the_rest(world):
    result = await world.do(chapters.delete_chapter, number=2)
    async with world.factory() as session:
        titles = (await session.scalars(select(Chapter.title).order_by(Chapter.number))).all()
        numbers = (await session.scalars(select(Chapter.number).order_by(Chapter.number))).all()
        facts = (await session.scalars(select(FactRow.chapter_number))).all()
    assert (titles, numbers, result.chapters) == (
        ["第1章 起", "第3章 转", "第4章 合"],
        [1, 2, 3],
        3,
    )
    assert sorted(facts) == [1, 2, 3]
    assert [ends(i) for i in result.issues_added] == [[1, 2]]  # 18 -> 16, now chapters 1 and 2
    assert [ends(i) for i in result.issues_removed] == [[2, 3]]


async def test_appending_extracts_only_the_new_chapters(world):
    calls = len(world.backend.calls)
    result = await world.do(chapters.append_chapters, text="第5章 新\n2年后。\n林远今年18岁。")
    assert (result.chapters, result.extracted, len(world.backend.calls)) == (5, 1, calls + 1)
    assert result.issues_added == [] and result.issues_removed == []  # 16 + 2 = 18


async def test_changing_the_chapter_that_revealed_a_name_undoes_the_merge(factory):
    world = await World(factory).load(
        "第1章 甲\n疤脸刀客今年30岁。\n第2章 乙\n沈砚今年30岁。\n第3章 丙\n原来疤脸刀客就是沈砚。"
    )

    async def names():
        async with factory() as session:
            return sorted((await session.scalars(select(Character.canonical_name))).all())

    assert await names() == ["沈砚"]
    await world.do(chapters.replace_chapter, number=3, content="什么也没发生。")
    assert await names() == ["沈砚", "疤脸刀客"]


async def dump(factory):
    """Every table except the usage ledger, as sorted rows."""
    async with factory() as session:
        return {
            table.name: sorted(map(repr, (await session.execute(select(table))).all()))
            for table in Base.metadata.sorted_tables
            if table.name != "llm_calls"
        }


@pytest.mark.parametrize(
    ("operation", "kwargs"),
    [
        (chapters.patch_chapter, {"number": 1, "old": "18岁", "new": "10岁"}),
        (chapters.delete_chapter, {"number": 2}),
        (chapters.append_chapters, {"text": "第5章 新\n林远今年40岁。"}),
    ],
)
async def test_a_dry_run_reports_but_changes_nothing(world, operation, kwargs):
    before = await dump(world.factory)
    tried = await world.do(operation, dry_run=True, **kwargs)
    assert tried.dry_run and tried.issues_added
    assert await dump(world.factory) == before

    done = await world.do(operation, **kwargs)  # the real thing gives the same answer
    assert [ends(i) for i in done.issues_added] == [ends(i) for i in tried.issues_added]
    assert await dump(world.factory) != before


async def test_invalid_edits_are_refused_without_changes(world):
    before = await dump(world.factory)
    with pytest.raises(InvalidEdit, match="找不到"):
        await world.do(chapters.patch_chapter, number=2, old="二十岁", new="三十岁")
    with pytest.raises(InvalidEdit, match="2 次"):
        await world.do(chapters.patch_chapter, number=2, old="年", new="月")
    with pytest.raises(InvalidEdit):
        await world.do(chapters.replace_chapter, number=2, content="  \n")
    with pytest.raises(NotFound):
        await world.do(chapters.delete_chapter, number=9)
    assert await dump(world.factory) == before


async def test_other_users_cannot_touch_the_book(world):
    with pytest.raises(NotFound):
        await chapters.delete_chapter(
            world.factory, world.llm, Settings(), user_id=uuid.uuid4(), book_id=world.book_id,
            number=1,
        )  # fmt: skip


# --- reusing stored extractions ---------------------------------------------------------


async def test_unchanged_chapters_reuse_their_extraction_even_if_the_cast_changes(world):
    # Chapter 1 now introduces 苏晚晴: every later prompt's list of known characters
    # changes, so without stored extractions chapters 2-4 would be read again.
    calls = len(world.backend.calls)
    result = await world.do(
        chapters.replace_chapter, number=1, content="林远今年18岁。\n苏晚晴今年19岁。"
    )
    assert (result.extracted, result.reused, len(world.backend.calls)) == (4, 3, calls + 1)
    async with world.factory() as session:
        su = (
            await session.scalars(select(Character).where(Character.canonical_name == "苏晚晴"))
        ).all()
        owners = (
            await session.scalars(
                select(FactRow.character_id).where(FactRow.value_num.in_([19, 20]))
            )
        ).all()
    assert len(su) == 1 and set(owners) == {su[0].id}  # chapter 4 still finds her by name


async def test_recomputing_unchanged_text_changes_nothing(world):
    async with world.factory() as session:
        content = (await session.scalar(select(Chapter).where(Chapter.number == 2))).content
    before = await dump(world.factory)
    result = await world.do(chapters.replace_chapter, number=2, content=content)
    assert (result.llm_calls, result.reused, result.issues_added, result.issues_removed) == (
        0, 3, [], [],
    )  # fmt: skip
    after = await dump(world.factory)
    # Same facts, characters and issues; only row ids and timestamps of the rewritten
    # rows differ, so compare the reports instead of raw rows.
    assert len(after["facts"]) == len(before["facts"])
    assert await world.issues() == [([2, 3], IssueStatus.OPEN)]


async def test_a_new_extraction_setup_reads_chapters_again(world):
    async def replace_with_settings(*args, **kwargs):
        args = list(args)
        args[2] = Settings(chunk_overlap=400)  # a different setup: stored results don't fit
        return await chapters.replace_chapter(*args, **kwargs)

    async with world.factory() as session:
        content = (await session.scalar(select(Chapter).where(Chapter.number == 3))).content
    result = await world.do(replace_with_settings, number=3, content=content)
    assert result.reused == 0
