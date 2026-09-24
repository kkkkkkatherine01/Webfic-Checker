"""End to end through the services: import → extract (fake LLM) → check → report.
Runs on SQLite so it needs neither Docker nor an API key."""

import uuid

import pytest
from sqlalchemy import delete, select, update

from tests import fakes
from tests.fakes import FakeBackend
from webfic.checkers.types import Confidence, IssueStatus
from webfic.config import Settings
from webfic.db.models import AgeFactRow, Chapter, Character, CharacterAlias, IssueRow
from webfic.services import checks, imports, reports

USER = uuid.uuid4()
OTHER_USER = uuid.uuid4()

BOOK = """《测试》
第一章 下山
林远今年十八岁，终于下山了。
第二章 入城
三年后，林少爷二十一岁了，走进了京城。
第三章 旧伤
十六岁的林远摸了摸旧伤。
"""

EMPTY = {"age_statements": [], "elapsed_time_statements": []}


def age(mention, resolved, raw, value):
    return {"mention": mention, "resolved_name": resolved, "raw_text": raw,
            "statement_type": "absolute_age", "value": value, "life_stage": None,
            "is_flashback": False, "years_before_present": None}  # fmt: skip


def respond(messages):
    user = messages[1].content
    if "第 1 章" in user:
        return {
            "age_statements": [
                age("林远", None, "林远今年十八岁", 18),
                age("林远", None, "林远今年二十岁", 20),  # not in the text: must be dropped
            ],
            "elapsed_time_statements": [],
        }
    if "第 2 章" in user:
        assert "- 林远" in user  # characters from chapter 1 are passed on
        return {
            "age_statements": [age("林少爷", "林远", "林少爷二十一岁了", 21)],
            "elapsed_time_statements": [
                {"raw_text": "三年后", "estimated_years": 3, "is_flashback": False}
            ],
        }
    if "第 3 章" in user:
        return {
            "age_statements": [age("林远", "林远", "十六岁的林远", 16)],
            "elapsed_time_statements": [],
        }
    return EMPTY


def make_llm(backend, factory, book_id):
    return fakes.make_llm(backend, factory, user_id=USER, book_id=book_id)


async def import_book(factory, backend):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="测试", text=BOOK)
    events = []
    result = await imports.run_import_job(
        factory, make_llm(backend, factory, job.book_id), Settings(),
        user_id=USER, book_id=job.book_id, on_progress=events.append,
    )  # fmt: skip
    return job, result, events


async def test_full_pipeline(factory):
    backend = FakeBackend(respond)
    job, result, events = await import_book(factory, backend)

    assert [c.title for c in job.chapters] == ["第一章 下山", "第二章 入城", "第三章 旧伤"]
    assert job.warnings  # the 《测试》 line before chapter 1
    assert (result.extracted, result.failed, result.dropped_statements) == (3, 0, 1)
    assert [e.done for e in events] == [1, 2, 3]

    async with factory() as session:
        characters = (await session.scalars(select(Character))).all()
        aliases = (await session.scalars(select(CharacterAlias))).all()
    assert [c.canonical_name for c in characters] == ["林远"]
    assert [a.alias for a in aliases] == ["林少爷"]

    async with factory() as session:
        check = await checks.run_checks(session, user_id=USER, book_id=job.book_id)
    assert (check.found, check.new) == (1, 1)

    async with factory() as session:
        report = await reports.get_report(session, user_id=USER, book_id=job.book_id)
    issue = report.issues[0]
    assert issue.confidence == Confidence.CONFIRMED
    assert [e.chapter_number for e in issue.evidence] == [2, 3]
    # Evidence offsets point at the quoted text in the stored chapter.
    ch3 = BOOK.split("第三章 旧伤\n")[1].strip()
    e = issue.evidence[1]
    assert ch3[e.char_start : e.char_end] == "十六岁的林远"

    async with factory() as session:
        usage = await reports.get_usage(session, user_id=USER, book_id=job.book_id)
    assert sum(line.calls for line in usage.lines) == 3
    assert usage.total_cost_usd > 0


async def test_reimport_of_same_text_is_fully_cached(factory):
    backend = FakeBackend(respond)
    await import_book(factory, backend)
    _, second, _ = await import_book(factory, backend)

    assert len(backend.calls) == 3
    assert second.cache_hits == 3 and second.cost_usd == 0


async def test_rechecking_keeps_author_status_and_resolves_vanished_issues(factory):
    job, _, _ = await import_book(factory, FakeBackend(respond))
    async with factory() as session:
        await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        row = await session.scalar(select(IssueRow))
        row.status = IssueStatus.INTENTIONAL
        await session.commit()

    async with factory() as session:
        again = await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        row = await session.scalar(select(IssueRow))
    assert again.new == 0
    assert row.status == IssueStatus.INTENTIONAL

    async with factory() as session:
        report = await reports.get_report(session, user_id=USER, book_id=job.book_id)
    assert report.issues == []  # intentional issues are hidden by default


async def test_re_extraction_keeps_author_status(factory):
    backend = FakeBackend(respond)
    job, _, _ = await import_book(factory, backend)
    async with factory() as session:
        await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        row = await session.scalar(select(IssueRow))
        row.status = IssueStatus.INTENTIONAL
        await session.commit()

    # Extract every chapter again (served from the cache): all facts get new ids.
    async with factory() as session:
        await session.execute(update(Chapter).values(status="pending"))
        await session.commit()
    await imports.run_import_job(
        factory, make_llm(backend, factory, job.book_id), Settings(),
        user_id=USER, book_id=job.book_id,
    )  # fmt: skip

    async with factory() as session:
        again = await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        rows = (await session.scalars(select(IssueRow))).all()
    assert (again.found, again.new, again.resolved) == (1, 0, 0)
    assert [r.status for r in rows] == [IssueStatus.INTENTIONAL]


async def test_acknowledged_issue_that_no_longer_occurs_is_resolved(factory):
    job, _, _ = await import_book(factory, FakeBackend(respond))
    async with factory() as session:
        await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        row = await session.scalar(select(IssueRow))
        row.status = IssueStatus.ACKNOWLEDGED
        await session.commit()

    # The author fixed chapter 3: its conflicting age is gone.
    async with factory() as session:
        await session.execute(delete(AgeFactRow).where(AgeFactRow.chapter_number == 3))
        await session.commit()
    async with factory() as session:
        again = await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        row = await session.scalar(select(IssueRow))
    assert again.resolved == 1
    assert row.status == IssueStatus.RESOLVED


async def test_failed_chapter_is_marked_and_resumable(factory):
    def flaky(messages):
        return "garbage" if "第 2 章" in messages[1].content else respond(messages)

    job, result, events = await import_book(factory, FakeBackend(flaky))
    assert (result.extracted, result.failed) == (2, 1)
    assert events[1].status == "failed"

    resumed = await imports.run_import_job(
        factory, make_llm(FakeBackend(respond), factory, job.book_id), Settings(),
        user_id=USER, book_id=job.book_id,
    )  # fmt: skip
    assert (resumed.extracted, resumed.failed) == (1, 0)


async def test_other_users_cannot_see_the_book(factory):
    job, _, _ = await import_book(factory, FakeBackend(respond))
    async with factory() as session:
        assert await reports.list_books(session, user_id=OTHER_USER) == []
        with pytest.raises(imports.NotFound):
            await reports.get_report(session, user_id=OTHER_USER, book_id=job.book_id)


MERGE_BOOK = """第一章 渡口
疤脸刀客今年二十七岁。沈砚这个名字，江湖上无人不知。
第二章 相认
疤脸刀客抱拳道：在下沈砚。二十四岁的沈砚握紧了刀。
"""


def merge_respond(messages):
    user = messages[1].content
    if "第 1 章" in user:
        return {
            "age_statements": [age("疤脸刀客", None, "疤脸刀客今年二十七岁", 27)],
            "elapsed_time_statements": [],
        }
    return {
        "age_statements": [age("沈砚", "疤脸刀客", "二十四岁的沈砚", 24)],
        "elapsed_time_statements": [],
        "revealed_names": [{"known_as": "疤脸刀客", "real_name": "沈砚"}],
    }


async def test_revealed_name_renames_and_merges_characters(factory):
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=MERGE_BOOK)
        # A separate "沈砚" created earlier by mistake, with an alias of its own.
        wrong = Character(user_id=USER, book_id=job.book_id, canonical_name="沈砚")
        session.add(wrong)
        await session.flush()
        session.add(
            CharacterAlias(
                user_id=USER, book_id=job.book_id, character_id=wrong.id,
                alias="沈大侠", first_chapter=1,
            )
        )  # fmt: skip
        await session.commit()

    await imports.run_import_job(
        factory, make_llm(FakeBackend(merge_respond), factory, job.book_id), Settings(),
        user_id=USER, book_id=job.book_id,
    )  # fmt: skip

    async with factory() as session:
        characters = (await session.scalars(select(Character))).all()
        aliases = sorted((await session.scalars(select(CharacterAlias.alias))).all())
        await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        report = await reports.get_report(session, user_id=USER, book_id=job.book_id)

    assert [c.canonical_name for c in characters] == ["沈砚"]
    assert aliases == ["沈大侠", "疤脸刀客"]
    issue = only_issue(report)
    assert issue.confidence == Confidence.CONFIRMED  # 27 -> 24 across the rename
    assert issue.subjects == [str(characters[0].id)]


def only_issue(report):
    assert len(report.issues) == 1, [i.description for i in report.issues]
    return report.issues[0]
