"""services.pipeline (step 5-0): the whole processing of a book and of a chapter
operation, whatever the entry point."""

from decimal import Decimal

from tests.test_chapters import BOOK, USER, World
from webfic.config import Settings
from webfic.services import chapters, imports, pipeline
from webfic.services.verification import VerifyResult


def fake_verify(calls):
    async def verify_issues(factory, llm, settings, *, user_id, book_id, archival=None):
        calls.append(book_id)
        return VerifyResult(pending=0, verified=[], already=0, cost_usd=Decimal(0))

    return verify_issues


async def test_a_book_is_extracted_checked_and_verified_in_order(factory, monkeypatch):
    world = World(factory)
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)
    calls, stages = [], []
    monkeypatch.setattr(pipeline, "verify_issues", fake_verify(calls))
    result = await pipeline.process_book(
        factory, world.llm, Settings(), user_id=USER, book_id=job.book_id,
        on_stage=stages.append,
    )  # fmt: skip
    assert result.extraction.extracted == 4 and result.checks.found == 1
    assert result.verification is not None and calls == [job.book_id]
    assert stages == ["extract", "check", "verify"]


async def test_without_checks_nothing_is_verified(factory, monkeypatch):
    world = World(factory)
    async with factory() as session:
        job = await imports.create_import_job(session, user_id=USER, title="t", text=BOOK)
    calls = []
    monkeypatch.setattr(pipeline, "verify_issues", fake_verify(calls))
    result = await pipeline.process_book(
        factory, world.llm, Settings(), user_id=USER, book_id=job.book_id, check=False
    )
    assert result.checks is None and result.verification is None and calls == []


async def test_a_chapter_change_is_verified_unless_it_is_a_dry_run(factory, monkeypatch):
    world = await World(factory).load()
    calls = []
    monkeypatch.setattr(pipeline, "verify_issues", fake_verify(calls))
    tried = await pipeline.change_book(
        chapters.replace_chapter, factory, world.llm, Settings(), user_id=USER,
        book_id=world.book_id, number=3, content="林远今年21岁。", dry_run=True,
    )  # fmt: skip
    assert tried.change.dry_run and tried.verification is None and calls == []
    done = await pipeline.change_book(
        chapters.replace_chapter, factory, world.llm, Settings(), user_id=USER,
        book_id=world.book_id, number=3, content="林远今年21岁。",
    )  # fmt: skip
    assert len(done.change.issues_removed) == 1 and calls == [world.book_id]
