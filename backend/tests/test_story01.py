"""story01 with an ideal (hand-written) extraction: verifies that, given correct
extraction, the checker reports exactly the planted contradictions and none of the
distractors. Extraction quality itself is measured by webfic-eval against the real model."""

import uuid

from tests.fakes import FakeBackend, make_llm
from tests.story01_ideal import STORY01, needs_golden, respond
from webfic.checkers.types import Confidence
from webfic.config import Settings
from webfic.services import checks, imports, reports

STORY = STORY01 / "text.txt"
USER = uuid.UUID("00000000-0000-0000-0000-000000000001")


@needs_golden
async def test_story01_ideal_extraction_finds_exactly_the_planted_issues(factory):
    async with factory() as session:
        job = await imports.create_import_job(
            session, user_id=USER, title="story01", text=STORY.read_text("utf-8")
        )
    assert len(job.chapters) == 5

    result = await imports.run_import_job(
        factory, make_llm(FakeBackend(respond), factory, user_id=USER, book_id=job.book_id),
        Settings(),
        user_id=USER, book_id=job.book_id,
    )  # fmt: skip
    assert (result.extracted, result.dropped_statements) == (5, 0)  # every quote located

    async with factory() as session:
        await checks.run_checks(session, user_id=USER, book_id=job.book_id)
        report = await reports.get_report(session, user_id=USER, book_id=job.book_id)

    found = sorted(
        (i.confidence, tuple(e.chapter_number for e in i.evidence)) for i in report.issues
    )
    assert found == sorted(
        [
            (Confidence.SUSPECTED_REVIEW, (1, 3, 3)),  # 苏晚晴 19 → 25 with 2 years
            (Confidence.SUSPECTED_REVIEW, (1, 4)),  # 白发老者 vs 30
            (Confidence.CONFIRMED, (4, 5)),  # 林远 21 → 19
        ]
    ), [i.description for i in report.issues]
