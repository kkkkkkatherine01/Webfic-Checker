"""The whole processing of a book, whatever the entry point (CLI now, the MCP server in
step 6, the web page in step 7): extract what is not extracted yet, check, and have the
verify agent look at the issues found (step 5-0: this used to be strung together in the
CLI).

Chapter operations (append, replace, delete, patch, author's notes) run the same way:
the operation recomputes and re-checks, and the new issues are verified unless it was a
dry run.
"""

import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.archival.index import Archival
from webfic.config import Settings
from webfic.llm.base import LLMClient
from webfic.services import imports
from webfic.services.chapters import ChangeResult
from webfic.services.checks import CheckResult, run_checks
from webfic.services.verification import VerifyResult, verify_issues

Factory = async_sessionmaker[AsyncSession]
Stage = Literal["extract", "check", "verify"]
OnStage = Callable[[Stage], None]


class ProcessResult(BaseModel):
    extraction: imports.ImportJobResult
    checks: CheckResult | None = None  # None: not checked
    verification: VerifyResult | None = None  # None: not verified


class ChangeOutcome(BaseModel):
    change: ChangeResult
    verification: VerifyResult | None = None  # None: not verified


async def process_book(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    archival: Archival | None = None,
    check: bool = True,
    verify: bool = True,
    on_progress: imports.ProgressCallback | None = None,
    on_stage: OnStage | None = None,
) -> ProcessResult:
    """Extract (resuming where it stopped), then check and verify. Verifying needs the
    checks: `verify` is ignored without `check`."""
    _stage(on_stage, "extract")
    extraction = await imports.run_import_job(
        factory, llm, settings, user_id=user_id, book_id=book_id,
        on_progress=on_progress, archival=archival,
    )  # fmt: skip
    result = ProcessResult(extraction=extraction)
    if not check:
        return result
    _stage(on_stage, "check")
    async with factory() as session:
        result.checks = await run_checks(session, user_id=user_id, book_id=book_id)
    if verify:
        _stage(on_stage, "verify")
        result.verification = await verify_issues(
            factory, llm, settings, user_id=user_id, book_id=book_id, archival=archival
        )
    return result


async def change_book(
    operation: Callable[..., Awaitable[ChangeResult]],
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    archival: Archival | None = None,
    verify: bool = True,
    on_stage: OnStage | None = None,
    **kwargs: Any,
) -> ChangeOutcome:
    """Run a chapter operation (`services.chapters`), then verify the issues it left,
    unless it was a dry run (whose database changes are gone by now)."""
    _stage(on_stage, "extract")
    change = await operation(
        factory, llm, settings, user_id=user_id, book_id=book_id, archival=archival, **kwargs
    )
    outcome = ChangeOutcome(change=change)
    if verify and not change.dry_run:
        _stage(on_stage, "verify")
        outcome.verification = await verify_issues(
            factory, llm, settings, user_id=user_id, book_id=book_id, archival=archival
        )
    return outcome


def _stage(on_stage: OnStage | None, stage: Stage) -> None:
    if on_stage is not None:
        on_stage(stage)
