"""Verify reported issues with the verify agent, and look up the verdicts for reports.

A verdict stays valid while its key matches: the issue (fingerprint), the quoted
evidence and the text of the chapters it quotes. Renumbering chapters does not change
the key (chapter texts are identified by their content hash), editing a quoted chapter
does. The model and prompt are recorded but not part of the key: switching models does
not silently turn verified reports back into unverified ones (verify with `again` for
that).
"""

import hashlib
import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.agent.budget import Budget
from webfic.agent.loop import run_agent
from webfic.agent.tools import ToolContext
from webfic.agent.trace import TraceWriter
from webfic.agent.verify import load_prompt, prompt_for, tools_for
from webfic.archival.index import Archival
from webfic.checkers.types import Confidence, IssueStatus, IssueType
from webfic.config import Settings
from webfic.db.models import (
    Book,
    Chapter,
    Character,
    CharacterAlias,
    ElapsedTimeFactRow,
    FactRow,
    IssueRow,
    IssueVerificationRow,
)
from webfic.facts.describe import describe_span
from webfic.facts.registry import AGE, APPEARANCE, KINSHIP, LIFE, describe_fact
from webfic.llm.base import LLMClient, Tier
from webfic.services.errors import NotFound

Factory = async_sessionmaker[AsyncSession]

# The kind of fact an issue's evidence quotes, by issue type (step 5-1).
_CATEGORY = {
    IssueType.CHARACTER_AGE: AGE.name,
    IssueType.FACT_APPEARANCE: APPEARANCE.name,
    IssueType.CHARACTER_KINSHIP: KINSHIP.name,
    IssueType.TIMELINE_REVIVAL: LIFE.name,
}

_CONFIDENCE = {
    Confidence.CONFIRMED: "确定矛盾",
    Confidence.SUSPECTED_REVIEW: "疑似矛盾",
    Confidence.INSUFFICIENT_INFO: "信息不足",
}


class VerificationView(BaseModel):
    status: str  # the agent run's: done / budget_exhausted / guard_failed / failed
    verdict: str | None  # contradiction / false_alarm / needs_author; None: no answer
    reason: str | None
    explanation: str | None
    evidence: list[dict[str, Any]]
    run_id: uuid.UUID | None
    model: str
    created_at: datetime


class IssueOutcome(BaseModel):
    issue_id: uuid.UUID
    description: str
    verification: VerificationView
    turns: int  # model calls
    tool_calls: int


class VerifyResult(BaseModel):
    pending: int  # open / acknowledged issues without a valid verdict (or all, with again)
    verified: list[IssueOutcome]
    already: int  # issues skipped because their verdict is still valid
    cost_usd: Decimal


# --- keys and lookup -------------------------------------------------------------------------


async def chapter_hashes(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID
) -> dict[int, str]:
    rows = await session.execute(
        select(Chapter.number, Chapter.content_hash).where(
            Chapter.user_id == user_id, Chapter.book_id == book_id
        )
    )
    return {n: h for n, h in rows}


def verification_key(issue: IssueRow, hashes: dict[int, str]) -> str:
    evidence = [
        [hashes.get(e["chapter_number"], "?"), e["quote"], e["char_start"], e["char_end"]]
        for e in issue.evidence
    ]
    blob = json.dumps([issue.fingerprint, evidence], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def _view(row: IssueVerificationRow) -> VerificationView:
    return VerificationView(
        status=row.status, verdict=row.verdict, reason=row.reason, explanation=row.explanation,
        evidence=row.evidence or [], run_id=row.run_id, model=row.model,
        created_at=row.created_at,
    )  # fmt: skip


async def current_verifications(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, issues: list[IssueRow]
) -> dict[uuid.UUID, VerificationView]:
    """The latest still-valid verdict of each issue that has one."""
    if not issues:
        return {}
    hashes = await chapter_hashes(session, user_id, book_id)
    keys = {i.id: verification_key(i, hashes) for i in issues}
    rows = await session.scalars(
        select(IssueVerificationRow)
        .where(
            IssueVerificationRow.user_id == user_id,
            IssueVerificationRow.issue_id.in_(list(keys)),
        )
        .order_by(IssueVerificationRow.created_at, IssueVerificationRow.id)
    )
    found: dict[uuid.UUID, VerificationView] = {}
    for row in rows:  # oldest first, so the latest valid one wins
        if keys.get(row.issue_id) == row.key:
            found[row.issue_id] = _view(row)
    return found


# --- the task -----------------------------------------------------------------------------


async def build_task(
    session: AsyncSession, user_id: uuid.UUID, book_id: uuid.UUID, issue: IssueRow
) -> str:
    """What the agent is told about the issue: the report, and for each piece of evidence
    what extraction made of it (the age statement with its character, or the time span)."""
    subjects = {uuid.UUID(s) for s in issue.subjects or []}
    lines = [
        f"待核实的矛盾（{_CONFIDENCE.get(Confidence(issue.confidence), issue.confidence)}）：",
        issue.description,
        "",
        "涉及的原文（位置是章内字符偏移，可直接用于 read_passage）：",
    ]
    for n, e in enumerate(issue.evidence, start=1):
        where = (
            f"[{n}] 第 {e['chapter_number']} 章 {e['char_start']}–{e['char_end']}「{e['quote']}」"
        )
        lines.append(where)
        lines += await _describe_evidence(
            session, user_id, book_id, e, subjects, _CATEGORY.get(issue.issue_type, AGE.name)
        )
    lines += ["", "请核实后调用 submit_verdict 提交结论。"]
    return "\n".join(lines)


async def _describe_evidence(
    session: AsyncSession,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    e: dict[str, Any],
    subjects: set[uuid.UUID],
    category: str,
) -> list[str]:
    same_place = (
        FactRow.user_id == user_id,
        FactRow.book_id == book_id,
        FactRow.category == category,
        FactRow.chapter_number == e["chapter_number"],
        FactRow.char_start == e["char_start"],
        FactRow.char_end == e["char_end"],
    )
    facts = (await session.scalars(select(FactRow).where(*same_place))).all()
    # Several people may share one quote; the issue is about its subject.
    fact = next((f for f in facts if f.character_id in subjects), facts[0] if facts else None)
    if fact is not None:
        character = await session.get(Character, fact.character_id)
        aliases = sorted(
            await session.scalars(
                select(CharacterAlias.alias).where(
                    CharacterAlias.user_id == user_id,
                    CharacterAlias.character_id == fact.character_id,
                )
            )
        )
        name = character.canonical_name if character else "?"
        also = f"（别名：{'、'.join(aliases)}）" if aliases else ""
        return [
            f"    角色：{name}{also}；原文称呼：{fact.mention}",
            f"    抽取结果：{describe_fact(fact)}",
        ]
    span = await session.scalar(
        select(ElapsedTimeFactRow).where(
            ElapsedTimeFactRow.user_id == user_id,
            ElapsedTimeFactRow.book_id == book_id,
            ElapsedTimeFactRow.chapter_number == e["chapter_number"],
            ElapsedTimeFactRow.char_start == e["char_start"],
            ElapsedTimeFactRow.char_end == e["char_end"],
        )
    )
    if span is not None:
        return [f"    时间段：{describe_span(span.kind, span.estimated_years, span.is_flashback)}"]
    return ["    （没有找到对应的抽取记录）"]


# --- verifying ------------------------------------------------------------------------------


async def verify_issues(
    factory: Factory,
    llm: LLMClient,
    settings: Settings,
    *,
    user_id: uuid.UUID,
    book_id: uuid.UUID,
    issue_ids: list[uuid.UUID] | None = None,
    again: bool = False,
    archival: Archival | None = None,
    budget: Budget | None = None,
    trace_factory: Factory | None = None,
    prompt_version: str | None = None,
) -> VerifyResult:
    """Run the verify agent on the book's open and acknowledged issues that have no
    valid verdict yet (all of them with `again`), one after another. `trace_factory`
    (default: `factory`) is where execution records go: the evaluation verifies inside a
    transaction it rolls back, but keeps the records."""
    async with factory() as session:
        owned = select(Book.id).where(Book.id == book_id, Book.user_id == user_id)
        if await session.scalar(owned) is None:
            raise NotFound(f"book {book_id}")
        query = select(IssueRow).where(
            IssueRow.user_id == user_id,
            IssueRow.book_id == book_id,
            IssueRow.status.in_([IssueStatus.OPEN, IssueStatus.ACKNOWLEDGED]),
        )
        if issue_ids is not None:
            query = query.where(IssueRow.id.in_(issue_ids))
        issues = (await session.scalars(query.order_by(IssueRow.created_at, IssueRow.id))).all()
        valid = {} if again else await current_verifications(session, user_id, book_id, issues)
        hashes = await chapter_hashes(session, user_id, book_id)
        todo = [(i, verification_key(i, hashes)) for i in issues if i.id not in valid]
        tasks = {i.id: await build_task(session, user_id, book_id, i) for i, _ in todo}

    context = ToolContext(factory=factory, user_id=user_id, book_id=book_id, archival=archival)
    prompts: dict[str, str] = {}  # version -> text
    verified: list[IssueOutcome] = []
    cost = Decimal(0)
    for issue, key in todo:
        # Each checker's issues have their own prompt; `prompt_version` overrides it for
        # all (evaluation of a new prompt).
        version = prompt_version or prompt_for(issue.checker)
        if version not in prompts:
            prompts[version] = load_prompt(version)
        config = {
            "prompt": version,
            "model": settings.llm_verify_model,
            "extra": settings.llm_verify_extra,
            "issue_id": str(issue.id),
        }
        result = await run_agent(
            llm, agent="verify", system=prompts[version], task=tasks[issue.id],
            tools=tools_for(version),
            context=context,
            trace=TraceWriter(trace_factory or factory, user_id=user_id, book_id=book_id),
            budget=budget, tier=Tier.VERIFY, subject=issue.fingerprint, config=config,
        )  # fmt: skip
        answer = result.result or {}
        row = IssueVerificationRow(
            user_id=user_id, book_id=book_id, issue_id=issue.id, key=key,
            run_id=result.run_id, status=result.status, verdict=answer.get("verdict"),
            reason=answer.get("reason"), explanation=answer.get("explanation") or result.error,
            evidence=answer.get("evidence", []), model=settings.llm_verify_model,
            prompt_version=version, cost_usd=result.spend.cost_usd,
            # Set here, to the microsecond: the latest valid verdict wins, and the
            # database's own timestamp is only to the second on SQLite.
            created_at=datetime.now(UTC),
        )  # fmt: skip
        async with factory() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
        cost += result.spend.cost_usd
        verified.append(
            IssueOutcome(
                issue_id=issue.id, description=issue.description, verification=_view(row),
                turns=result.spend.turns, tool_calls=result.spend.tool_calls,
            )
        )  # fmt: skip
    return VerifyResult(
        pending=len(todo), verified=verified, already=len(issues) - len(todo), cost_usd=cost
    )
