"""Evaluation of the verify agent (step 4-3).

Three sets of issues, each with the verdicts that count as right:

- keep: real contradictions, which must not be dismissed: the dev test stories' reported
  issues (story05, the held-out story, is left out) and contradictions injected into
  real novels by editing their text (step 3.5-2's injections that the checker detects)
- synthetic: false alarms made on purpose by corrupting stored extraction results
  without touching the text ("抽取错误注入"): a flashback age marked present, a guess
  marked a fact, a value changed, a retrospective or future span marked as the story
  moving on, an age moved to another character. The text still says what it said, so
  reading it shows the issue is false. The development set: prompts may be tuned on it
- real: the original-text reports on the injection bases, labelled by the user
  (`eval/external/inject/labels.yaml`). Held out: only run at milestones

Every case runs in a transaction that is rolled back (the text edit or fact corruption,
the re-check, the verdict), while the agent's execution records are kept.
"""

import asyncio
import random
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.agent.budget import Budget
from webfic.archival.index import Archival, reindex_book
from webfic.config import Settings
from webfic.db.models import (
    AgentStepRow,
    Book,
    Character,
    CharacterAlias,
    ElapsedTimeFactRow,
    FactRow,
    IssueRow,
)
from webfic.db.session import rolled_back
from webfic.evaluation import inject as ij
from webfic.evaluation.golden import load_story
from webfic.evaluation.metrics import Ratio
from webfic.evaluation.retrieval import RETRIEVAL_USER
from webfic.facts.registry import AGE
from webfic.llm.base import LLMClient
from webfic.services import chapters, checks, verification

Factory = async_sessionmaker[AsyncSession]
SetName = Literal["keep", "synthetic", "real"]

SEED = 20260927
CORRUPTIONS = ("flashback", "speculative", "value", "time_span", "misattributed")
LABELS_FILE = Path("external") / "inject" / "labels.yaml"  # under eval/


class Corruption(BaseModel):
    type: str  # one of CORRUPTIONS
    table: Literal["fact", "span"]
    row_id: uuid.UUID
    changes: dict[str, Any]
    chapter: int  # where the corrupted statement is
    start: int
    end: int


class Case(BaseModel):
    id: str
    set: SetName
    kind: str  # golden / injected:<kind> / synthetic:<type> / real
    book: str
    user_id: uuid.UUID
    book_id: uuid.UUID
    expect: list[str]  # verdicts that count as right
    reason: str | None = None  # the reason expected with a dismissal (synthetic)
    fingerprint: str | None = None  # the issue, once set up
    injection: ij.Injection | None = None
    corruption: Corruption | None = None
    note: str = ""


class CaseResult(BaseModel):
    case: Case
    sample: int
    set_up: bool  # the issue was produced (injections may not reproduce it)
    status: str | None = None  # the agent run's
    verdict: str | None = None
    reason: str | None = None
    explanation: str | None = None
    run_id: uuid.UUID | None = None
    turns: int = 0
    tool_calls: int = 0
    tools: dict[str, int] = {}
    cost_usd: Decimal = Decimal(0)
    seconds: float = 0.0

    @property
    def right(self) -> bool:
        return self.verdict in self.case.expect


# --- building the sets -----------------------------------------------------------------------


async def _books(session: AsyncSession, user_id: uuid.UUID, suffix: str) -> dict[str, uuid.UUID]:
    """name -> id of the user's books whose title ends with `suffix` (a version tag)."""
    rows = await session.execute(select(Book.title, Book.id).where(Book.user_id == user_id))
    return {t.split("#")[0]: i for t, i in rows if t.endswith(suffix)}


async def golden_cases(factory: Factory, golden_dir: Path) -> list[Case]:
    """Every issue the checker reports on the dev test stories (retrieval-eval books,
    which are extracted and indexed), judged by the stories' labels: one that matches an
    expected contradiction must be kept, one that matches none is a false alarm the agent
    should dismiss (step 4-3 relabelled story01's 赵无极), one the labels allow may go
    either way."""
    async with factory() as session:
        books = await _books(session, RETRIEVAL_USER, "@500-100")
    cases = []
    for name, book_id in sorted(books.items()):
        if not name.startswith("golden/"):
            continue
        golden = load_story(golden_dir / name.removeprefix("golden/")).golden
        async with rolled_back(factory) as scoped, scoped() as session:
            await checks.run_checks(session, user_id=RETRIEVAL_USER, book_id=book_id)
            issues = (
                await session.scalars(
                    select(IssueRow)
                    .where(IssueRow.book_id == book_id, IssueRow.status == "open")
                    .order_by(IssueRow.fingerprint)
                )
            ).all()
            subjects = {
                i.id: await _names(session, [uuid.UUID(s) for s in i.subjects]) for i in issues
            }
        for i in issues:
            ends = (i.evidence[0]["chapter_number"], i.evidence[-1]["chapter_number"])
            ends = (min(ends), max(ends))
            if any(_matches(s, golden.characters, subjects[i.id], ends) for s in golden.issues):
                set_, kind, expect = "keep", "golden", ["contradiction", "needs_author"]
            elif any(_matches(s, golden.characters, subjects[i.id], ends) for s in golden.allowed):
                set_, kind = "keep", "golden:allowed"
                expect = ["contradiction", "needs_author", "false_alarm"]
            else:
                set_, kind, expect = "synthetic", "golden:false_alarm", ["false_alarm"]
            cases.append(
                Case(
                    id=f"{name}/{i.fingerprint[:8]}",
                    set=set_,
                    kind=kind,
                    book=name,
                    user_id=RETRIEVAL_USER,
                    book_id=book_id,
                    expect=expect,
                    fingerprint=i.fingerprint,
                    note=i.description,
                )
            )
    return cases


def _matches(
    spec: Any, characters: dict[str, list[str]], subject_names: set[str], ends: tuple[int, int]
) -> bool:
    """Whether a reported issue (its subject's names, its two ends) is the labelled one."""
    names = {spec.character, *characters.get(spec.character, [])}
    return bool(names & subject_names) and ends in spec.endpoint_options()


async def _names(session: AsyncSession, ids: list[uuid.UUID]) -> set[str]:
    names = set(
        await session.scalars(select(Character.canonical_name).where(Character.id.in_(ids)))
    )
    names |= set(
        await session.scalars(
            select(CharacterAlias.alias).where(CharacterAlias.character_id.in_(ids))
        )
    )
    return names


async def injected_cases(
    factory: Factory, books: dict[str, uuid.UUID], count: int, seed: int = SEED
) -> list[Case]:
    """Contradictions injected into the text (3.5-2's detection kinds), spread over kinds."""
    async with factory() as session:
        views = [await ij.load_view(session, n, b) for n, b in sorted(books.items())]
    per_kind = {k: -(-count // len(ij.DETECT)) for k in ij.DETECT}
    injections = ij.plan(views, per_kind, seed)[:count]
    return [
        Case(
            id=inj.id,
            set="keep",
            kind=f"injected:{inj.kind}",
            book=inj.book,
            user_id=ij.INJECT_USER,
            book_id=books[inj.book],
            expect=["contradiction", "needs_author"],
            injection=inj,
            note=inj.detail,
        )
        for inj in injections
    ]


async def synthetic_cases(
    factory: Factory,
    books: dict[str, uuid.UUID],
    per_type: int,
    seed: int = SEED,
    on_progress: Callable[[str], None] = print,
    user_id: uuid.UUID = ij.INJECT_USER,
) -> list[Case]:
    """Corrupt one stored statement at a time and keep the corruptions that make the
    checker report a new issue involving that statement."""
    rng = random.Random(seed)
    candidates: dict[str, list[tuple[str, Corruption]]] = defaultdict(list)
    for name, book_id in sorted(books.items()):
        for c in await _corruptions(factory, user_id, book_id):
            candidates[c.type].append((name, c))
    cases: list[Case] = []
    for kind in CORRUPTIONS:
        pool = candidates[kind]
        rng.shuffle(pool)
        used: Counter[str] = Counter()
        found = 0
        for name, corruption in pool:
            if found >= per_type:
                break
            if used[name] >= 2:  # spread over books
                continue
            fingerprint = await _produce(factory, user_id, books[name], corruption)
            if fingerprint is None:
                continue
            used[name] += 1
            found += 1
            cases.append(
                Case(
                    id=f"{name}/{kind}/{found}",
                    set="synthetic",
                    kind=f"synthetic:{kind}",
                    book=name,
                    user_id=user_id,
                    book_id=books[name],
                    expect=["false_alarm"],
                    reason=kind,
                    fingerprint=fingerprint,
                    corruption=corruption,
                )
            )
        on_progress(f"合成误报 {kind}：{found} 条（候选 {len(pool)}）")
    return cases


async def _corruptions(
    factory: Factory, user_id: uuid.UUID, book_id: uuid.UUID
) -> list[Corruption]:
    async with factory() as session:
        facts = (
            await session.scalars(
                select(FactRow).where(
                    FactRow.user_id == user_id,
                    FactRow.book_id == book_id,
                    FactRow.category == AGE.name,
                    FactRow.attribute == "absolute_age",
                    FactRow.value_num.is_not(None),
                )
            )
        ).all()
        spans = (
            await session.scalars(
                select(ElapsedTimeFactRow).where(
                    ElapsedTimeFactRow.user_id == user_id,
                    ElapsedTimeFactRow.book_id == book_id,
                    ElapsedTimeFactRow.kind.in_(["retrospective", "future"]),
                    ElapsedTimeFactRow.estimated_years >= 1,
                    ElapsedTimeFactRow.is_flashback.is_(False),
                )
            )
        ).all()
        characters = list(
            await session.scalars(
                select(Character.id).where(
                    Character.user_id == user_id, Character.book_id == book_id
                )
            )
        )

    def at(row: Any) -> dict[str, int]:
        return {"chapter": row.chapter_number, "start": row.char_start, "end": row.char_end}

    found: list[Corruption] = []
    present = [
        f for f in facts if not f.is_flashback and not f.is_speculative and f.value_num <= 120
    ]
    with_ages = {f.character_id for f in present}
    for f in facts:
        if f.is_flashback and not f.is_speculative:
            found.append(Corruption(type="flashback", table="fact", row_id=f.id,
                                    changes={"is_flashback": False}, **at(f)))  # fmt: skip
        if f.is_speculative and not f.is_flashback:
            found.append(Corruption(type="speculative", table="fact", row_id=f.id,
                                    changes={"is_speculative": False}, **at(f)))  # fmt: skip
    for f in present:
        if f.value_max is None or f.value_max == f.value_num:
            found.append(Corruption(type="value", table="fact", row_id=f.id,
                                    changes={"value_num": f.value_num + 12, "value_max": None},
                                    **at(f)))  # fmt: skip
        others = [c for c in characters if c != f.character_id and c in with_ages]
        if others:
            found.append(Corruption(type="misattributed", table="fact", row_id=f.id,
                                    changes={"character_id": str(others[0])}, **at(f)))  # fmt: skip
    for s in spans:
        found.append(Corruption(type="time_span", table="span", row_id=s.id,
                                changes={"kind": "advance"}, **at(s)))  # fmt: skip
    return found


async def _apply(session: AsyncSession, corruption: Corruption) -> None:
    model = FactRow if corruption.table == "fact" else ElapsedTimeFactRow
    changes = dict(corruption.changes)
    if "character_id" in changes:
        changes["character_id"] = uuid.UUID(changes["character_id"])
    await session.execute(update(model).where(model.id == corruption.row_id).values(**changes))


def _touches(evidence: list[dict[str, Any]], chapter: int, start: int, end: int) -> bool:
    return any(
        e["chapter_number"] == chapter and e["char_start"] < end and start < e["char_end"]
        for e in evidence
    )


async def _produce(
    factory: Factory, user_id: uuid.UUID, book_id: uuid.UUID, corruption: Corruption
) -> str | None:
    """The fingerprint of the new issue the corruption causes, if any (rolled back)."""
    async with rolled_back(factory) as scoped, scoped() as session:
        before = set(
            await session.scalars(
                select(IssueRow.fingerprint).where(
                    IssueRow.book_id == book_id, IssueRow.status == "open"
                )
            )
        )
        await _apply(session, corruption)
        await session.flush()
        await checks.run_checks(session, user_id=user_id, book_id=book_id)
        for issue in await session.scalars(
            select(IssueRow).where(IssueRow.book_id == book_id, IssueRow.status == "open")
        ):
            if issue.fingerprint not in before and _touches(
                issue.evidence, corruption.chapter, corruption.start, corruption.end
            ):
                return issue.fingerprint
    return None


class Label(BaseModel):
    version: str  # the extraction version tag of the base books ("3f88cac4")
    book: str
    fingerprint: str
    expect: list[str]
    reason: str | None = None
    note: str = ""


def load_labels(eval_dir: Path) -> list[Label]:
    path = eval_dir / LABELS_FILE
    if not path.exists():
        return []
    return [Label.model_validate(x) for x in yaml.safe_load(path.read_text("utf-8"))["issues"]]


async def real_cases(factory: Factory, labels: list[Label]) -> list[Case]:
    cases = []
    by_version: dict[str, dict[str, uuid.UUID]] = {}
    async with factory() as session:
        for label in labels:
            if label.version not in by_version:
                by_version[label.version] = await _books(
                    session, ij.INJECT_USER, "@" + label.version
                )
            book_id = by_version[label.version][label.book]
            cases.append(
                Case(
                    id=f"{label.book}@{label.version}/{label.fingerprint[:8]}",
                    set="real",
                    kind="real",
                    book=label.book,
                    user_id=ij.INJECT_USER,
                    book_id=book_id,
                    expect=label.expect,
                    reason=label.reason,
                    fingerprint=label.fingerprint,
                    note=label.note,
                )
            )
    return cases


# --- running ---------------------------------------------------------------------------------


async def _set_up(
    case: Case, scoped: Factory, extract_llm: LLMClient, settings: Settings, archival: Archival
) -> uuid.UUID | None:
    """Produce the case's issue inside the rolled-back transaction; its id."""
    if case.injection is not None:
        inj = case.injection
        result = await chapters.patch_chapter(
            scoped, extract_llm, settings, user_id=case.user_id, book_id=case.book_id,
            number=inj.chapter, old=inj.old, new=inj.new, archival=archival,
        )  # fmt: skip
        start, end = inj.span
        for issue in result.issues_added:
            first, last = issue.evidence[0], issue.evidence[-1]
            if (
                inj.ref is not None
                and first.chapter_number == inj.ref[0]
                and first.char_start < inj.ref[2]
                and inj.ref[1] < first.char_end
                and last.chapter_number == inj.chapter
                and last.char_start < end
                and start < last.char_end
            ):
                return issue.id
        return None
    async with scoped() as session:
        if case.corruption is not None:
            await _apply(session, case.corruption)
            await session.flush()
        if case.corruption is not None or case.kind == "golden":
            await checks.run_checks(session, user_id=case.user_id, book_id=case.book_id)
        return await session.scalar(
            select(IssueRow.id).where(
                IssueRow.user_id == case.user_id,
                IssueRow.book_id == case.book_id,
                IssueRow.fingerprint == case.fingerprint,
                IssueRow.status.in_(["open", "acknowledged"]),
            )
        )


async def run_case(
    case: Case,
    sample: int,
    factory: Factory,
    extract_llm: LLMClient,
    verify_llm: LLMClient,
    settings: Settings,
    archival: Archival | None,
    budget: Budget | None = None,
    keep_traces: bool = True,
    prompt_version: str | None = None,
) -> CaseResult:
    """Set the case's issue up and verify it, all rolled back; the execution record is
    kept (`keep_traces`; SQLite, in tests, cannot write it from a second connection
    while the case's transaction is open)."""
    started = asyncio.get_running_loop().time()
    async with rolled_back(factory) as scoped:
        issue_id = await _set_up(case, scoped, extract_llm, settings, archival)
        if issue_id is None:
            return CaseResult(case=case, sample=sample, set_up=False)
        result = await verification.verify_issues(
            scoped, verify_llm, settings, user_id=case.user_id, book_id=case.book_id,
            issue_ids=[issue_id], again=True, archival=archival, budget=budget,
            trace_factory=factory if keep_traces else scoped,
            **({"prompt_version": prompt_version} if prompt_version else {}),
        )  # fmt: skip
    outcome = result.verified[0]
    v = outcome.verification
    tools: Counter[str] = Counter()
    if v.run_id is not None and keep_traces:
        async with factory() as session:
            tools.update(
                await session.scalars(
                    select(AgentStepRow.name).where(
                        AgentStepRow.run_id == v.run_id, AgentStepRow.kind == "tool"
                    )
                )
            )
    return CaseResult(
        case=case, sample=sample, set_up=True, status=v.status, verdict=v.verdict,
        reason=v.reason, explanation=v.explanation, run_id=v.run_id, turns=outcome.turns,
        tool_calls=outcome.tool_calls, tools=dict(tools), cost_usd=result.cost_usd,
        seconds=round(asyncio.get_running_loop().time() - started, 1),
    )  # fmt: skip


async def index_books(
    factory: Factory, archival: Archival, cases: list[Case], on_progress: Callable[[str], None]
) -> None:
    """Passages for every book a case uses (built once; later runs skip them)."""
    for user_id, book_id in sorted({(c.user_id, c.book_id) for c in cases}, key=str):
        result = await reindex_book(factory, archival, user_id=user_id, book_id=book_id)
        if result.rebuilt:
            on_progress(f"建检索索引：{result.rebuilt} 章，{result.seconds} 秒")


async def run_all(
    cases: list[Case],
    *,
    samples: int,
    factory: Factory,
    make_llms: Callable[[Case], tuple[LLMClient, LLMClient]],
    settings: Settings,
    archival: Archival,
    concurrency: int = 4,
    budget: Budget | None = None,
    on_done: Callable[[CaseResult], None] | None = None,
    prompt_version: str | None = None,
) -> list[CaseResult]:
    """Books run concurrently, the cases of one book one after another (they would
    otherwise wait on each other's locks)."""
    gate = asyncio.Semaphore(concurrency)
    by_book: dict[uuid.UUID, list[Case]] = defaultdict(list)
    for case in cases:
        by_book[case.book_id].append(case)
    results: list[CaseResult] = []

    async def book(items: list[Case]) -> None:
        async with gate:
            for case in items:
                extract_llm, verify_llm = make_llms(case)
                for sample in range(samples):
                    r = await run_case(
                        case, sample, factory, extract_llm, verify_llm, settings, archival,
                        budget, prompt_version=prompt_version,
                    )  # fmt: skip
                    results.append(r)
                    if on_done:
                        on_done(r)

    await asyncio.gather(*(book(items) for items in by_book.values()))
    order = {c.id: i for i, c in enumerate(cases)}
    return sorted(results, key=lambda r: (order[r.case.id], r.sample))


# --- scoring ---------------------------------------------------------------------------------


class SetScore(BaseModel):
    set: str
    cases: int
    runs: int
    not_set_up: int
    right: Ratio  # verdict among the expected ones
    dismissed: Ratio  # judged false_alarm
    needs_author: Ratio
    unfinished: Ratio  # no verdict (budget, guard, provider)
    reason_right: Ratio  # dismissals whose reason is the expected one (synthetic)
    stable: Ratio  # cases whose verdict is the same in every sample
    turns: float
    tool_calls: float
    cost_per_run: Decimal
    by_kind: dict[str, Ratio] = {}  # right, per kind


def score(results: list[CaseResult]) -> list[SetScore]:
    out = []
    for name in ("keep", "synthetic", "real"):
        rs = [r for r in results if r.case.set == name]
        if not rs:
            continue
        done = [r for r in rs if r.set_up]
        by_case: dict[str, list[CaseResult]] = defaultdict(list)
        for r in done:
            by_case[r.case.id].append(r)
        dismissed = [r for r in done if r.verdict == "false_alarm"]
        with_reason = [r for r in dismissed if r.case.reason]
        kinds: dict[str, list[CaseResult]] = defaultdict(list)
        for r in done:
            kinds[r.case.kind].append(r)
        n = len(done) or 1
        out.append(
            SetScore(
                set=name,
                cases=len({r.case.id for r in rs}),
                runs=len(rs),
                not_set_up=sum(not r.set_up for r in rs),
                right=Ratio(num=sum(r.right for r in done), den=len(done)),
                dismissed=Ratio(num=len(dismissed), den=len(done)),
                needs_author=Ratio(
                    num=sum(r.verdict == "needs_author" for r in done), den=len(done)
                ),
                unfinished=Ratio(num=sum(r.verdict is None for r in done), den=len(done)),
                reason_right=Ratio(
                    num=sum(r.reason == r.case.reason for r in with_reason), den=len(with_reason)
                ),
                stable=Ratio(
                    num=sum(len({r.verdict for r in group}) == 1 for group in by_case.values()),
                    den=len(by_case),
                ),
                turns=round(sum(r.turns for r in done) / n, 2),
                tool_calls=round(sum(r.tool_calls for r in done) / n, 2),
                cost_per_run=(sum((r.cost_usd for r in done), Decimal(0)) / n).quantize(
                    Decimal("0.000001")
                ),
                by_kind={
                    k: Ratio(num=sum(r.right for r in group), den=len(group))
                    for k, group in sorted(kinds.items())
                },
            )
        )
    return out
