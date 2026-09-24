"""Run the real pipeline (the same services the CLI uses) on golden stories, each sample
in a throw-away SQLite database, and collect what it produced."""

import asyncio
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from webfic.config import Settings
from webfic.db.models import (
    Base,
    Character,
    CharacterAlias,
    ElapsedTimeFactRow,
    FactRow,
    LLMCallRow,
)
from webfic.db.session import make_engine, make_session_factory
from webfic.evaluation.golden import Story
from webfic.evaluation.matching import (
    ModelAge,
    ModelCharacter,
    ModelElapsed,
    ModelIssue,
    Observation,
    score_story,
)
from webfic.evaluation.metrics import SampleResult
from webfic.facts.registry import AGE
from webfic.llm.base import LLMClient
from webfic.llm.cache import DbCallStore
from webfic.llm.client import CallRecord, CallStore
from webfic.services import checks, imports, reports

EVAL_USER = uuid.UUID("00000000-0000-0000-0000-00000000e7a1")

Factory = async_sessionmaker[AsyncSession]
# Builds the LLM client for one sample, given the store that records its calls.
ClientFactory = Callable[[CallStore], LLMClient]


class EvalCallStore:
    """Records every call in the sample's own database (for its usage numbers) and in
    the shared eval cache. Lookups hit the cache unless the run is `fresh`."""

    def __init__(self, run: CallStore, cache: CallStore, *, fresh: bool):
        self._run, self._cache, self._fresh = run, cache, fresh

    async def lookup(self, request_hash: str) -> str | None:
        return None if self._fresh else await self._cache.lookup(request_hash)

    async def record(self, record: CallRecord) -> None:
        await self._run.record(record)
        if not record.cache_hit:
            await self._cache.record(record)


async def open_cache(path: Path) -> Factory:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent samples write here; wait for the lock instead of failing.
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{path.as_posix()}", connect_args={"timeout": 30}
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all, tables=[LLMCallRow.__table__])
    return make_session_factory(engine)


async def _observe(factory: Factory, book_id: uuid.UUID, dropped: int) -> Observation:
    async with factory() as session:
        chars = (await session.scalars(select(Character).where(Character.book_id == book_id))).all()
        aliases = (
            await session.scalars(select(CharacterAlias).where(CharacterAlias.book_id == book_id))
        ).all()
        ages = (
            await session.scalars(
                select(FactRow).where(FactRow.book_id == book_id, FactRow.category == AGE.name)
            )
        ).all()
        elapsed = (
            await session.scalars(
                select(ElapsedTimeFactRow).where(ElapsedTimeFactRow.book_id == book_id)
            )
        ).all()
        report = await reports.get_report(session, user_id=EVAL_USER, book_id=book_id)

    names: dict[str, set[str]] = {str(c.id): {c.canonical_name} for c in chars}
    for a in aliases:
        names[str(a.character_id)].add(a.alias)

    return Observation(
        characters=[ModelCharacter(cid, n) for cid, n in names.items()],
        ages=[
            ModelAge(
                character_id=str(a.character_id),
                chapter=a.chapter_number,
                start=a.char_start,
                end=a.char_end,
                type=a.attribute,
                value=a.value_num,
                life_stage=a.value_text,
                flashback=a.is_flashback,
                years_before_present=a.years_before_present,
                value_max=a.value_max,
                speculative=a.is_speculative,
            )
            for a in ages
        ],
        elapsed=[
            ModelElapsed(
                chapter=e.chapter_number,
                start=e.char_start,
                end=e.char_end,
                kind=e.kind,
                years=e.estimated_years,
                flashback=e.is_flashback,
            )
            for e in elapsed
        ],
        issues=[
            ModelIssue(
                subjects=i.subjects,
                chapters={e.chapter_number for e in i.evidence},
                confidence=i.confidence,
                description=i.description,
            )
            for i in report.issues
        ],
        dropped=dropped,
    )


async def run_sample(
    story: Story,
    settings: Settings,
    make_client: ClientFactory,
    cache: Factory,
    *,
    fresh: bool,
) -> SampleResult:
    started = time.monotonic()
    overrides = story.golden.settings.model_dump(exclude_none=True)
    settings = settings.model_copy(update=overrides)
    with tempfile.TemporaryDirectory(prefix="webfic-eval-") as tmp:
        engine = make_engine(f"sqlite+aiosqlite:///{Path(tmp, 'run.db').as_posix()}")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            factory = make_session_factory(engine)

            async with factory() as session:
                job = await imports.create_import_job(
                    session, user_id=EVAL_USER, title=story.golden.title, text=story.text
                )
            store = EvalCallStore(
                DbCallStore(factory, user_id=EVAL_USER, book_id=job.book_id),
                DbCallStore(cache, user_id=None),
                fresh=fresh,
            )
            result = await imports.run_import_job(
                factory, make_client(store), settings, user_id=EVAL_USER, book_id=job.book_id
            )
            async with factory() as session:
                await checks.run_checks(session, user_id=EVAL_USER, book_id=job.book_id)
            observation = await _observe(factory, job.book_id, result.dropped_statements)
        finally:
            await engine.dispose()

    return SampleResult(
        score=score_story(story, observation),
        cost_usd=result.cost_usd,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        llm_calls=result.llm_calls,
        cache_hits=result.cache_hits,
        seconds=time.monotonic() - started,
        failed_chapters=result.failed,
    )


async def run_story(
    story: Story,
    settings: Settings,
    make_client: ClientFactory,
    cache: Factory,
    *,
    samples: int,
    fresh: bool,
    concurrency: int = 3,
) -> list[SampleResult]:
    """Samples of one story run concurrently; each has its own database."""
    gate = asyncio.Semaphore(concurrency)

    async def one() -> SampleResult:
        async with gate:
            return await run_sample(story, settings, make_client, cache, fresh=fresh)

    return list(await asyncio.gather(*(one() for _ in range(samples))))
