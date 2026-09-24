"""Run age extraction over one chapter: chunk, call the LLM, locate every statement in
the source text, and merge the results of overlapping chunks."""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from importlib import resources
from typing import Any

from webfic.extraction.locator import locate
from webfic.extraction.schemas import (
    AGE_SCHEMA_VERSION,
    SHORT_SPAN_YEARS,
    AgeExtraction,
    AgeStatement,
    ElapsedTimeStatement,
    RevealedName,
)
from webfic.ingest.chunker import Chunk, chunk_text
from webfic.llm.base import LLMClient, Tier, Usage

log = logging.getLogger(__name__)

# Covers long-lived beings in xianxia; anything beyond is almost surely an error.
_MAX_PLAUSIBLE_AGE = 100_000


def load_prompt(name: str = AGE_SCHEMA_VERSION) -> str:
    return resources.files("webfic.extraction.prompts").joinpath(f"{name}.txt").read_text("utf-8")


@dataclass
class LocatedAge:
    statement: AgeStatement
    char_start: int  # chapter coordinates
    char_end: int


@dataclass
class LocatedElapsed:
    statement: ElapsedTimeStatement
    char_start: int
    char_end: int


@dataclass
class ChapterExtraction:
    ages: list[LocatedAge] = field(default_factory=list)
    elapsed: list[LocatedElapsed] = field(default_factory=list)
    revealed_names: list[RevealedName] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # raw_texts that failed validation
    usage: Usage = field(default_factory=Usage)
    cost_usd: Decimal = Decimal(0)
    llm_calls: int = 0
    cache_hits: int = 0


def extraction_version(system_prompt: str, chunk_size: int, chunk_overlap: int) -> str:
    """Identifies everything besides the chapter text that shapes an extraction; a stored
    extraction is only reused while this stays the same."""
    key = f"{AGE_SCHEMA_VERSION}|{chunk_size}|{chunk_overlap}|{system_prompt}"
    return hashlib.sha256(key.encode()).hexdigest()[:32]


def dump_extraction(extraction: ChapterExtraction) -> dict[str, Any]:
    """The reusable part of an extraction (not its cost), as JSON."""
    return {
        "ages": [
            {
                "statement": a.statement.model_dump(mode="json"),
                "start": a.char_start,
                "end": a.char_end,
            }
            for a in extraction.ages
        ],
        "elapsed": [
            {
                "statement": e.statement.model_dump(mode="json"),
                "start": e.char_start,
                "end": e.char_end,
            }
            for e in extraction.elapsed
        ],
        "revealed_names": [r.model_dump(mode="json") for r in extraction.revealed_names],
        "dropped": list(extraction.dropped),
    }


def load_extraction(data: dict[str, Any]) -> ChapterExtraction:
    """A stored extraction; it made no model calls this time."""
    return ChapterExtraction(
        ages=[
            LocatedAge(AgeStatement.model_validate(a["statement"]), a["start"], a["end"])
            for a in data["ages"]
        ],
        elapsed=[
            LocatedElapsed(
                ElapsedTimeStatement.model_validate(e["statement"]), e["start"], e["end"]
            )
            for e in data["elapsed"]
        ],
        revealed_names=[RevealedName.model_validate(r) for r in data["revealed_names"]],
        dropped=list(data["dropped"]),
    )


def build_user_message(
    *, known_characters: str, chapter_number: int, chunk: Chunk, total_chunks: int
) -> str:
    part = f"，第 {chunk.index + 1}/{total_chunks} 段" if total_chunks > 1 else ""
    return f"已知角色：\n{known_characters}\n\n正文（第 {chapter_number} 章{part}）：\n{chunk.text}"


# "迟来十八年的长眠", "守了三十年": a number of years is a duration, not an age. Ages are
# written with 岁 or bare ("今年二十五", "年方二八", "小三十了"), never as "N年".
_DURATION = re.compile(r"[\d零〇一二两三四五六七八九十百几数]+年(?!纪|方|华|龄|岁)")


def _valid_age(s: AgeStatement) -> bool:
    match s.statement_type:
        case "absolute_age":
            if "岁" not in s.raw_text and _DURATION.search(s.raw_text):
                return False
            return s.value is not None and 0 <= s.value <= _MAX_PLAUSIBLE_AGE
        case "life_stage":
            return s.life_stage is not None
        case _:
            return True


def _checked_offset(s: AgeStatement, text: str) -> AgeStatement:
    """A flashback's distance from the present must be backed by text that states it
    ("十五年前"); otherwise it is the model's own inference, and wrong inferences produce
    false contradictions. Such offsets are cleared, which leaves the fact out of
    comparisons instead of comparing it on a guess."""
    if s.years_before_present is None:
        return s
    quote = (s.years_before_present_quote or "").strip()
    if quote and locate(text, quote) is not None:
        return s
    return s.model_copy(update={"years_before_present": None})


async def extract_chapter(
    llm: LLMClient,
    *,
    chapter_number: int,
    text: str,
    known_characters: str,
    system_prompt: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ChapterExtraction:
    result = ChapterExtraction()
    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
    seen_ages: set[tuple[int, int, str]] = set()
    seen_elapsed: set[tuple[int, int]] = set()

    for chunk in chunks:
        llm_result = await llm.generate_json(
            tier=Tier.EXTRACT,
            system=system_prompt,
            user=build_user_message(
                known_characters=known_characters,
                chapter_number=chapter_number,
                chunk=chunk,
                total_chunks=len(chunks),
            ),
            schema=AgeExtraction,
            purpose="extract.age",
        )
        result.usage += llm_result.usage
        result.cost_usd += llm_result.cost_usd
        result.llm_calls += 1
        result.cache_hits += int(llm_result.cache_hit)
        extraction = llm_result.data
        # Models list statements in text order; searching after the previous hit of the
        # same quote keeps two identical quotes ("十八岁" twice) from collapsing into one.
        last_end: dict[str, int] = {}

        for s in extraction.age_statements:
            span = locate(chunk.text, s.raw_text, start_from=last_end.get(s.raw_text, 0))
            if span is not None:
                last_end[s.raw_text] = span[1]
            if span is None or not _valid_age(s):
                result.dropped.append(s.raw_text)
                continue
            start, end = span[0] + chunk.start, span[1] + chunk.start
            key = (start, end, s.statement_type)
            if key in seen_ages:  # same statement seen again in the overlap region
                continue
            seen_ages.add(key)
            result.ages.append(LocatedAge(_checked_offset(s, chunk.text), start, end))

        last_end.clear()
        for e in extraction.elapsed_time_statements:
            span = locate(chunk.text, e.raw_text, start_from=last_end.get(e.raw_text, 0))
            if span is not None:
                last_end[e.raw_text] = span[1]
            if span is None:
                result.dropped.append(e.raw_text)
                continue
            start, end = span[0] + chunk.start, span[1] + chunk.start
            if (start, end) in seen_elapsed:
                continue
            seen_elapsed.add((start, end))
            if (
                e.kind == "advance"
                and e.estimated_years is not None
                and e.estimated_years < SHORT_SPAN_YEARS
            ):
                e = e.model_copy(update={"kind": "short"})
            result.elapsed.append(LocatedElapsed(e, start, end))

        for revealed in extraction.revealed_names:
            # The real name must actually appear in the text, like every other claim.
            if locate(chunk.text, revealed.real_name) is None:
                result.dropped.append(revealed.real_name)
            elif revealed not in result.revealed_names:
                result.revealed_names.append(revealed)

    if result.dropped:
        log.info("chapter %s: dropped %d statements", chapter_number, len(result.dropped))
    result.ages.sort(key=lambda a: a.char_start)
    result.elapsed.sort(key=lambda e: e.char_start)
    return result
