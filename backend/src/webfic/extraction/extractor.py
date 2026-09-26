"""Run age extraction over one chapter: chunk, call the LLM, locate every statement in
the source text, and merge the results of overlapping chunks."""

import hashlib
import logging
import re
from collections import Counter
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


# "再过三年", "至少还要五年", "若能…三年之内": markers of time that has not come yet. An
# "advance" carrying one is a plan or a condition, not the story moving on. "N年后" alone
# is not a marker: narration uses it for real jumps ("三年后，他回到了故乡").
# "还有" and "至少" also occur in real advances ("三年过去了，他还有些不适应", "至少过去了
# 五年"), so they count only as "还有 + a number of years / months" and not before "过去 /
# 过了 / 已".
_FUTURE_MARKER = re.compile(
    r"再过|还要|还得|还有[^，。！？,!?]{0,4}?[年月]|(?:至少|少说)(?!也?(?:过去|过了|已))"
    r"|打算|约定|约好|倘若|若是|若能|如果|要是|以内|之内"
)

# "迟来十八年的长眠", "守了三十年": a number of years is a duration, not an age. Ages are
# written with 岁 or bare ("今年二十五", "年方二八", "小三十了"), never as "N年".
_DURATION = re.compile(r"[\d零〇一二两三四五六七八九十百几数]+年(?!纪|方|华|龄|岁)")


# "连一千岁都不到", "还没到三十岁": only an upper bound, so the age is unknown. ("未满十八
# 周岁" is different: a fixed phrase for seventeen.)
_NUM = r"[\d零〇一二两三四五六七八九十百千几]+"
_UPPER_BOUND_ONLY = re.compile(rf"(?:不到|没到|未到|不足){_NUM}岁|{_NUM}岁(?:都|也)?(?:不到|没到)")


def _valid_age(s: AgeStatement) -> bool:
    match s.statement_type:
        case "absolute_age":
            if "岁" not in s.raw_text and _DURATION.search(s.raw_text):
                return False
            if _UPPER_BOUND_ONLY.search(s.raw_text):
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
    # A statement is sometimes listed twice, and chunks overlap, so one near a cut is
    # read twice. Several characters can share one quote ("两人都是十八岁"), though: within
    # a chunk, statements on one quote count once per person named; a later chunk adds
    # only people beyond the most any earlier chunk had on that quote (it may name them
    # differently, "他" for "林远", so names are not compared across chunks).
    seen_ages: Counter[tuple[int, int, str]] = Counter()
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
        in_chunk: dict[tuple[int, int, str], set[str]] = {}

        for s in extraction.age_statements:
            span = locate(chunk.text, s.raw_text, start_from=last_end.get(s.raw_text, 0))
            if span is not None:
                last_end[s.raw_text] = span[1]
            if span is None or s.generic or not _valid_age(s):
                result.dropped.append(s.raw_text)
                continue
            start, end = span[0] + chunk.start, span[1] + chunk.start
            key = (start, end, s.statement_type)
            people = in_chunk.setdefault(key, set())
            person = (s.resolved_name or s.mention).strip()
            if person in people:  # listed twice
                continue
            people.add(person)
            if len(people) <= seen_ages[key]:  # read already in the overlap region
                continue
            result.ages.append(LocatedAge(_checked_offset(s, chunk.text), start, end))

        seen_ages |= Counter({key: len(people) for key, people in in_chunk.items()})
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
            if e.kind == "advance" and _FUTURE_MARKER.search(e.raw_text):
                e = e.model_copy(update={"kind": "future"})
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
