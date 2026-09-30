"""Probe real text before adding a kind of check (step 1 of the procedure in
docs/steps/05-more-checkers.md A7): which facts about characters do real web novels
state, how often is the same fact of the same character stated twice or more (a fact
stated once can never contradict), and which pairs already read differently.

An exploratory prompt lets the model sort the facts itself; nothing here is stored in
the product's tables. Quotes that cannot be located in the chapter are dropped, as in
extraction.
"""

import asyncio
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib import resources
from typing import Literal

from pydantic import BaseModel

from webfic.extraction.locator import locate, normalized
from webfic.ingest.chunker import chunk_text
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import LLMClient, LLMError, Tier

PROMPT_VERSION = "probe_facts_v1"
CATEGORIES = ["外貌", "亲属", "生死", "境界", "身份", "出身", "门派", "能力", "称号", "其他"]
SEED = 5
CHUNK = 8000


class ProbeFact(BaseModel):
    character: str
    category: str
    attribute: str = ""
    value: str = ""
    quote: str
    status: Literal["present", "past", "guess"] = "present"


class ProbeReading(BaseModel):
    facts: list[ProbeFact] = []


@dataclass(frozen=True)
class Sample:
    book: str  # "webnovelbench/017", "shushan", "private/势同水火"
    chapter: int
    text: str


class Found(BaseModel):
    book: str
    chapter: int
    fact: ProbeFact


def load_prompt() -> str:
    return (
        resources.files("webfic.evaluation.prompts")
        .joinpath(f"{PROMPT_VERSION}.txt")
        .read_text("utf-8")
    )


def choose(
    webnovelbench: list[tuple[str, str]],
    shushan: str,
    private: dict[str, str],
    *,
    books: int = 8,
    private_chapters: int = 20,
    seed: int = SEED,
) -> list[Sample]:
    """`books` WebNovelBench excerpts (all their chapters, by a fixed seed), all of
    《蜀山》 we have, and the first `private_chapters` chapters of each private work."""
    rng = random.Random(seed)
    picked = rng.sample(sorted(webnovelbench), books)
    samples = []
    for name, text in [*picked, ("shushan", shushan)]:
        samples += [Sample(name, c.number, c.content) for c in split_chapters(text).chapters]
    for name, text in sorted(private.items()):
        chapters = split_chapters(text).chapters[:private_chapters]
        samples += [Sample(name, c.number, c.content) for c in chapters]
    return samples


async def read(llm: LLMClient, samples: list[Sample], *, concurrency: int = 6) -> list[Found]:
    """Every sample through the exploratory prompt; located facts only."""
    system = load_prompt()
    gate = asyncio.Semaphore(concurrency)
    found: list[Found] = []

    async def one(sample: Sample) -> None:
        async with gate:
            for chunk in chunk_text(sample.text, size=CHUNK, overlap=0):
                try:
                    result = await llm.generate_json(
                        tier=Tier.EXTRACT, system=system, user=f"正文：\n{chunk.text}",
                        schema=ProbeReading, purpose="probe.facts",
                    )  # fmt: skip
                except LLMError:
                    continue
                for fact in result.data.facts:
                    if fact.quote.strip() and locate(chunk.text, fact.quote) is not None:
                        found.append(Found(book=sample.book, chapter=sample.chapter, fact=fact))

    await asyncio.gather(*(one(s) for s in samples))
    found.sort(key=lambda f: (f.book, f.chapter))
    return found


class CategoryStats(BaseModel):
    category: str
    facts: int
    present: int  # status present
    characters: int  # distinct (book, character)
    groups: int  # distinct (book, character, attribute) with a present fact
    repeated: int  # of those, stated in two or more chapters
    differing: int  # of those, with two or more different values
    examples: list[str]  # differing groups, for a person to look at


def stats(found: list[Found], *, examples: int = 12) -> list[CategoryStats]:
    out = []
    for category in CATEGORIES:
        facts = [f for f in found if _category(f.fact.category) == category]
        present = [f for f in facts if f.fact.status == "present"]
        groups: dict[tuple[str, str, str], list[Found]] = defaultdict(list)
        for f in present:
            groups[(f.book, f.fact.character, f.fact.attribute)].append(f)
        repeated = {k: v for k, v in groups.items() if len({f.chapter for f in v}) >= 2}
        differing = {
            k: v for k, v in repeated.items() if len({normalized(f.fact.value) for f in v}) >= 2
        }
        shown = [
            f"{book} {character}·{attribute}："
            + "；".join(f"第{f.chapter}章「{f.fact.value}」" for f in _firsts(v))
            for (book, character, attribute), v in list(differing.items())[:examples]
        ]
        out.append(
            CategoryStats(
                category=category, facts=len(facts), present=len(present),
                characters=len({(f.book, f.fact.character) for f in facts}),
                groups=len(groups), repeated=len(repeated), differing=len(differing),
                examples=shown,
            )
        )  # fmt: skip
    return out


def attributes(found: list[Found], top: int = 40) -> list[tuple[str, str, int]]:
    """The most common (category, attribute) pairs."""
    counts = Counter((_category(f.fact.category), f.fact.attribute) for f in found)
    return [(c, a, n) for (c, a), n in counts.most_common(top)]


def by_book(found: list[Found]) -> dict[str, Counter[str]]:
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for f in found:
        out[f.book][_category(f.fact.category)] += 1
    return dict(out)


def _category(name: str) -> str:
    return name if name in CATEGORIES else "其他"


def _firsts(group: list[Found]) -> list[Found]:
    """One fact per distinct value, in order."""
    seen, out = set(), []
    for f in sorted(group, key=lambda f: f.chapter):
        key = normalized(f.fact.value)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
