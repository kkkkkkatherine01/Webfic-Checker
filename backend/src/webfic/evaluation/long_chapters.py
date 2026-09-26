"""Long chapters (step 4-3): the same novels with every three chapters merged into one
(about 9,000–12,000 characters, as in a short work of two or three long chapters).

Two uses:
- extraction on long chapters, which spec had never tested on real text: chapters over
  the 8,000-character chunk size are read in overlapping pieces. The same text read as
  short chapters (the injection bases) and as long ones should give the same ages
- the verify agent on long chapters: injections and synthetic false alarms are made on
  these books like on the originals (verify_eval), and results compared by length
"""

import random
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.db.models import Chapter, Character, FactRow
from webfic.evaluation import inject as ij
from webfic.evaluation.realtext import webnovelbench_books
from webfic.facts.registry import AGE
from webfic.ingest.splitter import split_chapters

Factory = async_sessionmaker[AsyncSession]

SEED = 20260927
GROUP = 3  # chapters merged into one; a remainder joins the last group
PREFIX = "long/"


@dataclass(frozen=True)
class LongBase:
    base: ij.Base  # the merged text, named "long/NNN"
    original: str  # "webnovelbench/NNN"
    # For each original chapter (1-based, in order): (long chapter, offset of its text in it)
    placement: list[tuple[int, int]]


def long_bases(external: Path, count: int = 20, seed: int = SEED) -> list[LongBase]:
    books = webnovelbench_books(external / "webnovelbench" / "novel_data_subset_d_100.json")
    rng = random.Random(seed)
    picked = sorted(rng.sample(books, min(count, len(books))), key=lambda b: b.name)
    out = []
    for book in picked:
        chapters = split_chapters(book.text).chapters
        groups = [chapters[i : i + GROUP] for i in range(0, len(chapters), GROUP)]
        if len(groups) > 1 and len(groups[-1]) < GROUP:
            rest = groups.pop()  # not `groups[-2] += groups.pop()`: that writes to the wrong slot
            groups[-1] = groups[-1] + rest
        lines, placement = [], []
        for n, group in enumerate(groups, start=1):
            lines.append(f"第{n}章 合并（原第{group[0].number}至{group[-1].number}章）")
            offset = 0
            for chapter in group:
                placement.append((n, offset))
                offset += len(chapter.content) + 1  # joined by one newline
            lines.append("\n".join(c.content for c in group))
        name = PREFIX + book.name.removeprefix("webnovelbench/")
        out.append(LongBase(ij.Base(name, "\n".join(lines)), book.name, placement))
    return out


# --- extraction: long vs original ----------------------------------------------------------


class Reading(BaseModel):
    """One extracted age, placed in the long book's coordinates."""

    chapter: int
    start: int
    end: int
    quote: str
    character: str
    attribute: str
    value: float | None
    flashback: bool
    speculative: bool


class BookComparison(BaseModel):
    name: str
    long_chapter_chars: list[int]
    original: int  # ages extracted from the short chapters
    long: int  # ages extracted from the long ones
    both: int  # the same quote extracted in both
    same_reading: int  # ... with the same character, type, value, flashback, guess labels
    # The same, for stated ages only (numbers), leaving out life stages ("少年", "老者"):
    original_ages: int = 0
    long_ages: int = 0
    both_ages: int = 0
    same_ages: int = 0


async def _readings(session: AsyncSession, book_id: uuid.UUID) -> list[tuple[int, Reading]]:
    rows = (
        await session.execute(
            select(FactRow, Character.canonical_name)
            .join(Character, Character.id == FactRow.character_id)
            .where(FactRow.book_id == book_id, FactRow.category == AGE.name)
        )
    ).all()
    return [
        (
            f.chapter_number,
            Reading(
                chapter=f.chapter_number,
                start=f.char_start,
                end=f.char_end,
                quote=f.raw_text,
                character=name,
                attribute=f.attribute,
                value=f.value_num,
                flashback=f.is_flashback,
                speculative=f.is_speculative,
            ),
        )
        for f, name in rows
    ]


async def compare(
    factory: Factory, long: LongBase, long_id: uuid.UUID, original_id: uuid.UUID
) -> BookComparison:
    async with factory() as session:
        short = await _readings(session, original_id)
        extracted = [r for _, r in await _readings(session, long_id)]
        sizes = list(
            await session.scalars(
                select(Chapter.char_count)
                .where(Chapter.book_id == long_id)
                .order_by(Chapter.number)
            )
        )
    moved = []
    for chapter, r in short:
        target, offset = long.placement[chapter - 1]
        moved.append(r.model_copy(update={
            "chapter": target, "start": r.start + offset, "end": r.end + offset,
        }))  # fmt: skip

    def key(r: Reading) -> tuple[int, int, int]:
        return (r.chapter, r.start, r.end)

    def reading(r: Reading) -> tuple:
        return (r.character, r.attribute, r.value, r.flashback, r.speculative)

    def match(ours: list[Reading], theirs: list[Reading]) -> tuple[int, int]:
        """(found in both, same reading): by quote position in the same long chapter."""
        unmatched = Counter(key(r) for r in theirs)
        by_key = {key(r): r for r in theirs}
        both = same = 0
        for r in ours:
            hit = next(
                (k for k in unmatched if unmatched[k] and k[0] == r.chapter
                 and k[1] < r.end and r.start < k[2]),
                None,
            )  # fmt: skip
            if hit is None:
                continue
            unmatched[hit] -= 1
            both += 1
            same += reading(by_key[hit]) == reading(r)
        return both, same

    both, same = match(moved, extracted)
    ages_short = [r for r in moved if r.attribute == "absolute_age"]
    ages_long = [r for r in extracted if r.attribute == "absolute_age"]
    both_ages, same_ages = match(ages_short, ages_long)
    return BookComparison(
        name=long.base.name, long_chapter_chars=sizes, original=len(moved), long=len(extracted),
        both=both, same_reading=same, original_ages=len(ages_short), long_ages=len(ages_long),
        both_ages=both_ages, same_ages=same_ages,
    )  # fmt: skip
