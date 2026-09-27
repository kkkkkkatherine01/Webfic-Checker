"""Author's-note evaluation (step 4.5).

Real notes are rare in the evaluation corpora (about one chapter in a thousand of
WebNovelBench has a marked one), so recall is measured on notes written for this
project, attached to the start or end of real chapters; story text that talks to "you",
readers or a crowd is attached the same way and must stay story. How often real story
paragraphs are taken for notes is measured on the unmodified chapters.
"""

import asyncio
import random
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from webfic.evaluation.metrics import Ratio
from webfic.evaluation.realtext import webnovelbench_books
from webfic.extraction.author_notes import find_author_notes, paragraphs, regions
from webfic.ingest.splitter import split_chapters
from webfic.llm.base import LLMClient

SEED = 20260927
SNIPPETS = Path("author_notes") / "snippets.yaml"  # under eval/


class Snippet(BaseModel):
    id: str
    text: str
    marker: str | None = None


class NoteCase(BaseModel):
    id: str
    kind: Literal["note", "story"]
    snippet: str
    marker: str | None  # the author's registered marker, if the case uses it
    chapter: str  # "webnovelbench/017#3"
    position: Literal["head", "tail"]
    content: str
    span: tuple[int, int]  # the attached text in `content`


class NoteResult(BaseModel):
    case: NoteCase
    ranges: list[tuple[int, int]]

    @property
    def found(self) -> bool:
        """The attached text lies entirely inside the recognised notes."""
        a, b = self.case.span
        return any(x <= a and b <= y for x, y in self.ranges)

    @property
    def touched(self) -> bool:
        a, b = self.case.span
        return any(x < b and a < y for x, y in self.ranges)

    @property
    def story_lost(self) -> int:
        """Characters of the real chapter taken for notes."""
        a, b = self.case.span
        lost = 0
        for x, y in self.ranges:
            lost += max(0, min(y, a) - x) + max(0, y - max(x, b))
        return lost


def load_snippets(eval_dir: Path) -> tuple[list[Snippet], list[Snippet]]:
    data = yaml.safe_load((eval_dir / SNIPPETS).read_text("utf-8"))
    return (
        [Snippet.model_validate(x) for x in data["notes"]],
        [Snippet.model_validate(x) for x in data["stories"]],
    )


def _chapters(external: Path) -> list[tuple[str, str]]:
    books = webnovelbench_books(external / "webnovelbench" / "novel_data_subset_d_100.json")
    return [
        (f"{b.name}#{c.number}", c.content) for b in books for c in split_chapters(b.text).chapters
    ]


def build_cases(eval_dir: Path, seed: int = SEED) -> list[NoteCase]:
    notes, stories = load_snippets(eval_dir)
    chapters = _chapters(eval_dir / "external")
    rng = random.Random(seed)
    cases = []

    def attach(kind: str, s: Snippet, marker: str | None, suffix: str = "") -> None:
        name, chapter = rng.choice(chapters)
        text = s.text.replace("\n\n", "\n")
        position = rng.choice(["head", "tail"])
        if position == "head":
            content, span = text + "\n" + chapter, (0, len(text))
        else:
            content = chapter + "\n" + text
            span = (len(chapter) + 1, len(content))
        cases.append(
            NoteCase(
                id=f"{s.id}{suffix}",
                kind=kind,
                snippet=s.id,
                marker=marker,
                chapter=name,
                position=position,
                content=content,
                span=span,
            )
        )

    for s in notes:
        attach("note", s, None)
        if s.marker:  # the same note once more, with the author's marker registered
            attach("note", s, s.marker, "+marker")
    for s in stories:
        attach("story", s, None)
    return cases


async def run_cases(
    llm: LLMClient, cases: list[NoteCase], concurrency: int = 8
) -> list[NoteResult]:
    gate = asyncio.Semaphore(concurrency)

    async def one(case: NoteCase) -> NoteResult:
        async with gate:
            scan = await find_author_notes(llm, case.content, [case.marker] if case.marker else [])
        return NoteResult(case=case, ranges=scan.ranges)

    return list(await asyncio.gather(*(one(c) for c in cases)))


class Flagged(BaseModel):
    chapter: str
    range: tuple[int, int]
    text: str


class OriginalsScore(BaseModel):
    chapters: int
    paragraphs_examined: int  # paragraphs in the chapters' first and last ~1500 characters
    paragraphs_flagged: int
    chapters_flagged: int
    flagged: list[Flagged]


async def run_originals(
    llm: LLMClient,
    external: Path,
    concurrency: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> OriginalsScore:
    chapters = _chapters(external)
    gate = asyncio.Semaphore(concurrency)
    examined = flagged_paragraphs = 0
    flagged: list[Flagged] = []
    done = 0

    async def one(name: str, content: str) -> None:
        nonlocal examined, flagged_paragraphs, done
        async with gate:
            scan = await find_author_notes(llm, content)
        head, tail = regions(paragraphs(content), len(content))
        examined += len(head) + len(tail)
        for a, b in scan.ranges:
            flagged_paragraphs += sum(1 for p in paragraphs(content) if a <= p.start < b)
            flagged.append(Flagged(chapter=name, range=(a, b), text=content[a:b]))
        done += 1
        if on_progress and done % 100 == 0:
            on_progress(done, len(chapters))

    await asyncio.gather(*(one(n, c) for n, c in chapters))
    return OriginalsScore(
        chapters=len(chapters), paragraphs_examined=examined,
        paragraphs_flagged=flagged_paragraphs,
        chapters_flagged=len({f.chapter for f in flagged}),
        flagged=sorted(flagged, key=lambda f: f.chapter),
    )  # fmt: skip


class NotesScore(BaseModel):
    group: str
    cases: int
    found: Ratio  # notes: recognised whole
    exact: Ratio  # notes: recognised and no story taken with them
    touched: Ratio  # stories: wrongly (partly) taken for notes


GROUPS = {
    "m": "常见标记",
    "c": "自定义标记（未登记）",
    "c+": "自定义标记（已登记）",
    "u": "无标记",
    "x": "多段",
    "s": "短句",
    "h": "像对读者说话的正文",
}


def score(results: list[NoteResult]) -> list[NotesScore]:
    def group(r: NoteResult) -> str:
        g = r.case.snippet[0]
        return "c+" if g == "c" and r.case.marker else g

    out = []
    for key, label in GROUPS.items():
        rs = [r for r in results if group(r) == key]
        if not rs:
            continue
        notes = [r for r in rs if r.case.kind == "note"]
        stories = [r for r in rs if r.case.kind == "story"]
        out.append(
            NotesScore(
                group=label,
                cases=len(rs),
                found=Ratio(num=sum(r.found for r in notes), den=len(notes)),
                exact=Ratio(num=sum(r.found and r.story_lost == 0 for r in notes), den=len(notes)),
                touched=Ratio(num=sum(r.touched for r in stories), den=len(stories)),
            )
        )
    return out
