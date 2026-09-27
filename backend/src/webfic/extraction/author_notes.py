"""Author's notes: what an author writes to readers at the start or end of a chapter
(update notices, thanks, requests for votes, explanations of the setting, previews),
which must not be read as the story.

Authors mark them in their own ways — "作者有话说", "PS", "Note", or not at all — so a
fixed list of markers cannot find them all. The model labels each paragraph of the
chapter's first and last ~1,500 characters as story or author (a label, not an
omission); explicit markers, built in or registered by the author for the book, are
recognised in code. Code also keeps notes where they are in practice: a run of
paragraphs from the very start or up to the very end. A note in the middle is never
recognised; a missed note costs a wrong fact, a story paragraph taken for a note costs
the facts in it, so in doubt a paragraph is story.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from typing import Literal

from pydantic import BaseModel

from webfic.llm.base import LLMClient, LLMError, Tier

log = logging.getLogger(__name__)

PROMPT_VERSION = "author_notes_v1"
REGION = 1500  # characters examined at each end of a chapter
MAX_PARAGRAPH = 400  # characters of each paragraph shown to the model

# Explicit markers at the start of a paragraph: "作者有话说：", "【作者的话】", "PS:"...
_WORDS = r"(?:作者有话要说|作者有话说|作者的话|作者注|作者按|作者留言|碎碎念|题外话|p\.?\s?s\.?)"
_MARKER = re.compile(
    # In brackets it stands on its own ("【作者的话】下周…"); bare, it needs a colon, a
    # space or the end of the line after it ("PS：", but not "PSP游戏机").
    rf"^\s*(?:[【\[（(]\s*{_WORDS}\s*[】\]）)]|{_WORDS}\s*(?:[:：]|$|\s))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Paragraph:
    index: int
    start: int
    end: int
    text: str


def paragraphs(content: str) -> list[Paragraph]:
    """Non-empty lines, with their offsets in the chapter."""
    found, start = [], 0
    for line in content.split("\n"):
        if line.strip():
            found.append(Paragraph(len(found), start, start + len(line), line))
        start += len(line) + 1
    return found


def regions(paras: list[Paragraph], length: int) -> tuple[list[Paragraph], list[Paragraph]]:
    """The paragraphs starting within REGION of the chapter's start, and those ending
    within REGION of its end. A chapter shorter than twice REGION is split into halves by
    paragraph count (the middle one going to the end), so that a note at its end is
    looked for as one at the end even when the story before it is only a line or two."""
    if length < 2 * REGION:
        half = len(paras) // 2
        return paras[:half], paras[half:]
    head = [p for p in paras if p.start < REGION]
    tail = [p for p in paras if p.end > length - REGION]
    return head, tail


class _Label(BaseModel):
    i: int
    who: Literal["story", "author"]


class NoteLabels(BaseModel):
    paragraphs: list[_Label] = []


class NoteScan(BaseModel):
    """Where a chapter's author's notes are: [start, end) ranges, at most one at the
    start and one at the end."""

    ranges: list[tuple[int, int]] = []
    model_used: bool = False  # False if only markers were used (no model, or it failed)
    llm_calls: int = 0
    cost_usd: Decimal = Decimal(0)

    def story(self, content: str) -> tuple[int, int]:
        """The [start, end) of the story text between the notes."""
        start, end = 0, len(content)
        for a, b in self.ranges:
            if a == 0:
                start = max(start, b)
            else:
                end = min(end, a)
        return start, max(start, end)


def version(markers: list[str]) -> str:
    """Identifies what shapes a scan besides the chapter text: the prompt and the book's
    own markers. A stored scan is reused while this stays the same."""
    key = PROMPT_VERSION + "|" + "|".join(sorted(m.strip() for m in markers if m.strip()))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def load_prompt(name: str = PROMPT_VERSION) -> str:
    return resources.files("webfic.extraction.prompts").joinpath(f"{name}.txt").read_text("utf-8")


def _marked(p: Paragraph, markers: list[str]) -> bool:
    text = p.text.strip()
    return bool(_MARKER.match(text)) or any(
        m.strip() and text.lower().startswith(m.strip().lower()) for m in markers
    )


async def find_author_notes(
    llm: LLMClient | None, content: str, markers: list[str] = ()
) -> NoteScan:
    """Scan one chapter. Without `llm`, or if the model fails, only markers are used."""
    paras = paragraphs(content)
    if not paras:
        return NoteScan()
    head, tail = regions(paras, len(content))
    shown = head + tail
    author: set[int] = set()
    scan = NoteScan()
    if llm is not None:
        numbered = {n: p for n, p in enumerate(shown)}
        user = "章节开头：\n" + "\n".join(
            f"[{n}] {p.text.strip()[:MAX_PARAGRAPH]}" for n, p in numbered.items() if p in head
        )
        if tail:
            user += "\n章节结尾：\n" + "\n".join(
                f"[{n}] {p.text.strip()[:MAX_PARAGRAPH]}" for n, p in numbered.items() if p in tail
            )
        try:
            result = await llm.generate_json(
                tier=Tier.EXTRACT, system=load_prompt(), user=user, schema=NoteLabels,
                purpose="extract.author_notes",
            )  # fmt: skip
        except LLMError as exc:
            log.warning("author-note scan fell back to markers: %s", exc)
        else:
            scan.model_used = True
            scan.llm_calls, scan.cost_usd = 1, result.cost_usd
            author = {numbered[x.i].index for x in result.data.paragraphs
                      if x.who == "author" and x.i in numbered}  # fmt: skip

    def note(p: Paragraph) -> bool:
        return p.index in author or _marked(p, list(markers))

    # At the start: paragraphs taken for notes, from the first one on.
    head_end = None
    for p in head:
        if not note(p):
            break
        head_end = p.end
    # At the end: from a marker on, everything is the note; else the run of paragraphs
    # taken for notes that reaches the last one.
    tail_start = None
    marked_at = next((p for p in tail if _marked(p, list(markers))), None)
    if marked_at is not None:
        tail_start = marked_at.start
    for p in reversed(tail):
        if p.index not in author and not _marked(p, list(markers)):
            break
        tail_start = p.start if tail_start is None else min(tail_start, p.start)
    if head_end is not None:
        scan.ranges.append((0, head_end))
    if tail_start is not None and (head_end is None or tail_start >= head_end):
        scan.ranges.append((tail_start, len(content)))
    return scan
