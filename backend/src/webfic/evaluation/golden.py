"""Load and validate the golden set: eval/golden/storyNN/{text.txt, expected.yaml}."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from webfic.checkers.types import Confidence, IssueType
from webfic.extraction.locator import locate
from webfic.extraction.schemas import LifeStage, StatementType
from webfic.ingest.splitter import RawChapter, split_chapters


class _IssueSpec(BaseModel):
    """`chapters` names the chapters of the issue's two ends: [1, 3], or [2] when both
    ends are in chapter 2. When the earlier end legitimately depends on extraction,
    list alternatives: [[1, 5], [2, 5]]."""

    character: str
    chapters: list[int] | list[list[int]] = Field(min_length=1)
    note: str = ""
    # The kind of issue (webfic.checkers.types.IssueType); ages by default. A kinship
    # issue names either of the two characters.
    type: str = IssueType.CHARACTER_AGE.value

    def endpoint_options(self) -> set[tuple[int, int]]:
        options = self.chapters if isinstance(self.chapters[0], list) else [self.chapters]
        return {(min(o), max(o)) for o in options}  # type: ignore[arg-type]

    def all_chapters(self) -> set[int]:
        return {c for pair in self.endpoint_options() for c in pair}


class ExpectedIssue(_IssueSpec):
    id: str
    confidence: Confidence | None = None


class AllowedIssue(_IssueSpec):
    pass


class ExpectedAge(BaseModel):
    chapter: int
    quote: str
    character: str
    type: StatementType
    value: float | None = None
    value_max: float | None = None  # approximate age range
    life_stage: LifeStage | None = None
    flashback: bool = False
    years_before_present: float | None = None
    no_years_before_present: bool = False  # the text gives no offset: must stay empty
    speculative: bool = False


class ExpectedElapsed(BaseModel):
    chapter: int
    quote: str
    kind: Literal["advance", "short", "retrospective", "future"]
    years: float | None = None
    flashback: bool = False


class Trap(BaseModel):
    chapter: int
    quote: str
    note: str = ""


class ExpectedTrait(BaseModel):
    """A fixed feature of appearance (step 5-1)."""

    chapter: int
    quote: str
    character: str
    attribute: Literal["eye_color", "hair_color", "mark"]
    value: str  # a colour word, or "部位·类型" for a mark
    flashback: bool = False
    speculative: bool = False
    disguised: bool = False


class ExpectedKinship(BaseModel):
    """`character` is the `relation` of `other` (webfic.facts.kinship.RELATIONS); the
    same fact stated the other way round counts too."""

    chapter: int
    quote: str
    character: str
    other: str
    relation: str


class ExpectedDeath(BaseModel):
    chapter: int
    quote: str
    character: str
    speculative: bool = False  # rumour, presumed dead


class ExpectedFacts(BaseModel):
    ages: list[ExpectedAge] = []
    elapsed: list[ExpectedElapsed] = []
    not_ages: list[Trap] = []  # must not be extracted as any age statement
    not_elapsed: list[Trap] = []  # must not be extracted as an "advance" time span
    # Character facts (step 5-1).
    traits: list[ExpectedTrait] = []
    kinship: list[ExpectedKinship] = []
    deaths: list[ExpectedDeath] = []
    not_traits: list[Trap] = []  # no comparable feature here (disguise, guess, a group...)
    not_kinship: list[Trap] = []  # no blood relation here (politeness, sworn kin...)
    not_deaths: list[Trap] = []  # no certain death here (rumour, feigned death...)
    not_presence: list[Trap] = []  # nobody appears in person here (dream, memory...)


class RetrievalQuestion(BaseModel):
    """A question the agent might ask, and where the text answers it; finding any one of
    the expected passages counts."""

    query: str
    character: str | None = None  # search only passages mentioning this character
    expect: list[Trap] = Field(min_length=1)


class SettingsOverride(BaseModel):
    """Per-story pipeline settings, e.g. a small chunk size so a short test chapter
    still exercises chunked extraction."""

    chunk_size: int | None = None
    chunk_overlap: int | None = None


class Golden(BaseModel):
    title: str
    settings: SettingsOverride = SettingsOverride()
    characters: dict[str, list[str]]  # standard name -> other names
    issues: list[ExpectedIssue] = []
    allowed: list[AllowedIssue] = []
    # Reports the rules cannot avoid but the text explains (a revealed parentage, a
    # dye): not the checker's false alarms; the verify agent should dismiss them.
    dismiss: list[AllowedIssue] = []
    not_aliases: list[str] = []  # must not be any character's name or alias
    facts: ExpectedFacts = ExpectedFacts()
    retrieval: list[RetrievalQuestion] = []

    @model_validator(mode="after")
    def _names_are_unique_and_known(self) -> "Golden":
        seen: dict[str, str] = {}
        for name, others in self.characters.items():
            for n in [name, *(others or [])]:
                if n in seen and seen[n] != name:
                    raise ValueError(f"称呼「{n}」同时属于「{seen[n]}」和「{name}」")
                seen[n] = name
        refs = [i.character for i in [*self.issues, *self.allowed, *self.dismiss]]
        refs += [a.character for a in self.facts.ages]
        refs += [t.character for t in self.facts.traits]
        refs += [k.character for k in self.facts.kinship] + [k.other for k in self.facts.kinship]
        refs += [d.character for d in self.facts.deaths]
        kinds = {t.value for t in IssueType}
        for issue in [*self.issues, *self.allowed, *self.dismiss]:
            if issue.type not in kinds:
                raise ValueError(f"未知的矛盾类型：{issue.type}")
        refs += [q.character for q in self.retrieval if q.character]
        for ref in refs:
            if ref not in self.characters:
                raise ValueError(f"「{ref}」不在 characters 里")
        return self

    def all_names(self, character: str) -> set[str]:
        return {character, *(self.characters.get(character) or [])}


@dataclass(frozen=True)
class Span:
    chapter: int
    start: int
    end: int

    def overlaps(self, chapter: int, start: int, end: int) -> bool:
        return self.chapter == chapter and self.start < end and start < self.end


@dataclass
class Story:
    id: str  # directory name, e.g. "story01"
    text: str
    chapters: list[RawChapter]
    golden: Golden

    def span(self, chapter: int, quote: str) -> Span:
        """Where a golden quote sits, in the same coordinates as stored facts."""
        content = self.chapters[chapter - 1].content
        found = locate(content, quote)
        if found is None:  # validated at load time
            raise KeyError((chapter, quote))
        return Span(chapter, *found)


class GoldenError(Exception):
    pass


def load_story(directory: Path) -> Story:
    text_path, expected_path = directory / "text.txt", directory / "expected.yaml"
    if not text_path.exists() or not expected_path.exists():
        raise GoldenError(f"{directory.name}: 缺少 text.txt 或 expected.yaml")

    text = text_path.read_text("utf-8-sig")
    try:
        golden = Golden.model_validate(yaml.safe_load(expected_path.read_text("utf-8")))
    except (ValidationError, yaml.YAMLError) as exc:
        raise GoldenError(f"{directory.name}: expected.yaml 格式错误\n{exc}") from exc

    story = Story(directory.name, text, split_chapters(text).chapters, golden)
    _check_quotes(story)
    return story


def _check_quotes(story: Story) -> None:
    g = story.golden
    problems: list[str] = []
    n = len(story.chapters)

    f = g.facts
    quoted = [(a.chapter, a.quote) for a in f.ages]
    quoted += [(e.chapter, e.quote) for e in f.elapsed]
    quoted += [(t.chapter, t.quote) for t in [*f.not_ages, *f.not_elapsed]]
    quoted += [(x.chapter, x.quote) for x in [*f.traits, *f.kinship, *f.deaths]]
    quoted += [
        (t.chapter, t.quote)
        for t in [*f.not_traits, *f.not_kinship, *f.not_deaths, *f.not_presence]
    ]
    quoted += [(e.chapter, e.quote) for q in g.retrieval for e in q.expect]
    for chapter, quote in quoted:
        if not 1 <= chapter <= n:
            problems.append(f"第 {chapter} 章不存在（共 {n} 章）：{quote}")
        elif locate(story.chapters[chapter - 1].content, quote) is None:
            problems.append(f"第 {chapter} 章找不到引文：{quote}")

    for issue in [*g.issues, *g.allowed, *g.dismiss]:
        problems += [
            f"矛盾引用了不存在的第 {c} 章" for c in issue.all_chapters() if not 1 <= c <= n
        ]

    if problems:
        raise GoldenError(f"{story.id}:\n  " + "\n  ".join(problems))


def unconverted(directory: Path) -> bool:
    """A story written with its answers in free form (`answers.txt`) and not yet turned
    into expected.yaml: a holdout still being written or not yet converted (step 5-1).
    Skipped without reading it."""
    return (directory / "answers.txt").exists() and not (directory / "expected.yaml").exists()


def discover(golden_dir: Path, only: list[str] | None = None) -> list[Path]:
    dirs = sorted(
        p
        for p in golden_dir.iterdir()
        if p.is_dir() and p.name.startswith("story") and not unconverted(p)
    )
    if only:
        wanted = {o if o.startswith("story") else f"story{o.zfill(2)}" for o in only}
        dirs = [d for d in dirs if d.name in wanted]
    return dirs
