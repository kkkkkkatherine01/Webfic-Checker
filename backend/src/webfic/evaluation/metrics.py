"""Aggregate per-sample scores into the metrics of a run report."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, computed_field

from webfic.evaluation.matching import StoryScore


class Ratio(BaseModel):
    num: int = 0
    den: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float | None:
        return round(self.num / self.den, 4) if self.den else None

    def __add__(self, other: "Ratio") -> "Ratio":
        return Ratio(num=self.num + other.num, den=self.den + other.den)

    def __str__(self) -> str:
        return "—" if self.value is None else f"{self.value:.0%} ({self.num}/{self.den})"


@dataclass
class SampleResult:
    score: StoryScore
    cost_usd: Decimal
    input_tokens: int
    output_tokens: int
    llm_calls: int
    cache_hits: int
    seconds: float
    # Chapters the pipeline gave up on (content filter, invalid output...): a recall drop
    # with failed chapters is not a prompt regression.
    failed_chapters: int = 0


class Metrics(BaseModel):
    story: str
    samples: int = 0
    chars: int = 0
    # issue level
    issue_recall: Ratio = Ratio()
    issue_precision: Ratio = Ratio()
    confidence_accuracy: Ratio = Ratio()
    insufficient: int = 0
    issue_hits: dict[str, int] = {}  # "story/issue-id" -> samples in which it was found
    false_positives: dict[str, int] = {}  # description -> count
    # extraction level
    age_recall: Ratio = Ratio()
    age_accuracy: Ratio = Ratio()  # of found ages, fully correct
    elapsed_recall: Ratio = Ratio()
    elapsed_kind_accuracy: Ratio = Ratio()
    trap_pass: Ratio = Ratio()
    merges: int = 0
    splits: int = 0
    dropped: int = 0
    fact_errors: dict[str, int] = {}
    # cost
    cost_usd: Decimal = Decimal(0)
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    seconds: float = 0.0
    failed_chapters: int = 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_per_1k_chars(self) -> Decimal | None:
        processed = self.chars * self.samples
        return (
            (self.cost_usd * 1000 / processed).quantize(Decimal("0.000001")) if processed else None
        )

    def unstable_issues(self) -> dict[str, int]:
        """Expected issues found in some samples but not all (or in none)."""
        return {k: v for k, v in self.issue_hits.items() if v < self.samples}


def aggregate_story(story: str, chars: int, samples: list[SampleResult]) -> Metrics:
    m = Metrics(story=story, samples=len(samples), chars=chars)
    hits: Counter[str] = Counter()
    fps: Counter[str] = Counter()
    errors: Counter[str] = Counter()

    for s in samples:
        i, f = s.score.issues, s.score.facts
        matched = len(i.matched)
        m.issue_recall += Ratio(num=matched, den=i.expected)
        m.issue_precision += Ratio(num=matched, den=matched + len(i.false_positives))
        m.confidence_accuracy += Ratio(num=sum(i.matched.values()), den=matched)
        m.insufficient += i.insufficient
        for issue_id in [*i.matched, *i.missed]:
            hits[f"{story}/{issue_id}"] += issue_id in i.matched
        fps.update(i.false_positives)
        errors.update(i.confidence_errors)

        m.age_recall += Ratio(num=f.ages_found, den=f.ages_expected)
        m.age_accuracy += Ratio(num=f.ages_correct, den=f.ages_found)
        m.elapsed_recall += Ratio(num=f.elapsed_found, den=f.elapsed_expected)
        m.elapsed_kind_accuracy += Ratio(num=f.elapsed_kind_correct, den=f.elapsed_found)
        m.trap_pass += Ratio(num=f.traps_passed, den=f.traps)
        m.merges += len(f.merges)
        m.splits += len(f.splits)
        m.dropped += f.dropped
        errors.update([*f.age_errors, *f.elapsed_errors, *f.trap_errors, *f.merges, *f.splits])

        m.cost_usd += s.cost_usd
        m.input_tokens += s.input_tokens
        m.output_tokens += s.output_tokens
        m.llm_calls += s.llm_calls
        m.cache_hits += s.cache_hits
        m.seconds += s.seconds
        m.failed_chapters += s.failed_chapters

    m.issue_hits = dict(hits)
    m.false_positives = dict(fps)
    m.fact_errors = dict(errors)
    return m


_SUMMED = (
    "issue_recall",
    "issue_precision",
    "confidence_accuracy",
    "insufficient",
    "age_recall",
    "age_accuracy",
    "elapsed_recall",
    "elapsed_kind_accuracy",
    "trap_pass",
    "merges",
    "splits",
    "dropped",
    "cost_usd",
    "input_tokens",
    "output_tokens",
    "llm_calls",
    "cache_hits",
    "seconds",
    "failed_chapters",
)


def combine(parts: list[Metrics], name: str = "全部") -> Metrics:
    """All stories in a run use the same number of samples."""
    total = Metrics(story=name, samples=parts[0].samples if parts else 0)
    for p in parts:
        for key in _SUMMED:
            setattr(total, key, getattr(total, key) + getattr(p, key))
        total.chars += p.chars
        total.issue_hits |= p.issue_hits
        total.false_positives |= {f"{p.story} {k}": v for k, v in p.false_positives.items()}
        total.fact_errors |= {f"{p.story} {k}": v for k, v in p.fact_errors.items()}
    return total


class RunMeta(BaseModel):
    label: str
    created_at: datetime
    provider: str
    extract_model: str
    prompt_hash: str
    git_commit: str | None
    samples: int
    fresh: bool  # True = cache bypassed
    stories: list[str]


class RunReport(BaseModel):
    meta: RunMeta
    overall: Metrics
    stories: list[Metrics]
