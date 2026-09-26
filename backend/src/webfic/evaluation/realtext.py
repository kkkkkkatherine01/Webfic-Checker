"""Extraction quality on real web-novel text (step 3.5-1).

Twenty chapters are picked with a fixed seed and annotated completely (every age and
time span), so extraction can be scored for precision as well as recall. Chapters are
extracted in reading order together with the chapters before them, as in real use, and
only the picked chapters are scored.
"""

import asyncio
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from webfic.config import Settings
from webfic.evaluation.golden import Span
from webfic.evaluation.matching import ModelAge, ModelCharacter, ModelElapsed, Observation
from webfic.evaluation.metrics import Ratio
from webfic.evaluation.runner import ClientFactory, Factory, import_and_observe
from webfic.extraction.locator import locate
from webfic.extraction.schemas import LifeStage, StatementType
from webfic.ingest.splitter import RawChapter, split_chapters

SEED = 20260925
DENSE, RANDOM, SHUSHAN = 14, 4, 2
FIRST_PICKABLE = 4  # earlier chapters give the extractor some characters to know

_NUM = r"[\d零〇一二两三四五六七八九十百几]+"
AGE_MENTION = re.compile(_NUM + r"岁")


@dataclass(frozen=True)
class Book:
    name: str  # "webnovelbench/017", "shushan"
    title: str
    text: str


@dataclass(frozen=True)
class Pick:
    book: str
    title: str
    chapter: int  # chapter number as the import splits the book
    kind: str  # "dense" / "random" / "shushan"
    age_mentions: int


def webnovelbench_books(path: Path) -> list[Book]:
    """The WebNovelBench excerpts (10 consecutive chapters each), leaving out those the
    dataset itself split wrongly (our split finds a different number of chapters)."""
    books = []
    for i, novel in enumerate(json.loads(path.read_text("utf-8"))):
        text = "\n".join(novel["chapters"])
        if len(split_chapters(text).chapters) == len(novel["chapters"]):
            books.append(Book(f"webnovelbench/{i:03d}", novel["novel"], text))
    return books


def select(webnovelbench: list[Book], shushan: Book, seed: int = SEED) -> list[Pick]:
    rng = random.Random(seed)
    chapters = {b.name: split_chapters(b.text).chapters for b in webnovelbench}
    titles = {b.name: b.title for b in webnovelbench}

    def mentions(content: str) -> int:
        return len(AGE_MENTION.findall(content))

    dense_by_book = {
        name: [c for c in chs if c.number >= FIRST_PICKABLE and mentions(c.content) >= 2]
        for name, chs in chapters.items()
    }
    dense_books = rng.sample(sorted(n for n, cs in dense_by_book.items() if cs), DENSE)
    picks = []
    for name in dense_books:
        c = rng.choice(dense_by_book[name])
        picks.append(Pick(name, titles[name], c.number, "dense", mentions(c.content)))
    others = sorted(set(chapters) - set(dense_books))
    for name in rng.sample(others, RANDOM):
        c = rng.choice([c for c in chapters[name] if c.number >= FIRST_PICKABLE])
        picks.append(Pick(name, titles[name], c.number, "random", mentions(c.content)))
    ranked = sorted(
        split_chapters(shushan.text).chapters, key=lambda c: (-mentions(c.content), c.number)
    )
    for c in sorted(ranked[:SHUSHAN], key=lambda c: c.number):
        picks.append(Pick(shushan.name, shushan.title, c.number, "shushan", mentions(c.content)))
    return picks


def export(picks: list[Pick], books: dict[str, Book], folder: Path) -> None:
    """Write the selection and each picked chapter (with the chapters before it, for the
    annotator's context) for annotation."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "selection.json").write_text(
        json.dumps(
            {"seed": SEED, "picks": [asdict(p) for p in picks]}, ensure_ascii=False, indent=1
        ),
        "utf-8",
    )
    for n, pick in enumerate(picks, start=1):
        chapters = split_chapters(books[pick.book].text).chapters
        target = chapters[pick.chapter - 1]
        before = "\n\n".join(f"【{c.title}】\n{c.content}" for c in chapters[: pick.chapter - 1])
        name = f"{n:02d}-{pick.kind}-{pick.book.replace('/', '_')}-ch{pick.chapter}"
        (folder / f"{name}.txt").write_text(f"【{target.title}】\n{target.content}\n", "utf-8")
        (folder / f"{name}.context.txt").write_text(before + "\n", "utf-8")


def shushan_book(path: Path) -> Book:
    return Book("shushan", "蜀山剑侠传", path.read_text("utf-8"))


def load_picks(path: Path) -> list[Pick]:
    return [Pick(**p) for p in json.loads(path.read_text("utf-8"))["picks"]]


# --- annotations -----------------------------------------------------------------------

TOLERANCE = 0.01
Either = Literal["any"]  # both values count as right


class AnnotatedAge(BaseModel):
    quote: str
    character: str
    type: StatementType
    value: float | None = None
    value_max: float | None = None
    approx: bool = False  # any range overlapping value(–value_max) counts
    life_stage: LifeStage | None = None
    flashback: bool | Either = False
    years_before_present: float | None = None
    years_quote: str | None = None
    no_years_before_present: bool = False
    speculative: bool | Either = False
    note: str = ""


class AnnotatedElapsed(BaseModel):
    quote: str
    kind: Literal["advance", "short", "retrospective", "future"]
    years: float | None = None
    note: str = ""


class Labelled(BaseModel):
    """An acceptable entry or a trap: only its position matters."""

    model_config = {"extra": "allow"}

    quote: str
    note: str = ""


class AnnotatedChapter(BaseModel):
    id: str
    book: str
    chapter: int
    characters: dict[str, list[str]]
    ages: list[AnnotatedAge] = []
    acceptable: list[Labelled] = []
    elapsed: list[AnnotatedElapsed] = []
    not_ages: list[Labelled] = []
    not_elapsed: list[Labelled] = []

    @model_validator(mode="after")
    def _characters_known(self) -> "AnnotatedChapter":
        seen: dict[str, str] = {}
        for name, others in self.characters.items():
            for n in [name, *others]:
                if seen.setdefault(n, name) != name:
                    raise ValueError(f"{self.id}：称呼「{n}」同时属于「{seen[n]}」和「{name}」")
        for age in self.ages:
            if age.character not in self.characters:
                raise ValueError(f"{self.id}：「{age.character}」不在 characters 里")
        return self

    def names(self, character: str) -> set[str]:
        return {character, *self.characters[character]}

    def quotes(self) -> list[str]:
        entries = [*self.ages, *self.acceptable, *self.elapsed, *self.not_ages, *self.not_elapsed]
        return [e.quote for e in entries] + [a.years_quote for a in self.ages if a.years_quote]


class AnnotationError(Exception):
    pass


def load_annotations(path: Path) -> list[AnnotatedChapter]:
    try:
        data = yaml.safe_load(path.read_text("utf-8"))
        return [AnnotatedChapter.model_validate(c) for c in data["chapters"]]
    except (yaml.YAMLError, ValueError, KeyError) as exc:
        raise AnnotationError(f"{path.name} 格式错误\n{exc}") from exc


# --- what to run -----------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One annotated chapter, and where it sits in the text a run imports."""

    annotation: AnnotatedChapter
    kind: str  # "dense" / "random" / "shushan"
    chapter: int  # chapter number within the run's text
    content: str  # that chapter's content, as the pipeline stores it


@dataclass(frozen=True)
class Job:
    """One import: a book up to its last annotated chapter (context mode), or a lone
    annotated chapter (single mode)."""

    name: str
    text: str
    targets: tuple[Target, ...]


def _rebuild(chapters: list[RawChapter]) -> str:
    return "\n".join(f"{c.title}\n{c.content}" for c in chapters)


def _check_same_split(text: str, expected: list[RawChapter]) -> None:
    got = split_chapters(text).chapters
    if [c.content for c in got] != [c.content for c in expected]:
        raise AnnotationError("截取后重新切章与原书不一致")


def _target(annotation: AnnotatedChapter, kind: str, chapter: RawChapter, number: int) -> Target:
    missing = [q for q in annotation.quotes() if locate(chapter.content, q) is None]
    if missing:
        raise AnnotationError(f"{annotation.id}：原文中找不到 {missing}")
    return Target(annotation, kind, number, chapter.content)


def plan_jobs(
    annotations: list[AnnotatedChapter], picks: list[Pick], books: dict[str, Book], mode: str
) -> list[Job]:
    """Context mode: each book from chapter 1 up to its last annotated chapter, extracted
    in order as in real use. Single mode: each annotated chapter imported on its own."""
    if len(annotations) != len(picks):
        raise AnnotationError(f"标注 {len(annotations)} 章，选章 {len(picks)} 章")
    by_book: dict[str, list[tuple[AnnotatedChapter, Pick]]] = {}
    for ann, pick in zip(annotations, picks, strict=True):
        if (ann.book, ann.chapter) != (pick.book, pick.chapter):
            raise AnnotationError(f"{ann.id}：标注的是 {ann.book} 第 {ann.chapter} 章，与选章不符")
        by_book.setdefault(pick.book, []).append((ann, pick))

    jobs = []
    for book, items in by_book.items():
        chapters = split_chapters(books[book].text).chapters
        if mode == "single":
            for ann, pick in items:
                chapter = chapters[pick.chapter - 1]
                text = _rebuild([chapter])
                _check_same_split(text, [chapter])
                jobs.append(Job(f"{ann.id}-single", text, (_target(ann, pick.kind, chapter, 1),)))
        else:
            last = max(p.chapter for _, p in items)
            text = _rebuild(chapters[:last])
            _check_same_split(text, chapters[:last])
            targets = tuple(
                _target(ann, p.kind, chapters[p.chapter - 1], p.chapter) for ann, p in items
            )
            jobs.append(Job(book, text, targets))
    return jobs


# --- scoring one chapter of one run ----------------------------------------------------
#
# Time spans only matter to the age check when they advance story time by a number of
# years, so they are scored on that: required spans found, "advance" or not judged right,
# advances that should not be there, and future / planned durations taken as spans. Short
# and retrospective spans are not annotated exhaustively (step 5 will need that).


@dataclass
class ChapterScore:
    id: str
    kind: str
    ages_expected: int = 0
    ages_found: int = 0
    ages_correct: int = 0
    ages_extracted: int = 0
    ages_extracted_ok: int = 0  # on a required or acceptable quote
    elapsed_expected: int = 0
    elapsed_found: int = 0
    elapsed_correct: int = 0  # advance or not, and the years of an advance
    elapsed_extracted: int = 0
    elapsed_extracted_ok: int = 0  # reference only: spans are not annotated exhaustively
    advances: int = 0
    advances_ok: int = 0  # on a required or acceptable quote
    future: int = 0  # annotated future / planned durations (not_elapsed)
    future_taken: int = 0  # ... extracted as a time span labelled anything but "future"
    traps: int = 0
    traps_passed: int = 0
    merges: int = 0  # one model character = several annotated ones
    splits: int = 0  # one annotated character = several model ones
    errors: list[str] = field(default_factory=list)
    # required entry -> kinds of error in its extraction (None: missed); compared across
    # samples: an entry is stable when every sample judged it the same way
    verdicts: dict[str, tuple[str, ...] | None] = field(default_factory=dict)
    # which quote every age extraction sits on; compared across samples
    age_keys: set[str] = field(default_factory=set)


def _spans(target: Target, quotes: list[str]) -> list[Span]:
    spans = []
    for quote in quotes:
        found = locate(target.content, quote)
        assert found is not None  # checked when the target was built
        spans.append(Span(target.chapter, *found))
    return spans


def _on(span: Span, items: list) -> list:
    return [x for x in items if span.overlaps(x.chapter, x.start, x.end)]


def _value_ok(exp: AnnotatedAge, hit: ModelAge) -> bool:
    if exp.value is None:
        return True
    if hit.value is None:
        return False
    lo, hi = hit.value, hit.value if hit.value_max is None else hit.value_max
    if exp.value_max is None and not exp.approx:
        return abs(lo - exp.value) <= TOLERANCE and abs(hi - exp.value) <= TOLERANCE
    exp_hi = exp.value if exp.value_max is None else exp.value_max
    return lo <= exp_hi + TOLERANCE and exp.value <= hi + TOLERANCE


def _flag_ok(expected: bool | Either, got: bool) -> bool:
    return expected == "any" or expected == got


def _fmt(lo: float | None, hi: float | None) -> str:
    if lo is None:
        return "空"
    return f"{lo:g}" if hi is None or abs(hi - lo) <= TOLERANCE else f"{lo:g}–{hi:g}"


def _age_problems(exp: AnnotatedAge, hit: ModelAge, standard: set[str]) -> list[str]:
    """Each problem starts with its kind ("角色", "数值"...), then details."""
    wrong = []
    if standard != {exp.character}:
        merged = f"（与{'、'.join(sorted(standard - {exp.character}))}合并）"
        wrong.append("角色" + (merged if exp.character in standard else ""))
    if hit.type != exp.type:
        wrong.append(f"类型 {hit.type}≠{exp.type}")
    if not _value_ok(exp, hit):
        wrong.append(f"数值 {_fmt(hit.value, hit.value_max)}≠{_fmt(exp.value, exp.value_max)}")
    if exp.life_stage is not None and hit.life_stage != exp.life_stage:
        wrong.append(f"人生阶段 {hit.life_stage}≠{exp.life_stage}")
    if not _flag_ok(exp.flashback, hit.flashback):
        wrong.append("回忆标记")
    if not _flag_ok(exp.speculative, hit.speculative):
        wrong.append("推测标记")
    if exp.years_before_present is not None and (
        hit.years_before_present is None
        or abs(hit.years_before_present - exp.years_before_present) > TOLERANCE
    ):
        wrong.append(f"距今年数 {hit.years_before_present}≠{exp.years_before_present:g}")
    if exp.no_years_before_present and hit.years_before_present is not None:
        wrong.append(f"距今年数应为空，实际 {hit.years_before_present:g}")
    return wrong


def _elapsed_problems(exp: AnnotatedElapsed, hit: ModelElapsed) -> list[str]:
    if (exp.kind == "advance") != (hit.kind == "advance"):
        return [f"推进判断 {hit.kind}≠{exp.kind}"]
    years_wrong = hit.years is None or abs(hit.years - (exp.years or 0)) > TOLERANCE
    if exp.kind == "advance" and exp.years is not None and years_wrong:
        return [f"年数 {hit.years}≠{exp.years:g}"]
    return []


def _kinds(problems: list[str]) -> tuple[str, ...]:
    return tuple(sorted(p.split(" ")[0].split("（")[0] for p in problems))


def _standard_names(ann: AnnotatedChapter, characters: list[ModelCharacter]) -> dict[str, set[str]]:
    """Model character id -> annotated characters it corresponds to."""
    return {c.id: {std for std in ann.characters if c.names & ann.names(std)} for c in characters}


def score_chapter(target: Target, obs: Observation) -> ChapterScore:
    ann = target.annotation
    s = ChapterScore(ann.id, target.kind)
    ages = [a for a in obs.ages if a.chapter == target.chapter]
    elapsed = [e for e in obs.elapsed if e.chapter == target.chapter]
    standard = _standard_names(ann, obs.characters)
    names = {c.id: c.names for c in obs.characters}

    def snippet(x: ModelAge | ModelElapsed) -> str:
        return target.content[x.start : x.end]

    for cid, stds in standard.items():
        if len(stds) > 1:
            s.merges += 1
            model_names = "、".join(sorted(names[cid]))
            s.errors.append(f"{ann.id} 角色合并：「{model_names}」实际是 {'、'.join(sorted(stds))}")
    for std in ann.characters:
        ids = [cid for cid, stds in standard.items() if std in stds]
        if len(ids) > 1:
            s.splits += 1
            s.errors.append(f"{ann.id} 角色拆分：{std} 被拆成 {len(ids)} 个角色")

    age_spans = _spans(target, [a.quote for a in ann.ages])
    s.ages_expected = len(ann.ages)
    for exp, span in zip(ann.ages, age_spans, strict=True):
        key = f"{ann.id}/年龄/{exp.quote}"
        hits = _on(span, ages)
        if not hits:
            s.errors.append(f"{ann.id} 漏抽年龄：{exp.quote}")
            s.verdicts[key] = None
            continue
        s.ages_found += 1

        def problems(a: ModelAge) -> list[str]:
            return _age_problems(exp, a, standard.get(a.character_id, set()))  # noqa: B023

        wrong = min((problems(a) for a in hits), key=len)
        if wrong:
            s.errors.append(f"{ann.id} 年龄「{exp.quote}」：{'、'.join(wrong)}错误")
        else:
            s.ages_correct += 1
        s.verdicts[key] = _kinds(wrong)

    elapsed_spans = _spans(target, [e.quote for e in ann.elapsed])
    s.elapsed_expected = len(ann.elapsed)
    for exp_e, span in zip(ann.elapsed, elapsed_spans, strict=True):
        key = f"{ann.id}/时间段/{exp_e.quote}"
        hits_e = _on(span, elapsed)
        if not hits_e:
            s.errors.append(f"{ann.id} 漏抽时间段：{exp_e.quote}")
            s.verdicts[key] = None
            continue
        s.elapsed_found += 1
        wrong = min((_elapsed_problems(exp_e, e) for e in hits_e), key=len)
        if wrong:
            s.errors.append(f"{ann.id} 时间段「{exp_e.quote}」：{'、'.join(wrong)}错误")
        else:
            s.elapsed_correct += 1
        s.verdicts[key] = _kinds(wrong)

    acceptable = _spans(target, [a.quote for a in ann.acceptable])
    not_ages = _spans(target, [t.quote for t in ann.not_ages])
    not_elapsed = _spans(target, [t.quote for t in ann.not_elapsed])

    # Ages: every extraction must sit on a required or acceptable quote.
    labelled = [("必抽", age_spans), ("可接受", acceptable), ("陷阱", not_ages)]
    s.ages_extracted = len(ages)
    for a in ages:
        where = next(
            (f"{ann.id}/{label}{i}" for label, spans in labelled
             for i, sp in enumerate(spans) if sp.overlaps(a.chapter, a.start, a.end)),
            None,
        )  # fmt: skip
        s.age_keys.add(where or f"{ann.id}/其他/{snippet(a)}")
        if where and "陷阱" not in where:
            s.ages_extracted_ok += 1
        else:
            s.errors.append(f"{ann.id} 误抽年龄：{snippet(a)}")

    # Time spans: advances must sit on a required advance or an acceptable quote.
    elapsed_ok = elapsed_spans + acceptable
    advance_ok = [
        sp for sp, exp_e in zip(elapsed_spans, ann.elapsed, strict=True) if exp_e.kind == "advance"
    ] + acceptable
    s.elapsed_extracted = len(elapsed)
    for e in elapsed:
        s.elapsed_extracted_ok += any(sp.overlaps(e.chapter, e.start, e.end) for sp in elapsed_ok)
        if e.kind == "advance":
            ok = any(sp.overlaps(e.chapter, e.start, e.end) for sp in advance_ok)
            s.advances += 1
            s.advances_ok += ok
            if not ok:
                s.errors.append(f"{ann.id} 误抽为推进：{snippet(e)}（{e.years} 年）")
    s.future = len(not_elapsed)
    for span in not_elapsed:
        # Since prompt v4 a "future" label is the right answer, not a mistake.
        taken = [e for e in _on(span, elapsed) if e.kind != "future"]
        if taken:
            s.future_taken += 1
            if all(e.kind != "advance" for e in taken):  # advances are reported above
                kinds = "、".join(sorted({e.kind for e in taken}))
                s.errors.append(f"{ann.id} 将来时长标错（{kinds}）：{snippet(taken[0])}")

    # Traps, as in the golden set: any age, or an "advance" time span.
    s.traps = len(not_ages) + len(not_elapsed)
    s.traps_passed = sum(not _on(sp, ages) for sp in not_ages) + sum(
        not _on(sp, [e for e in elapsed if e.kind == "advance"]) for sp in not_elapsed
    )
    return s


# --- aggregating samples ---------------------------------------------------------------


class RealtextMetrics(BaseModel):
    name: str  # chapter id, a group ("dense"...) or "全部"
    samples: int = 0
    age_recall: Ratio = Ratio()
    age_precision: Ratio = Ratio()
    age_accuracy: Ratio = Ratio()  # of found, every attribute right
    elapsed_recall: Ratio = Ratio()
    elapsed_accuracy: Ratio = Ratio()  # of found, advance or not (and years) right
    elapsed_precision: Ratio = Ratio()  # reference only (spans not annotated exhaustively)
    advance_precision: Ratio = Ratio()  # advances on a required or acceptable quote
    future_taken: Ratio = Ratio()  # future / planned durations taken as other time spans
    trap_pass: Ratio = Ratio()
    merges: int = 0
    splits: int = 0
    # consistency across samples
    found_in: dict[int, int] = {}  # required entries found in k of the samples
    stable: Ratio = Ratio()  # of entries found at least once: judged the same every time
    age_consistency: Ratio = Ratio()  # age extractions (by position) present in every sample
    errors: dict[str, int] = {}


def aggregate(name: str, samples: list[list[ChapterScore]]) -> RealtextMetrics:
    """`samples[i]` holds sample i's scores of the chapters in this group."""
    m = RealtextMetrics(name=name, samples=len(samples))
    errors: Counter[str] = Counter()
    verdicts: dict[str, list[tuple[str, ...] | None]] = {}
    age_keys: dict[str, list[set[str]]] = {}
    for scores in samples:
        for s in scores:
            m.age_recall += Ratio(num=s.ages_found, den=s.ages_expected)
            m.age_precision += Ratio(num=s.ages_extracted_ok, den=s.ages_extracted)
            m.age_accuracy += Ratio(num=s.ages_correct, den=s.ages_found)
            m.elapsed_recall += Ratio(num=s.elapsed_found, den=s.elapsed_expected)
            m.elapsed_accuracy += Ratio(num=s.elapsed_correct, den=s.elapsed_found)
            m.elapsed_precision += Ratio(num=s.elapsed_extracted_ok, den=s.elapsed_extracted)
            m.advance_precision += Ratio(num=s.advances_ok, den=s.advances)
            m.future_taken += Ratio(num=s.future_taken, den=s.future)
            m.trap_pass += Ratio(num=s.traps_passed, den=s.traps)
            m.merges += s.merges
            m.splits += s.splits
            errors.update(s.errors)
            for key, verdict in s.verdicts.items():
                verdicts.setdefault(key, []).append(verdict)
            age_keys.setdefault(s.id, []).append(s.age_keys)
    found_in: Counter[int] = Counter()
    for vs in verdicts.values():
        found = sum(v is not None for v in vs)
        found_in[found] += 1
        if found:
            m.stable += Ratio(num=int(all(v == vs[0] for v in vs)), den=1)
    for sets in age_keys.values():
        m.age_consistency += Ratio(num=len(set.intersection(*sets)), den=len(set.union(*sets)))
    m.found_in = dict(sorted(found_in.items(), reverse=True))
    m.errors = dict(errors.most_common())
    return m


GROUPS = ("dense", "random", "shushan")


def summarize(ids: list[str], samples: list[list[ChapterScore]]) -> list[RealtextMetrics]:
    """Overall, per group and per chapter; `samples[i]` holds sample i's chapter scores."""
    rows = [aggregate("全部", samples)]
    for kind in GROUPS:
        rows.append(aggregate(kind, [[s for s in sample if s.kind == kind] for sample in samples]))
    for id_ in ids:
        rows.append(aggregate(id_, [[s for s in sample if s.id == id_] for sample in samples]))
    return rows


# --- running ---------------------------------------------------------------------------


class RealtextRun(BaseModel):
    mode: str  # "context" / "single"
    rows: list[RealtextMetrics] = []
    failed_chapters: int = 0
    cost_usd: Decimal = Decimal(0)
    llm_calls: int = 0
    cache_hits: int = 0


class RealtextReport(BaseModel):
    label: str
    created_at: datetime
    extract_model: str
    prompt_hash: str
    git_commit: str | None
    samples: int
    fresh: bool
    replay: bool = False
    runs: list[RealtextRun]


async def run_mode(
    jobs: list[Job],
    mode: str,
    settings: Settings,
    client_for_sample: Callable[[int], ClientFactory],
    cache: Factory,
    *,
    samples: int,
    fresh: bool,
    concurrency: int = 6,
    on_done: Callable[[Job, int], None] | None = None,
) -> RealtextRun:
    """Every job × every sample, each in its own throw-away database."""
    gate = asyncio.Semaphore(concurrency)
    per_sample: list[list[ChapterScore]] = [[] for _ in range(samples)]
    run = RealtextRun(mode=mode)

    async def one(job: Job, sample: int) -> None:
        async with gate:
            obs, result = await import_and_observe(
                job.name, job.text, settings, client_for_sample(sample), cache, fresh=fresh
            )
        per_sample[sample].extend(score_chapter(t, obs) for t in job.targets)
        run.failed_chapters += result.failed
        run.cost_usd += result.cost_usd
        run.llm_calls += result.llm_calls
        run.cache_hits += result.cache_hits
        if on_done:
            on_done(job, sample)

    await asyncio.gather(*(one(job, i) for job in jobs for i in range(samples)))
    ids = sorted(t.annotation.id for job in jobs for t in job.targets)
    run.rows = summarize(ids, per_sample)
    return run
