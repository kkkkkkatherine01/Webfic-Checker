"""webfic-eval: score the pipeline against the golden set."""

import asyncio
import hashlib
import re
import subprocess
from collections import Counter
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from webfic.config import get_settings
from webfic.evaluation.golden import GoldenError, Story, discover, load_story
from webfic.evaluation.metrics import Metrics, Ratio, RunMeta, RunReport, aggregate_story, combine
from webfic.evaluation.runner import open_cache, run_story
from webfic.extraction.extractor import load_prompt
from webfic.llm.client import CallStore
from webfic.llm.factory import LLMNotConfigured, platform_client

app = typer.Typer(help="用黄金测试集评估抽取与检查效果", no_args_is_help=True, add_completion=False)
console = Console()

EVAL_DIR = Path(__file__).resolve().parents[4] / "eval"


def _file_label(label: str) -> str:
    """A run label as part of a file name: characters Windows forbids become "-" (a
    colon would otherwise write the result into an NTFS alternate stream)."""
    return re.sub(r'[<>:"/\\|?*]', "-", label).strip() or "run"


GoldenDir = Annotated[Path, typer.Option(help="黄金测试集目录")]


def _load(golden_dir: Path, only: list[str] | None) -> list[Story]:
    stories, failed = [], False
    for directory in discover(golden_dir, only):
        try:
            stories.append(load_story(directory))
        except GoldenError as exc:
            console.print(f"[red]✗ {exc}[/]")
            failed = True
    if failed:
        raise typer.Exit(1)
    if not stories:
        console.print("[red]没有找到测试稿[/]")
        raise typer.Exit(1)
    return stories


def _git_commit() -> str | None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        return commit + ("+改动" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


# --- printing --------------------------------------------------------------------------


def _metrics_table(rows: list[Metrics]) -> Table:
    table = Table(
        "篇目", "矛盾召回", "精确率", "置信度", "年龄召回", "年龄全对", "时间段召回",
        "推进/回顾", "陷阱", "合并/拆分", "丢弃", "花费 $",
        title="评估结果", show_lines=False,
    )  # fmt: skip
    for m in rows:
        style = "bold" if m.story == "全部" else None
        table.add_row(
            m.story, str(m.issue_recall), str(m.issue_precision), str(m.confidence_accuracy),
            str(m.age_recall), str(m.age_accuracy), str(m.elapsed_recall),
            str(m.elapsed_kind_accuracy), str(m.trap_pass), f"{m.merges}/{m.splits}",
            str(m.dropped), f"{m.cost_usd:.4f}", style=style,
        )  # fmt: skip
    return table


def _char_table(rows: list[Metrics]) -> Table | None:
    """Character facts (step 5-1), when any story has them."""
    dens = (
        "char_issue_recall", "char_issue_precision", "trait_recall", "kinship_recall",
        "death_recall", "char_trap_pass",
    )  # fmt: skip
    if not any(getattr(m, d).den for m in rows for d in dens):
        return None
    table = Table(
        "篇目", "矛盾召回", "精确率", "外貌召回", "外貌全对", "亲属召回", "亲属全对",
        "死亡召回", "死亡全对", "陷阱", title="人物事实", show_lines=False,
    )  # fmt: skip
    for m in rows:
        table.add_row(
            m.story, str(m.char_issue_recall), str(m.char_issue_precision), str(m.trait_recall),
            str(m.trait_accuracy), str(m.kinship_recall), str(m.kinship_accuracy),
            str(m.death_recall), str(m.death_accuracy), str(m.char_trap_pass),
            style="bold" if m.story == "全部" else None,
        )  # fmt: skip
    return table


def _print_details(overall: Metrics) -> None:
    if overall.failed_chapters:
        console.print(
            f"\n[bold red]有 {overall.failed_chapters} 个章节（各次采样累计）抽取失败[/]"
            "（内容审核拒绝、输出无法解析等），相关指标偏低不一定是 prompt 的问题。"
        )
    unstable = overall.unstable_issues()
    if unstable:
        console.print("\n[bold]漏报 / 不稳定的矛盾[/]（检出次数 / 采样次数）")
        for issue, hits in sorted(unstable.items()):
            console.print(f"  {issue}: {hits}/{overall.samples}")
    if overall.char_allowed:
        console.print(
            f"\n人物事实：另有 {overall.char_allowed} 次报出标注允许或应由校验 agent 驳回的问题"
            "（原文交代了原因），不算误报"
        )
    if overall.false_positives:
        console.print("\n[bold]误报[/]")
        for desc, n in sorted(overall.false_positives.items(), key=lambda kv: -kv[1]):
            console.print(f"  ×{n} {desc}")
    if overall.fact_errors:
        console.print("\n[bold]抽取与置信度错误[/]（出现次数）")
        for err, n in sorted(overall.fact_errors.items(), key=lambda kv: -kv[1])[:30]:
            console.print(f"  ×{n} {err}")
    console.print(
        f"\nLLM 调用 {overall.llm_calls} 次（缓存命中 {overall.cache_hits}），"
        f"输入 {overall.input_tokens} / 输出 {overall.output_tokens} tokens，"
        f"约 ${overall.cost_per_1k_chars or 0} / 千字，耗时 {overall.seconds:.0f} 秒"
    )


# --- commands --------------------------------------------------------------------------


@app.command()
def validate(golden_dir: GoldenDir = EVAL_DIR / "golden") -> None:
    """只校验 expected.yaml（不调用模型）。"""
    for story in _load(golden_dir, None):
        g = story.golden
        traps = len(g.facts.not_ages) + len(g.facts.not_elapsed) + len(g.not_aliases)
        console.print(
            f"[green]✓[/] {story.id}《{g.title}》{len(story.chapters)} 章 "
            f"{len(story.text)} 字：矛盾 {len(g.issues)}、年龄 {len(g.facts.ages)}、"
            f"时间段 {len(g.facts.elapsed)}、陷阱 {traps}"
        )


@app.command()
def run(
    stories: Annotated[str | None, typer.Option(help="只跑部分篇目，如 01,03")] = None,
    samples: Annotated[int, typer.Option(min=1, help="每篇运行次数；>1 时自动绕过缓存")] = 1,
    fresh: Annotated[bool, typer.Option(help="绕过缓存，真实调用模型")] = False,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "run",
    baseline: Annotated[bool, typer.Option(help="同时保存为 eval/baseline.json")] = False,
    golden_dir: GoldenDir = EVAL_DIR / "golden",
) -> None:
    """跑测试稿并打分，结果保存到 eval/runs/。"""
    settings = get_settings()
    selected = _load(golden_dir, stories.split(",") if stories else None)
    fresh = fresh or samples > 1

    def make_client(store: CallStore):
        return platform_client(settings, store)

    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    async def main() -> list[Metrics]:
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")
        results = []
        for story in selected:
            console.print(f"运行 {story.id}（{samples} 次{'，绕过缓存' if fresh else ''}）…")
            sample_results = await run_story(
                story, settings, make_client, cache, samples=samples, fresh=fresh
            )
            results.append(aggregate_story(story.id, len(story.text), sample_results))
        return results

    per_story = asyncio.run(main())
    overall = combine(per_story)
    report = RunReport(
        meta=RunMeta(
            label=label,
            created_at=datetime.now().astimezone(),
            provider=settings.llm_base_url,
            extract_model=settings.llm_extract_model,
            prompt_hash=hashlib.sha256(load_prompt().encode()).hexdigest()[:12],
            git_commit=_git_commit(),
            samples=samples,
            fresh=fresh,
            stories=[s.id for s in selected],
        ),
        overall=overall,
        stories=per_story,
    )

    console.print(_metrics_table([*per_story, overall]))
    if (chars := _char_table([*per_story, overall])) is not None:
        console.print(chars)
    _print_details(overall)

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report.meta.created_at:%Y%m%d-%H%M%S}-{_file_label(label)}.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    console.print(f"\n结果已保存：{path}")
    if baseline:
        (EVAL_DIR / "baseline.json").write_text(report.model_dump_json(indent=2), "utf-8")
        console.print(f"已保存为基线：{EVAL_DIR / 'baseline.json'}")


def _resolve_report(ref: str) -> Path:
    if ref == "baseline":
        return EVAL_DIR / "baseline.json"
    if ref == "latest":
        runs = sorted((EVAL_DIR / "runs").glob("*.json"))
        if not runs:
            raise typer.BadParameter("eval/runs/ 里还没有结果")
        return runs[-1]
    path = Path(ref)
    return path if path.exists() else EVAL_DIR / "runs" / ref


def _delta(a: Ratio, b: Ratio) -> str:
    if a.value is None or b.value is None:
        return ""
    diff = b.value - a.value
    if abs(diff) < 0.0005:
        return "[dim]=[/]"
    return f"[green]+{diff:.0%}[/]" if diff > 0 else f"[red]{diff:.0%}[/]"


@app.command()
def compare(
    a: Annotated[str, typer.Argument(help="旧结果：文件名、路径、baseline 或 latest")],
    b: Annotated[str, typer.Argument(help="新结果")] = "latest",
) -> None:
    """对比两次运行结果。"""
    old = RunReport.model_validate_json(_resolve_report(a).read_text("utf-8"))
    new = RunReport.model_validate_json(_resolve_report(b).read_text("utf-8"))

    console.print(
        f"旧：{old.meta.label}  {old.meta.created_at:%m-%d %H:%M}  "
        f"prompt {old.meta.prompt_hash}  ×{old.meta.samples}"
    )
    console.print(
        f"新：{new.meta.label}  {new.meta.created_at:%m-%d %H:%M}  "
        f"prompt {new.meta.prompt_hash}  ×{new.meta.samples}"
    )

    table = Table("指标", "旧", "新", "变化")
    for field, name in [
        ("issue_recall", "矛盾召回"), ("issue_precision", "精确率"),
        ("confidence_accuracy", "置信度"), ("age_recall", "年龄召回"),
        ("age_accuracy", "年龄全对"), ("elapsed_recall", "时间段召回"),
        ("elapsed_kind_accuracy", "推进/回顾"), ("trap_pass", "陷阱"),
        ("char_issue_recall", "人物事实·矛盾召回"), ("char_issue_precision", "人物事实·精确率"),
        ("trait_recall", "外貌召回"), ("kinship_recall", "亲属召回"),
        ("death_recall", "死亡召回"), ("char_trap_pass", "人物事实·陷阱"),
    ]:  # fmt: skip
        ra, rb = getattr(old.overall, field), getattr(new.overall, field)
        table.add_row(name, str(ra), str(rb), _delta(ra, rb))
    table.add_row(
        "每千字花费 $", str(old.overall.cost_per_1k_chars), str(new.overall.cost_per_1k_chars), ""
    )
    console.print(table)

    def rate(r: RunReport, key: str) -> float | None:
        hits = r.overall.issue_hits.get(key)
        return None if hits is None else hits / r.overall.samples

    changes = []
    for key in sorted(set(old.overall.issue_hits) | set(new.overall.issue_hits)):
        ra, rb = rate(old, key), rate(new, key)
        if ra != rb:
            fmt = lambda x: "—" if x is None else f"{x:.0%}"  # noqa: E731
            changes.append(f"  {key}: {fmt(ra)} → {fmt(rb)}")
    if changes:
        console.print("\n[bold]检出率有变化的矛盾[/]")
        console.print("\n".join(changes))

    gone = set(old.overall.false_positives) - set(new.overall.false_positives)
    came = set(new.overall.false_positives) - set(old.overall.false_positives)
    for title, items, color in [("消失的误报", gone, "green"), ("新增的误报", came, "red")]:
        if items:
            console.print(f"\n[bold {color}]{title}[/]")
            for item in sorted(items):
                console.print(f"  {item}")


REALTEXT_DIR = EVAL_DIR / "external" / "realtext"


def _realtext_table(run) -> Table:
    table = Table(
        "章节", "年龄召回", "年龄精确", "属性全对", "时间段召回", "推进判断", "推进精确",
        "将来时长标错", "陷阱", "合并/拆分", "判定一致", "年龄抽取一致",
        title=f"真实文本抽取（{'按顺序抽取整段' if run.mode == 'context' else '单章抽取'}）",
    )  # fmt: skip
    for m in run.rows:
        style = "bold" if not m.name.isdigit() else None
        table.add_row(
            m.name, str(m.age_recall), str(m.age_precision), str(m.age_accuracy),
            str(m.elapsed_recall), str(m.elapsed_accuracy), str(m.advance_precision),
            str(m.future_taken), str(m.trap_pass), f"{m.merges}/{m.splits}", str(m.stable),
            str(m.age_consistency), style=style,
        )  # fmt: skip
    return table


@app.command()
def realtext(
    samples: Annotated[int, typer.Option(min=1, help="每章运行次数；>1 时自动绕过缓存")] = 5,
    fresh: Annotated[bool, typer.Option(help="绕过缓存，真实调用模型")] = False,
    replay: Annotated[
        bool,
        typer.Option(help="从评估缓存重放之前各次采样的输出；缓存里没有的请求才调用模型"),
    ] = False,
    single: Annotated[bool, typer.Option(help="同时做单章抽取的对比")] = True,
    check: Annotated[bool, typer.Option(help="只校验标注，不调用模型")] = False,
    concurrency: Annotated[int, typer.Option(min=1, help="同时运行的导入数")] = 6,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "realtext",
    baseline: Annotated[bool, typer.Option(help="同时保存为 eval/realtext-baseline.json")] = False,
) -> None:
    """真实网文抽取质量（3.5-1）：召回、精确、属性、推进误抽、陷阱、多次采样的一致性。"""
    from webfic.evaluation import realtext as rt
    from webfic.evaluation.runner import ReplayCallStore

    external = EVAL_DIR / "external"
    try:
        annotations = rt.load_annotations(REALTEXT_DIR / "annotations.yaml")
        picks = rt.load_picks(REALTEXT_DIR / "selection.json")
        books = {
            b.name: b
            for b in rt.webnovelbench_books(
                external / "webnovelbench" / "novel_data_subset_d_100.json"
            )
        }
        books["shushan"] = rt.shushan_book(external / "shushan" / "shushan_001-012.txt")
        modes = ["context", "single"] if single else ["context"]
        plans = {mode: rt.plan_jobs(annotations, picks, books, mode) for mode in modes}
    except (OSError, rt.AnnotationError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    required = sum(len(a.ages) + len(a.elapsed) for a in annotations)
    console.print(f"[green]✓[/] 标注 {len(annotations)} 章，必抽 {required} 条")
    if check:
        return

    settings = get_settings()
    fresh = not replay and (fresh or samples > 1)
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    async def main() -> list:
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")

        def client_for_sample(sample: int):
            if replay:
                store = ReplayCallStore(cache, sample)
                return lambda _: platform_client(settings, store)
            return lambda store: platform_client(settings, store)

        runs = []
        count = {"done": 0, "total": sum(len(jobs) for jobs in plans.values()) * samples}

        def progress(job, sample) -> None:
            count["done"] += 1
            console.print(f"  [{count['done']}/{count['total']}] {job.name} 第 {sample + 1} 次")

        for mode, jobs in plans.items():
            console.print(f"{mode}：{len(jobs)} 个导入 × {samples} 次…")
            run = await rt.run_mode(
                jobs, mode, settings, client_for_sample, cache, samples=samples, fresh=fresh,
                concurrency=concurrency, on_done=None if replay else progress,
            )  # fmt: skip
            runs.append(run)
        return runs

    runs = asyncio.run(main())
    report = rt.RealtextReport(
        label=label,
        created_at=datetime.now().astimezone(),
        extract_model=settings.llm_extract_model,
        prompt_hash=hashlib.sha256(load_prompt().encode()).hexdigest()[:12],
        git_commit=_git_commit(),
        samples=samples,
        fresh=fresh,
        replay=replay,
        runs=runs,
    )
    for run in runs:
        console.print(_realtext_table(run))
        overall = run.rows[0]
        spread = "、".join(f"{k} 次 {n} 条" for k, n in overall.found_in.items())
        console.print(
            f"必抽条目在 {samples} 次中被抽到的次数：{spread}；"
            f"时间段精确（仅供参考，时长未完整标注）{overall.elapsed_precision}"
        )
        if run.failed_chapters:
            console.print(f"[bold red]抽取失败的章节（各次累计）：{run.failed_chapters}[/]")
        console.print(
            f"LLM 调用 {run.llm_calls} 次（缓存命中 {run.cache_hits}），花费 ${run.cost_usd:.4f}"
        )
    console.print("\n[bold]错误（按顺序抽取；出现次数）[/]")
    for err, n in list(runs[0].rows[0].errors.items())[:60]:
        console.print(f"  ×{n} {err}")

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report.created_at:%Y%m%d-%H%M%S}-realtext-{_file_label(label)}.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    console.print(f"\n结果已保存：{path}")
    if baseline:
        (EVAL_DIR / "realtext-baseline.json").write_text(report.model_dump_json(indent=2), "utf-8")
        console.print(f"已保存为基线：{EVAL_DIR / 'realtext-baseline.json'}")


class _LayeredStore:
    """Looks a request up in several stores in turn; records to all of them. `fresh`
    skips the lookups (a new sample)."""

    def __init__(self, *stores: CallStore, fresh: bool = False):
        self._stores, self._fresh = stores, fresh

    async def lookup(self, request_hash: str) -> str | None:
        if self._fresh:
            return None
        for store in self._stores:
            if (found := await store.lookup(request_hash)) is not None:
                return found
        return None

    async def record(self, record) -> None:
        for store in self._stores:
            if not record.cache_hit or store is self._stores[0]:
                await store.record(record)


@app.command()
def inject(
    prepare_only: Annotated[bool, typer.Option(help="只导入并抽取底稿，不注入")] = False,
    books: Annotated[int, typer.Option(help="只用前 N 部底稿（试跑用；0 = 全部）")] = 0,
    scale: Annotated[float, typer.Option(help="各类型数量乘以这个系数（试跑用）")] = 1.0,
    fresh: Annotated[bool, typer.Option(help="注入时绕过缓存，真实调用模型")] = False,
    seed: Annotated[int, typer.Option(help="选注入点的随机种子")] = 0,
    concurrency: Annotated[int, typer.Option(min=1, help="同时处理的作品数")] = 6,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "inject",
    baseline: Annotated[bool, typer.Option(help="同时保存为 eval/inject-baseline.json")] = False,
) -> None:
    """错误注入评估（3.5-2）：在真实网文里注入年龄矛盾和对照改动，逐处试算，看能否报出。"""
    from webfic.db.session import make_engine, make_session_factory
    from webfic.evaluation import inject as ij
    from webfic.llm.cache import DbCallStore

    settings = get_settings()
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    items = ij.bases(EVAL_DIR / "external", EVAL_DIR / "private")
    if books:
        items = items[:books]
    console.print(f"底稿 {len(items)} 部，{sum(len(b.text) for b in items)} 字")

    async def main():
        engine = make_engine(settings.database_url)
        factory = make_session_factory(engine)
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")

        def make_llm(book_id, *, fresh_calls: bool = False):
            store = _LayeredStore(
                DbCallStore(factory, user_id=ij.INJECT_USER, book_id=book_id),
                DbCallStore(cache, user_id=None),
                fresh=fresh_calls,
            )
            return platform_client(settings, store)

        try:
            prepared = await ij.prepare(
                factory, make_llm, settings, items, concurrency=concurrency,
                on_progress=console.print,
            )  # fmt: skip
            async with factory() as session:
                views = [await ij.load_view(session, b.name, prepared[b.name]) for b in items]
            if prepare_only:
                return views, []
            quotas = {k: max(1, round(q * scale)) for k, q in ij.QUOTAS.items()}
            injections = ij.plan(views, quotas, seed or ij.SEED)
            console.print(
                "注入："
                + "、".join(
                    f"{label} {sum(i.kind == k for i in injections)}"
                    for k, label in ij.KINDS.items()
                )
            )
            count = {"done": 0}

            def done(outcome) -> None:
                count["done"] += 1
                if count["done"] % 10 == 0:
                    console.print(f"  [{count['done']}/{len(injections)}]")

            outcomes = await ij.run_all(
                factory, lambda book_id: make_llm(book_id, fresh_calls=fresh), settings,
                prepared, injections, concurrency=concurrency, on_done=done,
            )  # fmt: skip
            return views, outcomes
        finally:
            await engine.dispose()

    views, outcomes = asyncio.run(main())
    everything = [i for v in views for i in v.issues]
    original = [i for i in everything if i.checker == ij.AGE_CHECKER]
    char_original = [i for i in everything if i.checker != ij.AGE_CHECKER]
    by_confidence = Counter(str(i.confidence) for i in original)
    console.print(
        f"原文上的年龄矛盾报告：{len(original)} 条（"
        + "、".join(f"{k} {n}" for k, n in by_confidence.most_common())
        + "）"
    )
    if char_original:
        by_type = Counter(str(i.issue_type) for i in char_original)
        console.print(
            f"原文上的人物事实报告：{len(char_original)} 条（"
            + "、".join(f"{k} {n}" for k, n in by_type.most_common())
            + "）"
        )
    if prepare_only:
        return

    kinds = ij.score(outcomes)
    table = Table(
        "类型", "数量", "检出", "置信度对", "误报", "注入年龄抽对", "抽成推进", "报告有变化",
        "连带变化", "花费 $", title="错误注入评估",
    )  # fmt: skip
    for m in kinds:
        detect = m.kind in ij.DETECT
        table.add_row(
            m.label, str(m.n), str(m.detected) if detect else "",
            str(m.confidence_right) if detect else "", "" if detect else str(m.false_alarm),
            str(m.age_extracted_right), str(m.taken_as_advance), str(m.report_changed),
            str(m.collateral), f"{m.cost_usd:.4f}",
        )  # fmt: skip
    console.print(table)
    for m in kinds:
        if m.misses:
            console.print(f"\n[bold]{m.label} 漏报原因[/]：" + "、".join(
                f"{k} {n}" for k, n in m.misses.items()))  # fmt: skip
        if m.by_distance:
            console.print(
                f"  {m.label} 按距离：" + "、".join(f"{k} {r}" for k, r in m.by_distance.items())
            )
    for o in outcomes:
        if o.injection.kind not in ij.DETECT and o.involving:
            console.print(f"\n[red]误报[/] {o.injection.id}：{o.injection.detail}")
            for text in o.involving:
                console.print(f"    {text}")
    failed = sum(o.failed for o in outcomes)
    if failed:
        console.print(f"[bold red]注入所在章节抽取失败：{failed} 处[/]")

    report = ij.InjectReport(
        label=label,
        created_at=datetime.now().astimezone().isoformat(),
        extract_model=settings.llm_extract_model,
        prompt_hash=hashlib.sha256(load_prompt().encode()).hexdigest()[:12],
        git_commit=_git_commit(),
        seed=seed or ij.SEED,
        books=len(views),
        original_issues=len(original),
        original_char_issues=len(char_original),
        kinds=kinds,
        outcomes=outcomes,
    )
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-inject-{_file_label(label)}.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    console.print(f"\n结果已保存：{path}")
    if baseline:
        (EVAL_DIR / "inject-baseline.json").write_text(report.model_dump_json(indent=2), "utf-8")
        console.print(f"已保存为基线：{EVAL_DIR / 'inject-baseline.json'}")


@app.command("retrieval-prepare")
def retrieval_prepare(
    passage_size: Annotated[int | None, typer.Option(help="检索段落长度，默认用配置")] = None,
    passage_overlap: Annotated[int | None, typer.Option(help="相邻段落重叠，默认用配置")] = None,
) -> None:
    """为检索评估准备 DetectiveQA 语料：导入并建索引（本地计算，可中断后续跑）。"""
    from webfic.archival.index import platform_archival
    from webfic.db.session import make_engine, make_session_factory
    from webfic.evaluation.retrieval import detectiveqa_corpus, prepare

    settings = get_settings()
    updates = {
        k: v
        for k, v in {"passage_size": passage_size, "passage_overlap": passage_overlap}.items()
        if v
    }
    settings = settings.model_copy(update=updates)
    corpus = detectiveqa_corpus(EVAL_DIR / "external" / "detectiveqa")
    if not corpus:
        console.print(
            "[red]没有找到 DetectiveQA 数据：先运行 python eval/download_external.py detectiveqa[/]"
        )
        raise typer.Exit(1)

    async def main() -> None:
        engine = make_engine(settings.database_url)
        try:
            await prepare(
                make_session_factory(engine), platform_archival(settings), corpus, console.print
            )
        finally:
            await engine.dispose()

    console.print(
        f"DetectiveQA {len(corpus)} 部，{sum(len(c.text) for c in corpus)} 字；"
        f"段落 {settings.passage_size}/{settings.passage_overlap}"
    )
    asyncio.run(main())


@app.command()
def retrieval(
    corpus: Annotated[
        str, typer.Option(help="golden、detectiveqa（原问题 + 线索描述两套查询），逗号分隔")
    ] = "golden",
    sizes: Annotated[str, typer.Option(help="段落长度-重叠，逗号分隔，如 300-60,200-40")] = "",
    modes: Annotated[str, typer.Option(help="hybrid、vector、keyword，逗号分隔")] = (
        "hybrid,vector,keyword"
    ),
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "retrieval",
    golden_dir: GoldenDir = EVAL_DIR / "golden",
) -> None:
    """检索评估：命中率@5/@10、线索覆盖率@10、MRR（本地计算；缺的索引会先建好）。"""
    import dataclasses

    from webfic.archival.index import platform_archival
    from webfic.db.session import make_engine, make_session_factory
    from webfic.evaluation import retrieval as rv
    from webfic.evaluation.runner import EvalCallStore
    from webfic.llm.cache import DbCallStore

    settings = get_settings()
    corpora = [c.strip() for c in corpus.split(",") if c.strip()]
    wanted_modes = [m.strip() for m in modes.split(",") if m.strip()]
    base = platform_archival(settings)
    variants = [base]
    if sizes:
        variants = []
        for item in sizes.split(","):
            size, _, overlap = item.strip().partition("-")
            variants.append(
                dataclasses.replace(base, passage_size=int(size), passage_overlap=int(overlap))
            )

    stories = (
        [s for s in _load(golden_dir, None) if s.golden.retrieval] if "golden" in corpora else []
    )
    dqa_folder = EVAL_DIR / "external" / "detectiveqa"
    dqa_questions, dqa_clues, dqa_clues_any, skipped = (
        rv.detectiveqa_questions(dqa_folder) if "detectiveqa" in corpora else ([], [], [], 0)
    )

    async def main() -> list[rv.Scores]:
        engine = make_engine(settings.database_url)
        factory = make_session_factory(engine)
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")

        def extract(item: rv.CorpusText):
            story = next(s for s in stories if f"golden/{s.id}" == item.name)
            story_settings = settings.model_copy(
                update=story.golden.settings.model_dump(exclude_none=True)
            )
            store = EvalCallStore(
                DbCallStore(factory, user_id=rv.RETRIEVAL_USER), DbCallStore(cache, user_id=None),
                fresh=False,
            )  # fmt: skip
            return platform_client(story_settings, store), story_settings

        results: list[rv.Scores] = []
        try:
            for archival in variants:
                runs = []
                if stories:
                    books = await rv.prepare(
                        factory, archival, rv.golden_corpus(stories), console.print, extract
                    )
                    runs.append(("golden", books, rv.golden_questions(stories)))
                if dqa_questions:
                    books = await rv.prepare(
                        factory, archival, rv.detectiveqa_corpus(dqa_folder), console.print
                    )
                    runs.append(("detectiveqa", books, dqa_questions))
                    runs.append(("detectiveqa-clues", books, dqa_clues))
                    runs.append(("detectiveqa-clues-any", books, dqa_clues_any))
                for name, books, questions in runs:
                    for mode in wanted_modes:
                        outcomes = await rv.evaluate(factory, archival, books, questions, mode)
                        results.append(rv.aggregate(name, archival, mode, outcomes))
        finally:
            await engine.dispose()
        return results

    scores = asyncio.run(main())
    table = Table("语料", "段落", "方式", "问题数", "命中@5", "命中@10", "线索覆盖@10", "MRR")
    for s in scores:
        table.add_row(
            s.corpus, f"{s.passage_size}/{s.passage_overlap}", s.mode, str(s.questions),
            f"{s.hit_at_5:.0%}", f"{s.hit_at_10:.0%}", f"{s.coverage_at_10:.0%}", f"{s.mrr:.2f}",
        )  # fmt: skip
    console.print(table)
    for s in scores:
        if s.corpus == "golden" and s.missed:
            console.print(f"\n[bold]{s.passage_size}/{s.passage_overlap} {s.mode} 没找到[/]")
            for q in s.missed:
                console.print(f"  {q}")
    report = rv.RetrievalReport(
        label=label, created_at=datetime.now().astimezone(), embed_model=settings.embed_model,
        scores=scores, skipped_clues=skipped,
    )  # fmt: skip
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report.created_at:%Y%m%d-%H%M%S}-retrieval-{_file_label(label)}.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    console.print(f"\n结果已保存：{path}")


# --- verify agent (step 4-3) ------------------------------------------------------------


@app.command("verify")
def verify_cmd(
    sets: Annotated[
        str, typer.Option(help="keep（真矛盾）、synthetic（合成误报）、real（真实报告，保留集）")
    ] = "keep,synthetic",
    injected: Annotated[int, typer.Option(help="keep 集里注入的年龄真矛盾条数")] = 40,
    injected_facts: Annotated[
        int, typer.Option(help="keep 集里注入的人物事实真矛盾条数（5-1c）")
    ] = 30,
    per_type: Annotated[int, typer.Option(help="合成误报每种类型的条数")] = 15,
    samples: Annotated[int, typer.Option(min=1, help="每条运行次数；>1 时自动绕过缓存")] = 1,
    fresh: Annotated[bool, typer.Option(help="绕过缓存，真实调用模型")] = False,
    model: Annotated[str | None, typer.Option(help="覆盖校验 agent（verify 档）的模型")] = None,
    extra: Annotated[
        str | None,
        typer.Option(
            help='覆盖 verify 档的额外参数（JSON），如 {"thinking": {"type": "disabled"}}'
        ),
    ] = None,
    replan: Annotated[bool, typer.Option(help="重新生成评估用的问题集合")] = False,
    corpus: Annotated[
        str,
        typer.Option(
            help="normal：原底稿；long：每 3 章合并的长章节底稿（webfic-eval long-chapters）"
        ),
    ] = "normal",
    prompt: Annotated[
        str | None, typer.Option(help="校验 agent 的 prompt 版本（默认当前版本）")
    ] = None,
    concurrency: Annotated[int, typer.Option(min=1, help="同时处理的作品数")] = 4,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "verify",
    baseline: Annotated[bool, typer.Option(help="同时保存为 eval/verify-baseline.json")] = False,
) -> None:
    """校验 agent 评估（4-3）：真矛盾不能误杀、合成误报要驳回、真实报告（保留集）。"""
    import json

    from rich.markup import escape

    from webfic.agent.verify import PROMPT_VERSION
    from webfic.archival.index import platform_archival
    from webfic.db.session import make_engine, make_session_factory
    from webfic.evaluation import inject as ij
    from webfic.evaluation import verify_eval as ve
    from webfic.extraction.extractor import extraction_version
    from webfic.llm.cache import DbCallStore

    wanted = {s.strip() for s in sets.split(",") if s.strip()}
    if not wanted <= {"keep", "synthetic", "real"}:
        raise typer.BadParameter(f"未知的集合：{sets}")
    settings = get_settings()
    overrides = {}
    if model:
        overrides["llm_verify_model"] = model
    if extra is not None:
        overrides["llm_verify_extra"] = json.loads(extra)
    verify_settings = settings.model_copy(update=overrides)
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    if corpus not in ("normal", "long"):
        raise typer.BadParameter("corpus 只能是 normal 或 long")
    plan_path = EVAL_DIR / ("verify-plan-long.json" if corpus == "long" else "verify-plan.json")

    async def main():
        engine = make_engine(settings.database_url)
        factory = make_session_factory(engine)
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")
        archival = platform_archival(settings)
        try:
            if plan_path.exists() and not replan:
                plan = [ve.Case.model_validate(c) for c in json.loads(plan_path.read_text("utf-8"))]
                console.print(f"沿用问题集合 {plan_path.name}（{len(plan)} 条；--replan 重新生成）")
            else:
                version = extraction_version(
                    load_prompt(), settings.chunk_size, settings.chunk_overlap
                )
                async with factory() as session:
                    books = await ve._books(session, ij.INJECT_USER, "@" + version[:8])
                long = corpus == "long"
                books = {n: b for n, b in books.items() if n.startswith("long/") == long}
                if not books:
                    raise typer.BadParameter(
                        "找不到底稿，请先运行 webfic-eval long-chapters"
                        if long
                        else "找不到注入评估的底稿，请先运行 webfic-eval inject --prepare-only"
                    )
                if not long:
                    # Test stories added since, and kinds of extraction added since.
                    def extract(item):
                        store = _LayeredStore(
                            DbCallStore(factory, user_id=rv_user, book_id=None),
                            DbCallStore(cache, user_id=None),
                        )
                        return platform_client(settings, store), settings

                    from webfic.evaluation.retrieval import RETRIEVAL_USER as rv_user

                    await ve.prepare_golden(
                        factory, archival, EVAL_DIR / "golden", extract, console.print
                    )
                plan = [] if long else await ve.golden_cases(factory, EVAL_DIR / "golden")
                plan += await ve.injected_cases(factory, books, 400)
                plan += await ve.synthetic_cases(factory, books, 40, on_progress=console.print)
                plan_path.write_text(
                    json.dumps(
                        [c.model_dump(mode="json") for c in plan], ensure_ascii=False, indent=1
                    ),
                    "utf-8",
                )
                console.print(f"问题集合已保存：{plan_path.name}（{len(plan)} 条）")
            chosen: list[ve.Case] = []
            if "keep" in wanted:
                chosen += [c for c in plan if c.kind.startswith("golden") and c.set == "keep"]
                # Round-robin over the injection kinds (the plan lists them kind by kind).
                by_kind: dict[str, list[ve.Case]] = {}
                for c in plan:
                    if c.kind.startswith("injected:"):
                        by_kind.setdefault(c.kind, []).append(c)
                ages = [v for k, v in by_kind.items() if not k.startswith("injected:fact_")]
                facts = [v for k, v in by_kind.items() if k.startswith("injected:fact_")]
                for groups, count in ((ages, injected), (facts, injected_facts)):
                    mixed = [c for group in zip_longest(*groups) for c in group if c]
                    chosen += mixed[:count]
            if "synthetic" in wanted:
                chosen += [c for c in plan if c.kind == "golden:false_alarm"]
                for kind in ve.CORRUPTIONS:
                    chosen += [c for c in plan if c.kind == f"synthetic:{kind}"][:per_type]
            if "real" in wanted:  # from the labels file every time, so corrections apply
                real = await ve.real_cases(factory, ve.load_labels(EVAL_DIR))
                if not real:
                    raise typer.BadParameter("没有真实报告的标注：eval/external/inject/labels.yaml")
                chosen += real
            await ve.index_books(factory, archival, chosen, console.print)

            new_samples = fresh or samples > 1

            def make_llms(case):
                def store(fresh_calls):
                    return _LayeredStore(
                        DbCallStore(factory, user_id=case.user_id, book_id=case.book_id),
                        DbCallStore(cache, user_id=None),
                        fresh=fresh_calls,
                    )

                return (
                    platform_client(settings, store(False)),  # the set-up: text edits re-read
                    platform_client(verify_settings, store(new_samples)),
                )

            count = {"done": 0}
            total = len(chosen) * samples

            def done(result) -> None:
                count["done"] += 1
                if count["done"] % 10 == 0:
                    console.print(f"  [{count['done']}/{total}]")

            console.print(
                f"核实 {len(chosen)} 条 × {samples} 次，模型 {verify_settings.llm_verify_model}"
                f" {json.dumps(verify_settings.llm_verify_extra, ensure_ascii=False)}"
            )
            results = await ve.run_all(
                chosen, samples=samples, factory=factory, make_llms=make_llms,
                settings=verify_settings, archival=archival, concurrency=concurrency,
                on_done=done, prompt_version=prompt,
            )  # fmt: skip
            return results
        finally:
            await engine.dispose()

    results = asyncio.run(main())
    scores = ve.score(results)
    table = Table(
        "集合", "条数", "次数", "未复现", "正确", "判为误报", "需作者确认", "未完成", "原因对",
        "稳定", "轮数", "工具调用", "每次 $", title="校验 agent 评估",
    )  # fmt: skip
    for s in scores:
        table.add_row(
            s.set, str(s.cases), str(s.runs), str(s.not_set_up), str(s.right), str(s.dismissed),
            str(s.needs_author), str(s.unfinished), str(s.reason_right), str(s.stable),
            f"{s.turns:.1f}", f"{s.tool_calls:.1f}", f"{s.cost_per_run:.4f}",
        )  # fmt: skip
    console.print(table)
    for s in scores:
        console.print(
            f"[bold]{s.set}[/] 按类型：" + "、".join(f"{k} {v}" for k, v in s.by_kind.items())
        )
    wrong = [r for r in results if r.set_up and not r.right]
    if wrong:
        console.print("\n[bold]结论不对的[/]")
        for r in wrong:
            expected = "/".join(r.case.expect)
            run = str(r.run_id)[:8] if r.run_id else ""
            console.print(
                f"  {r.case.id}（{r.case.kind}，期望 {expected}）→ {r.verdict} {r.reason}"
                f"  [dim]{run}[/]\n    {escape(r.explanation or '')}"
            )
    report = {
        "label": label,
        "created_at": datetime.now().astimezone().isoformat(),
        "model": verify_settings.llm_verify_model,
        "extra": verify_settings.llm_verify_extra,
        "prompt": prompt or PROMPT_VERSION,
        "samples": samples,
        "fresh": fresh or samples > 1,
        "scores": [s.model_dump(mode="json") for s in scores],
        "results": [r.model_dump(mode="json") for r in results],
    }
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = runs_dir / f"{stamp}-verify-{_file_label(label)}.json"
    text = json.dumps(report, ensure_ascii=False, indent=1)
    path.write_text(text, "utf-8")
    console.print(f"\n结果已保存：{path}")
    if baseline:
        (EVAL_DIR / "verify-baseline.json").write_text(text, "utf-8")
        console.print(f"已保存为基线：{EVAL_DIR / 'verify-baseline.json'}")


@app.command("long-chapters")
def long_chapters_cmd(
    count: Annotated[int, typer.Option(help="用多少部 WebNovelBench（每 3 章合并成 1 章）")] = 20,
    concurrency: Annotated[int, typer.Option(min=1, help="同时处理的作品数")] = 6,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "long-chapters",
) -> None:
    """长章节（4-3）：导入合并后的长章节底稿，并与原章节的抽取结果对比。"""
    import json

    from webfic.db.session import make_engine, make_session_factory
    from webfic.evaluation import inject as ij
    from webfic.evaluation import long_chapters as lc
    from webfic.evaluation import verify_eval as ve
    from webfic.extraction.extractor import extraction_version
    from webfic.llm.cache import DbCallStore

    settings = get_settings()
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    longs = lc.long_bases(EVAL_DIR / "external", count)

    async def main():
        engine = make_engine(settings.database_url)
        factory = make_session_factory(engine)
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")

        def make_llm(book_id):
            store = _LayeredStore(
                DbCallStore(factory, user_id=ij.INJECT_USER, book_id=book_id),
                DbCallStore(cache, user_id=None),
            )
            return platform_client(settings, store)

        try:
            prepared = await ij.prepare(
                factory, make_llm, settings, [lg.base for lg in longs], concurrency=concurrency,
                on_progress=console.print,
            )  # fmt: skip
            version = extraction_version(load_prompt(), settings.chunk_size, settings.chunk_overlap)
            async with factory() as session:
                originals = await ve._books(session, ij.INJECT_USER, "@" + version[:8])
            return [
                await lc.compare(factory, lg, prepared[lg.base.name], originals[lg.original])
                for lg in longs
            ]
        finally:
            await engine.dispose()

    rows = asyncio.run(main())
    table = Table(
        "作品", "长章字数", "原章抽到", "长章抽到", "共同", "标注一致", title="长章节抽取对比"
    )
    for r in rows:
        table.add_row(
            r.name, "、".join(str(c) for c in r.long_chapter_chars), str(r.original), str(r.long),
            str(r.both), str(r.same_reading),
        )  # fmt: skip
    total = {
        k: sum(getattr(r, k) for r in rows) for k in ("original", "long", "both", "same_reading")
    }
    table.add_row(
        "全部", "", str(total["original"]), str(total["long"]), str(total["both"]),
        str(total["same_reading"]),
    )  # fmt: skip
    console.print(table)
    kept = Ratio(num=total["both"], den=total["original"])
    extra = Ratio(num=total["long"] - total["both"], den=total["long"])
    agree = Ratio(num=total["same_reading"], den=total["both"])
    console.print(
        f"原章抽到的年龄，长章也抽到：{kept}；长章多出来的：{extra}；"
        f"两边都抽到的，标注完全一致：{agree}"
    )
    ages = {
        k: sum(getattr(r, k) for r in rows)
        for k in ("original_ages", "long_ages", "both_ages", "same_ages")
    }
    console.print(
        f"只看写明数字的年龄（不含「少年」「老者」这类人生阶段）：原章 {ages['original_ages']}、"
        f"长章 {ages['long_ages']}；原章的长章也抽到 "
        f"{Ratio(num=ages['both_ages'], den=ages['original_ages'])}；长章多出来的 "
        f"{Ratio(num=ages['long_ages'] - ages['both_ages'], den=ages['long_ages'])}；"
        f"标注一致 {Ratio(num=ages['same_ages'], den=ages['both_ages'])}"
    )
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{_file_label(label)}.json"
    path.write_text(
        json.dumps(
            {"label": label, "rows": [r.model_dump() for r in rows], "totals": total},
            ensure_ascii=False, indent=1,
        ),
        "utf-8",
    )  # fmt: skip
    console.print(f"\n结果已保存：{path}")


@app.command("author-notes")
def author_notes_cmd(
    originals: Annotated[
        bool, typer.Option(help="同时在约 1000 章原文上统计误判（约 $0.2，之后走缓存）")
    ] = True,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "author-notes",
) -> None:
    """作者的话（4.5）：写好的作者注接到真实章节首尾能否识别，正文会不会被误判。"""
    import html
    import json

    from webfic.evaluation import author_notes_eval as an
    from webfic.llm.cache import DbCallStore

    settings = get_settings()
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    cases = an.build_cases(EVAL_DIR)

    async def main():
        # No database: the scans only need the model (and the eval cache).
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")
        llm = platform_client(settings, _LayeredStore(DbCallStore(cache, user_id=None)))
        results = await an.run_cases(llm, cases)
        originals_score = None
        if originals:
            originals_score = await an.run_originals(
                llm, EVAL_DIR / "external",
                on_progress=lambda d, t: console.print(f"  原文 [{d}/{t}]"),
            )  # fmt: skip
        return results, originals_score

    results, originals_score = asyncio.run(main())
    table = Table("类型", "条数", "识别出（作者注）", "边界准确", "误判（正文）", title="作者的话")
    for sc in an.score(results):
        table.add_row(sc.group, str(sc.cases), str(sc.found), str(sc.exact), str(sc.touched))
    console.print(table)
    missed = [r for r in results if r.case.kind == "note" and not r.found]
    wrong = [r for r in results if (r.case.kind == "story" and r.touched) or r.story_lost]
    for title, items in (("没识别出的作者注", missed), ("误判或多带了正文", wrong)):
        if items:
            console.print(f"\n[bold]{title}[/]")
            for r in items:
                console.print(
                    f"  {r.case.id}（{r.case.position}，{r.case.chapter}）："
                    f"{r.case.content[slice(*r.case.span)][:40]}  多带正文 {r.story_lost} 字"
                )
    report = {
        "label": label,
        "created_at": datetime.now().astimezone().isoformat(),
        "scores": [sc.model_dump(mode="json") for sc in an.score(results)],
        "results": [
            {"case": r.case.id, "found": r.found, "touched": r.touched,
             "story_lost": r.story_lost, "ranges": r.ranges}
            for r in results
        ],
    }  # fmt: skip
    if originals_score is not None:
        o = originals_score
        rate = Ratio(num=o.paragraphs_flagged, den=o.paragraphs_examined)
        console.print(
            f"\n原文 {o.chapters} 章：被判为作者的话的段落 {rate}（章首 / 章末共 "
            f"{o.paragraphs_examined} 段），涉及 {o.chapters_flagged} 章"
        )
        report["originals"] = o.model_dump(mode="json", exclude={"flagged"})
        report["originals"]["flagged"] = [f.model_dump(mode="json") for f in o.flagged]
        page = EVAL_DIR / "author_notes" / "flagged.html"
        rows = "".join(
            f"<section><h2>{html.escape(f.chapter)} {f.range[0]}–{f.range[1]}</h2>"
            f"<pre>{html.escape(f.text)}</pre></section>"
            for f in o.flagged
        )
        page.write_text(
            "<!doctype html><meta charset='utf-8'><title>被判为作者的话的原文段落</title>"
            "<style>body{font:15px/1.7 system-ui;max-width:900px;margin:auto;padding:16px}"
            "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}</style>"
            f"<h1>原文里被判为作者的话的部分（{len(o.flagged)} 处）</h1>{rows}",
            "utf-8",
        )
        console.print(f"逐条核对：{page}")
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{_file_label(label)}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    console.print(f"\n结果已保存：{path}")


@app.command("probe")
def probe_cmd(
    books: Annotated[int, typer.Option(help="抽几部 WebNovelBench（每部全部 10 章）")] = 8,
    private_chapters: Annotated[int, typer.Option(help="eval/private 每部取前几章")] = 20,
    concurrency: Annotated[int, typer.Option(min=1, help="同时处理的章节数")] = 6,
    label: Annotated[str, typer.Option(help="本次运行的说明，用于文件名")] = "probe",
) -> None:
    """试探性调查（5-1a）：真实文本里有哪些人物设定事实、同一角色写到两次以上的比例、前后说法不同的有哪些。"""
    import json

    from webfic.evaluation import probe as pb
    from webfic.evaluation.realtext import shushan_book, webnovelbench_books
    from webfic.llm.cache import DbCallStore

    settings = get_settings()
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    external, private = EVAL_DIR / "external", EVAL_DIR / "private" / "published work"
    wnb = webnovelbench_books(external / "webnovelbench" / "novel_data_subset_d_100.json")
    samples = pb.choose(
        [(b.name, b.text) for b in wnb],
        shushan_book(external / "shushan" / "shushan_001-012.txt").text,
        {f"private/{p.stem}": p.read_text("utf-8-sig") for p in sorted(private.glob("*.txt"))},
        books=books, private_chapters=private_chapters,
    )  # fmt: skip
    console.print(
        f"{len(samples)} 章（{len({s.book for s in samples})} 部），"
        f"{sum(len(s.text) for s in samples)} 字"
    )

    async def main():
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")
        return await pb.read(
            platform_client(settings, DbCallStore(cache, user_id=None)), samples,
            concurrency=concurrency,
        )  # fmt: skip

    found = asyncio.run(main())
    stats = pb.stats(found)
    table = Table(title=f"人物设定事实（{pb.PROMPT_VERSION}）")
    for column in ("类别", "条数", "现在时", "角色数", "角色·属性", "写到 ≥2 章", "说法不同"):
        table.add_column(column)
    for c in stats:
        table.add_row(
            c.category, str(c.facts), str(c.present), str(c.characters), str(c.groups),
            f"{c.repeated}（{c.repeated / c.groups:.0%}）" if c.groups else "0",
            str(c.differing),
        )  # fmt: skip
    console.print(table)
    common = "、".join(f"{c}·{a} {n}" for c, a, n in pb.attributes(found))
    console.print(f"\n最常见的属性：{common}")
    for c in stats:
        if c.examples:
            console.print(f"\n[bold]{c.category}：说法不同的例子[/]")
            for line in c.examples:
                console.print(f"  {line}")
    report = {
        "label": label, "prompt": pb.PROMPT_VERSION, "created_at": datetime.now().isoformat(),
        "samples": [{"book": s.book, "chapter": s.chapter} for s in samples],
        "stats": [c.model_dump() for c in stats],
        "by_book": {b: dict(c) for b, c in pb.by_book(found).items()},
        "facts": [f.model_dump() for f in found],
    }  # fmt: skip
    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-probe-{_file_label(label)}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    console.print(f"\n结果已保存：{path}")


@app.command("review")
def review_cmd(
    checker: Annotated[str, typer.Option(help="检查器（consistency_issues.checker）")] = (
        "character_facts"
    ),
    verify: Annotated[bool, typer.Option(help="先用校验 agent 核实还没核实过的问题")] = True,
    out: Annotated[Path | None, typer.Option(help="审阅页路径")] = None,
    sample: Annotated[int, typer.Option(help="被驳回的问题里抽几条请人核对（0：全部列出）")] = 30,
) -> None:
    """原文报告审阅页（5-1c）：注入底稿上某个检查器报出的问题 + 校验 agent 的结论，供人工核对。"""
    from webfic.archival.index import platform_archival, reindex_book
    from webfic.evaluation import inject as ij
    from webfic.evaluation import review as rv
    from webfic.evaluation import verify_eval as ve
    from webfic.llm.cache import DbCallStore
    from webfic.services.verification import verify_issues

    settings = get_settings()
    try:
        platform_client(settings)
    except LLMNotConfigured as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    path = out or EVAL_DIR / "external" / "inject" / f"review-{checker.replace('_', '-')}.html"

    async def main():
        from sqlalchemy import select

        from webfic.db.models import IssueRow
        from webfic.db.session import make_engine, make_session_factory

        engine = make_engine(settings.database_url)
        factory = make_session_factory(engine)
        cache = await open_cache(EVAL_DIR / ".cache" / "llm.sqlite")
        version = load_prompt_version(settings)
        try:
            async with factory() as session:
                books = await ve._books(session, ij.INJECT_USER, "@" + version[:8])
                ids = {
                    name: list(
                        await session.scalars(
                            select(IssueRow.id).where(
                                IssueRow.user_id == ij.INJECT_USER,
                                IssueRow.book_id == book_id,
                                IssueRow.checker == checker,
                                IssueRow.status.in_(["open", "acknowledged"]),
                            )
                        )
                    )
                    for name, book_id in books.items()
                }
            cost = 0
            if verify:
                archival = platform_archival(settings)
                for name, issue_ids in sorted(ids.items()):
                    if not issue_ids:
                        continue
                    book_id = books[name]
                    await reindex_book(factory, archival, user_id=ij.INJECT_USER, book_id=book_id)
                    store = _LayeredStore(
                        DbCallStore(factory, user_id=ij.INJECT_USER, book_id=book_id),
                        DbCallStore(cache, user_id=None),
                    )
                    result = await verify_issues(
                        factory, platform_client(settings, store), settings,
                        user_id=ij.INJECT_USER, book_id=book_id, issue_ids=issue_ids,
                        archival=archival,
                    )  # fmt: skip
                    cost += result.cost_usd
                    console.print(f"{name}：核实 {len(result.verified)} 条，${result.cost_usd:.4f}")
            items = await rv.collect(factory, books, ij.INJECT_USER, checker)
            return items, cost
        finally:
            await engine.dispose()

    items, cost = asyncio.run(main())
    shown, rest = rv.choose(items, sample) if sample else (items, [])
    kept = sum(
        1 for it in items if not (it.verification and it.verification.verdict == "false_alarm")
    )
    intro = (
        f"注入底稿（{len({it.book for it in items})} 部有报告）上检查器 {checker} 共报出 "
        f"{len(items)} 条，校验 agent 保留 {kept} 条、驳回 {len(items) - kept} 条。"
        f"下面是保留的 {kept} 条和按类型抽样的 {len(shown) - kept} 条被驳回的（固定种子）。"
        "请逐条判断：真矛盾 / 误报 / 存疑（写编号和判断即可）。你的判断给出这一类检查在真实文本上的"
        "精确率和校验 agent 误驳的比例，也会作为校验 agent 的保留集标注。"
    )
    page = rv.page(shown, title=f"原文报告核对：{checker}", intro=intro, rest=rest)
    path.write_text(page, "utf-8")
    console.print(f"{len(items)} 条；核实花费 ${cost:.4f}；审阅页：{path}")


def load_prompt_version(settings) -> str:
    """The age-extraction version tag the base books' titles carry."""
    from webfic.extraction.extractor import extraction_version

    return extraction_version(load_prompt(), settings.chunk_size, settings.chunk_overlap)


if __name__ == "__main__":
    app()
