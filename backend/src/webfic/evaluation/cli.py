"""webfic-eval: score the pipeline against the golden set."""

import asyncio
import hashlib
import subprocess
from datetime import datetime
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
    _print_details(overall)

    runs_dir = EVAL_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{report.meta.created_at:%Y%m%d-%H%M%S}-{label}.json"
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
    dqa_questions, dqa_clues, skipped = (
        rv.detectiveqa_questions(dqa_folder) if "detectiveqa" in corpora else ([], [], 0)
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
    path = runs_dir / f"{report.created_at:%Y%m%d-%H%M%S}-retrieval-{label}.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    console.print(f"\n结果已保存：{path}")


if __name__ == "__main__":
    app()
