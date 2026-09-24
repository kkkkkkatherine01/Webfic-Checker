"""Command-line entry point. A thin shell: every command calls one service function."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.checkers.types import Confidence
from webfic.config import Settings, get_settings
from webfic.db.session import make_engine, make_session_factory
from webfic.llm.base import ProviderError
from webfic.llm.cache import DbCallStore
from webfic.llm.factory import LLMNotConfigured, platform_client
from webfic.memory import core
from webfic.services import chapters, checks, imports, reports
from webfic.services.errors import InvalidEdit, NotFound

app = typer.Typer(help="网文一致性检查工具", no_args_is_help=True, add_completion=False)
console = Console()

_CONFIDENCE_LABEL = {
    Confidence.CONFIRMED: "[bold red]确定矛盾[/]",
    Confidence.SUSPECTED_REVIEW: "[yellow]疑似[/]",
    Confidence.INSUFFICIENT_INFO: "[dim]信息不足[/]",
}

Factory = async_sessionmaker[AsyncSession]


def _run[T](fn: Callable[[Settings, Factory], Awaitable[T]]) -> T:
    settings = get_settings()

    async def main() -> T:
        engine = make_engine(settings.database_url)
        try:
            return await fn(settings, make_session_factory(engine))
        finally:
            await engine.dispose()

    try:
        return asyncio.run(main())
    except (NotFound, InvalidEdit, LLMNotConfigured) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    except ProviderError as exc:  # only auth errors reach here; the rest fail per chapter
        console.print(
            f"[red]模型服务商拒绝了请求，已停止：{exc.message}\n请检查 API Key 和账户余额。[/]"
        )
        raise typer.Exit(1) from exc
    except (OSError, DBAPIError) as exc:
        console.print(
            f"[red]无法连接数据库（{settings.database_url.split('@')[-1]}）："
            f"请先运行 docker compose up -d postgres，并执行 uv run alembic upgrade head。[/]\n"
            f"[dim]{type(exc).__name__}: {exc}[/]"
        )
        raise typer.Exit(1) from exc


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030"):  # Chinese .txt files are often GBK
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise typer.BadParameter(f"无法识别文件编码：{path}")


async def _resolve_book(factory: Factory, user_id: uuid.UUID, book: str) -> uuid.UUID:
    """Accept a full id or a unique prefix of one."""
    async with factory() as session:
        books = await reports.list_books(session, user_id=user_id)
    matches = [b.id for b in books if str(b.id).startswith(book.lower())]
    if len(matches) != 1:
        raise NotFound(f"找不到唯一匹配「{book}」的作品（匹配到 {len(matches)} 个）")
    return matches[0]


async def _extract(settings: Settings, factory: Factory, book_id: uuid.UUID) -> None:
    user_id = settings.dev_user_id
    llm = platform_client(settings, DbCallStore(factory, user_id=user_id, book_id=book_id))

    with Progress(
        TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console
    ) as progress:
        task = progress.add_task("抽取中", total=None)

        def on_progress(event: imports.ProgressEvent) -> None:
            progress.update(task, total=event.total, completed=event.done,
                            description=f"抽取 {event.chapter_title}")  # fmt: skip
            if event.status == "failed":
                progress.console.print(f"[red]✗ {event.chapter_title} 失败：{event.error}[/]")

        result = await imports.run_import_job(
            factory, llm, settings, user_id=user_id, book_id=book_id, on_progress=on_progress
        )

    console.print(
        f"抽取完成：成功 {result.extracted} 章，失败 {result.failed} 章"
        f"{_failure_summary(result.failures)}；"
        f"丢弃无法定位/归属的表述 {result.dropped_statements} 条。\n"
        f"LLM 调用 {result.llm_calls} 次（缓存命中 {result.cache_hits}），"
        f"输入 {result.input_tokens} / 输出 {result.output_tokens} tokens，"
        f"费用约 ${result.cost_usd:.4f}"
    )


_FAILURE_LABEL = {
    "content_filter": "被服务商内容审核拒绝",
    "rate_limit": "请求过于频繁",
    "server": "服务商故障",
    "network": "网络问题",
    "bad_request": "请求无效",
    "invalid_output": "模型输出无法解析",
}


def _failure_summary(failures: dict[str, int]) -> str:
    if not failures:
        return ""
    parts = [f"{_FAILURE_LABEL.get(k, k)} {n} 章" for k, n in failures.items()]
    return "（" + "，".join(parts) + "）"


async def _check(settings: Settings, factory: Factory, book_id: uuid.UUID) -> None:
    async with factory() as session:
        result = await checks.run_checks(session, user_id=settings.dev_user_id, book_id=book_id)
    console.print(
        f"检查完成：发现 {result.found} 个问题（新增 {result.new}），"
        f"{result.resolved} 个旧问题已不再出现。"
    )


@app.command()
def ingest(
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="整本 txt")],
    title: Annotated[str | None, typer.Option(help="作品名，默认用文件名")] = None,
    check: Annotated[bool, typer.Option(help="抽取后立即运行检查")] = True,
) -> None:
    """导入作品：切章 → 抽取 → （检查）。"""
    text = _read_text(file)

    async def main(settings: Settings, factory: Factory) -> None:
        platform_client(settings)  # fail fast before storing anything if no API key
        async with factory() as session:
            job = await imports.create_import_job(
                session, user_id=settings.dev_user_id, title=title or file.stem, text=text
            )
        for warning in job.warnings:
            console.print(f"[yellow]⚠ {warning}[/]")
        console.print(
            f"作品 ID：[bold]{job.book_id}[/]，共 {len(job.chapters)} 章，{job.total_chars} 字"
        )
        await _extract(settings, factory, job.book_id)
        if check:
            await _check(settings, factory, job.book_id)

    _run(main)


@app.command()
def resume(book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")]) -> None:
    """继续抽取未完成或失败的章节。"""

    async def main(settings: Settings, factory: Factory) -> None:
        book_id = await _resolve_book(factory, settings.dev_user_id, book)
        await _extract(settings, factory, book_id)

    _run(main)


@app.command("check")
def check_cmd(book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")]) -> None:
    """对已抽取的内容运行一致性检查。"""

    async def main(settings: Settings, factory: Factory) -> None:
        await _check(settings, factory, await _resolve_book(factory, settings.dev_user_id, book))

    _run(main)


@app.command()
def report(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    all_: Annotated[bool, typer.Option("--all", help="包含已解决/标记为刻意设计的问题")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="输出 JSON")] = False,
) -> None:
    """查看矛盾报告。"""

    async def main(settings: Settings, factory: Factory) -> None:
        book_id = await _resolve_book(factory, settings.dev_user_id, book)
        async with factory() as session:
            rep = await reports.get_report(
                session, user_id=settings.dev_user_id, book_id=book_id, include_closed=all_
            )
        if as_json:
            print(rep.model_dump_json(indent=2))
            return
        console.print(f"[bold]《{rep.title}》[/] 共 {len(rep.issues)} 个问题\n")
        for n, issue in enumerate(rep.issues, start=1):
            console.print(f"{n}. {_CONFIDENCE_LABEL[issue.confidence]}  {issue.description}")
            for e in issue.evidence:
                console.print(f"     [dim]第 {e.chapter_number} 章[/] 「{e.quote}」")
            console.print()

    _run(main)


# --- chapter management ----------------------------------------------------------------

DryRun = Annotated[bool, typer.Option("--dry-run", help="试算：报告会新增 / 消失的矛盾，不保存")]


def _print_issues(title: str, issues: list[reports.IssueView]) -> None:
    if not issues:
        return
    console.print(f"\n[bold]{title}[/]（{len(issues)}）")
    for issue in issues:
        console.print(f"  {_CONFIDENCE_LABEL[issue.confidence]}  {issue.description}")
        for e in issue.evidence:
            console.print(f"     [dim]第 {e.chapter_number} 章[/] 「{e.quote}」")


def _print_change(result: chapters.ChangeResult) -> None:
    for warning in result.warnings:
        console.print(f"[yellow]⚠ {warning}[/]")
    mode = "[bold yellow]试算（未保存）[/]" if result.dry_run else "[bold green]已保存[/]"
    console.print(
        f"{mode}：全书 {result.chapters} 章；重算 {result.extracted} 章"
        f"（其中 {result.reused} 章原文未改、复用上次的抽取结果；失败 {result.failed}），"
        f"LLM 调用 {result.llm_calls} 次"
        f"（缓存命中 {result.cache_hits}），费用约 ${result.cost_usd:.4f}"
    )
    _print_issues("会新增的矛盾" if result.dry_run else "新增的矛盾", result.issues_added)
    _print_issues("会消失的矛盾" if result.dry_run else "消失的矛盾", result.issues_removed)
    if not result.issues_added and not result.issues_removed:
        console.print("矛盾报告没有变化。")


def _change(book: str, operation: Callable[..., Awaitable[chapters.ChangeResult]], **kwargs):
    """Run a chapter operation with the platform LLM and print what changed."""

    async def main(settings: Settings, factory: Factory) -> None:
        user_id = settings.dev_user_id
        book_id = await _resolve_book(factory, user_id, book)
        llm = platform_client(settings, DbCallStore(factory, user_id=user_id, book_id=book_id))
        with console.status("处理中（原文未改的章节复用上次的抽取结果）…"):
            result = await operation(
                factory, llm, settings, user_id=user_id, book_id=book_id, **kwargs
            )
        _print_change(result)

    _run(main)


@app.command()
def append(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="新章节 txt")],
    dry_run: DryRun = False,
) -> None:
    """在作品末尾追加一章或多章。"""
    _change(book, chapters.append_chapters, text=_read_text(file), dry_run=dry_run)


@app.command("replace-chapter")
def replace_chapter(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    number: Annotated[int, typer.Argument(help="章号（按导入后的顺序，从 1 开始）")],
    file: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="新正文 txt")],
    dry_run: DryRun = False,
) -> None:
    """用新正文替换第 N 章，并从第 N 章起重算。"""
    _change(
        book, chapters.replace_chapter, number=number, content=_read_text(file), dry_run=dry_run
    )


@app.command("delete-chapter")
def delete_chapter(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    number: Annotated[int, typer.Argument(help="章号")],
    dry_run: DryRun = False,
) -> None:
    """删除第 N 章，后面的章节编号前移并重算。"""
    _change(book, chapters.delete_chapter, number=number, dry_run=dry_run)


@app.command()
def patch(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    number: Annotated[int, typer.Argument(help="章号")],
    old: Annotated[str, typer.Argument(help="原片段（在这一章里必须唯一）")],
    new: Annotated[str, typer.Argument(help="新片段")],
    dry_run: DryRun = False,
) -> None:
    """把第 N 章里的一段原文改成新片段，并从第 N 章起重算。"""
    _change(book, chapters.patch_chapter, number=number, old=old, new=new, dry_run=dry_run)


def _fmt_age(low: float | None, high: float | None) -> str:
    if low is None:
        return "—"
    low, high = round(low, 1), round(high if high is not None else low, 1)  # months add decimals
    text = f"{low:g}" if high == low else f"{low:g}–{high:g}"
    return f"{text} 岁"


@app.command()
def character(
    book: Annotated[str, typer.Argument(help="作品 ID（可只写前几位）")],
    name: Annotated[str, typer.Argument(help="角色名或别名")],
    chapter: Annotated[int | None, typer.Option(help="查看截至第几章的状态")] = None,
) -> None:
    """查看角色当前状态（年龄、人生阶段、别名）。"""

    async def main(settings: Settings, factory: Factory) -> None:
        book_id = await _resolve_book(factory, settings.dev_user_id, book)
        async with factory() as session:
            view = await core.get_character(
                session, user_id=settings.dev_user_id, book_id=book_id, name=name,
                as_of_chapter=chapter,
            )  # fmt: skip
        console.print(f"[bold]{view.canonical_name}[/]（截至第 {view.as_of_chapter} 章）")
        console.print(f"  别名：{'、'.join(view.aliases) or '—'}")
        if view.age_low is not None:
            console.print(
                f"  最近写明的年龄：{_fmt_age(view.age_low, view.age_high)}"
                f"（第 {view.age_chapter} 章「{view.age_quote}」）"
            )
            estimate = _fmt_age(view.estimated_age_low, view.estimated_age_high)
            note = (
                "" if view.estimated_age_low is not None else "（中间有无法量化的时间跳跃，不推算）"
            )
            console.print(f"  推算此时：{estimate}{note}")
        else:
            console.print("  年龄：原文没有写明")
        if view.life_stage is not None:
            console.print(f"  人生阶段：{view.life_stage}（第 {view.life_stage_chapter} 章）")

    _run(main)


@app.command()
def usage(
    book: Annotated[str | None, typer.Argument(help="作品 ID；不填则统计全部")] = None,
) -> None:
    """查看 LLM 用量与费用。"""

    async def main(settings: Settings, factory: Factory) -> None:
        book_id = await _resolve_book(factory, settings.dev_user_id, book) if book else None
        async with factory() as session:
            summary = await reports.get_usage(
                session, user_id=settings.dev_user_id, book_id=book_id
            )
        table = Table("用途", "模型", "调用", "缓存命中", "输入", "其中缓存", "输出", "费用 $")
        for line in summary.lines:
            table.add_row(
                line.purpose, line.model, str(line.calls), str(line.cache_hits),
                str(line.input_tokens), str(line.cached_input_tokens), str(line.output_tokens),
                f"{line.cost_usd:.4f}",
            )  # fmt: skip
        console.print(table)
        console.print(f"合计：${summary.total_cost_usd:.4f}")

    _run(main)


@app.command()
def books() -> None:
    """列出所有作品。"""

    async def main(settings: Settings, factory: Factory) -> None:
        async with factory() as session:
            items = await reports.list_books(session, user_id=settings.dev_user_id)
        table = Table("ID", "作品", "章节", "已抽取", "失败", "角色", "导入时间")
        for b in items:
            table.add_row(
                str(b.id)[:8], b.title, str(b.chapters), str(b.extracted), str(b.failed),
                str(b.characters), b.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            )  # fmt: skip
        console.print(table)

    _run(main)


if __name__ == "__main__":
    app()
