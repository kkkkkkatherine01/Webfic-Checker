"""Original-text reports for a person to judge (step 5-1c; step 7 of the procedure in
docs/steps/05-more-checkers.md A7): every open issue one checker reports on the
evaluation base texts, with the verify agent's verdict, as an HTML page. The person's
judgement is the real precision of a new kind of check; it becomes the holdout labels of
the verify agent.
"""

import html
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from webfic.agent.verify import REASON_LABEL, VERDICT_LABEL
from webfic.db.models import Chapter, IssueRow
from webfic.services.verification import VerificationView, current_verifications

Factory = async_sessionmaker[AsyncSession]
CONTEXT = 220  # characters shown around each quote

_TYPE = {
    "fact.appearance": "外貌前后不一",
    "character.kinship": "亲属关系矛盾",
    "timeline.revival": "死后出场",
    "character.age": "年龄",
}


@dataclass
class Item:
    book: str
    issue: IssueRow
    contexts: list[tuple[int, str, int, int]]  # chapter, text, mark start, mark end
    verification: VerificationView | None


async def collect(
    factory: Factory, books: dict[str, uuid.UUID], user_id: uuid.UUID, checker: str
) -> list[Item]:
    """The open issues of `checker` on `books` (name -> id), with context around their
    evidence and their current verdicts."""
    items = []
    async with factory() as session:
        for name, book_id in sorted(books.items()):
            issues = (
                await session.scalars(
                    select(IssueRow)
                    .where(
                        IssueRow.user_id == user_id,
                        IssueRow.book_id == book_id,
                        IssueRow.checker == checker,
                        IssueRow.status.in_(["open", "acknowledged"]),
                    )
                    .order_by(IssueRow.fingerprint)
                )
            ).all()
            if not issues:
                continue
            verdicts = await current_verifications(session, user_id, book_id, list(issues))
            contents = dict(
                (
                    await session.execute(
                        select(Chapter.number, Chapter.content).where(
                            Chapter.user_id == user_id, Chapter.book_id == book_id
                        )
                    )
                ).all()
            )
            for issue in issues:
                contexts = []
                for e in issue.evidence:
                    text = contents.get(e["chapter_number"], "")
                    a = max(0, e["char_start"] - CONTEXT)
                    b = min(len(text), e["char_end"] + CONTEXT)
                    contexts.append(
                        (e["chapter_number"], text[a:b], e["char_start"] - a, e["char_end"] - a)
                    )
                items.append(Item(name, issue, contexts, verdicts.get(issue.id)))
    return items


_STYLE = """
:root{--bg:#fbfaf7;--fg:#222;--mute:#666;--card:#fff;--line:#e3e0d8;--mark:#ffe38a;
--t:#1f6f43;--f:#a3322b;--m:#8a6d00}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1d;--fg:#e6e6e6;--mute:#9a9a9a;
--card:#242428;--line:#3a3a40;--mark:#6b5a12;--t:#6fcf97;--f:#ef7b72;--m:#e2c15a}}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:12px 16px;margin:16px 0}
h2{font-size:17px;margin:4px 0} .conf{font-size:13px;color:var(--mute);font-weight:normal}
.desc{margin:4px 0} .ev li{margin:6px 0;word-break:break-all;color:var(--mute)}
.ch{color:var(--fg);margin-right:6px} mark{background:var(--mark);color:var(--fg)}
.verdict{border-top:1px dashed var(--line);padding-top:8px}
.contradiction b{color:var(--f)} .false_alarm b{color:var(--t)} .needs_author b{color:var(--m)}
table{border-collapse:collapse} td,th{border:1px solid var(--line);padding:2px 10px}
"""


def choose(items: list[Item], sample: int, seed: int = 7) -> tuple[list[Item], list[Item]]:
    """What a person checks: every issue the agent kept (or could not settle), and a
    seeded sample of those it dismissed, spread over the issue types in proportion; the
    rest of the dismissed ones, for reference."""
    import random

    kept = [i for i in items if not (i.verification and i.verification.verdict == "false_alarm")]
    dismissed = [i for i in items if i not in kept]
    rng = random.Random(seed)
    by_type: dict[str, list[Item]] = {}
    for it in dismissed:
        by_type.setdefault(it.issue.issue_type, []).append(it)
    chosen: list[Item] = []
    for _, group in sorted(by_type.items()):
        n = round(sample * len(group) / len(dismissed)) if dismissed else 0
        chosen += rng.sample(group, min(n, len(group)))
    rest = [i for i in dismissed if i not in chosen]
    return kept + chosen, rest


def page(items: list[Item], *, title: str, intro: str, rest: list[Item] | None = None) -> str:
    counts: dict[str, int] = {}
    for it in items:
        key = it.verification.verdict if it.verification and it.verification.verdict else "none"
        counts[key] = counts.get(key, 0) + 1
    summary = "".join(
        f"<tr><td>{html.escape(VERDICT_LABEL.get(k, '未核实 / 未完成'))}</td><td>{n}</td></tr>"
        for k, n in sorted(counts.items())
    )
    parts = [
        f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p>{intro}</p>"
        f"<table><tr><th>校验 agent 的结论</th><th>条数</th></tr>{summary}</table>"
    ]
    parts.append(_sections(items))
    if rest:
        parts.append(
            f"<details><summary>另外 {len(rest)} 条被校验 agent 驳回的问题（不必核对，供参考）"
            f"</summary>{_sections(rest, start=len(items) + 1)}</details>"
        )
    parts.append("</main></body></html>")
    return "".join(parts)


def _sections(items: list[Item], start: int = 1) -> str:
    parts: list[str] = []
    for n, it in enumerate(items, start=start):
        i, v = it.issue, it.verification
        evidence = "".join(
            f'<li><span class="ch">第 {ch} 章</span>…{html.escape(t[:a])}'
            f"<mark>{html.escape(t[a:b])}</mark>{html.escape(t[b:])}…</li>"
            for ch, t, a, b in it.contexts
        )
        if v is not None and v.verdict:
            reason = REASON_LABEL.get(v.reason or "", v.reason or "")
            verdict = (
                f'<p class="verdict {v.verdict}"><b>校验 agent：{VERDICT_LABEL[v.verdict]}</b>'
                f"（{html.escape(reason)}）{html.escape(v.explanation or '')}"
                f' <span class="conf">webfic trace {str(v.run_id)[:8]}</span></p>'
            )
        else:
            verdict = '<p class="verdict"><b>校验 agent：未核实 / 核实未完成</b></p>'
        parts.append(
            f"<section><h2>{n}. {html.escape(it.book)} "
            f'<span class="conf">{html.escape(_TYPE.get(i.issue_type, i.issue_type))} · '
            f"#{i.fingerprint[:8]}</span></h2>"
            f'<p class="desc">{html.escape(i.description)}</p><ol class="ev">{evidence}</ol>'
            f"{verdict}</section>"
        )
    return "".join(parts)
