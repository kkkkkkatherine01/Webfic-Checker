"""The verify agent: given one issue the rule checkers reported, go back to the text and
decide whether it is a real contradiction, a false alarm, or something only the author
can settle, with the passages that show it.

Its tools are the memory queries of step 3, each returning a compact view (a model pays
for every field on every later turn). It cannot change anything: its verdict only
decides how the report shows the issue.
"""

from importlib import resources
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from webfic.agent.tools import RunState, Tool, ToolArgs, ToolContext, ToolRegistry
from webfic.checkers.age import CHECKER_NAME as AGE_CHECKER
from webfic.checkers.character_facts import CHECKER_NAME as CHARACTER_FACTS_CHECKER
from webfic.db.models import Chapter, ChapterNoteRow
from webfic.extraction.locator import locate
from webfic.facts.describe import describe_span, fmt_range
from webfic.facts.registry import AGE, APPEARANCE, KINSHIP, LIFE, describe_fact
from webfic.memory import archival as archival_memory
from webfic.memory import core, recall

PROMPT_VERSION = "verify_v2"
# The prompt that verifies each checker's issues (step 5-0). Ages keep verify_v2 word for
# word; a new checker brings its own prompt and reasons (docs/steps/05-more-checkers.md).
PROMPTS = {AGE_CHECKER: PROMPT_VERSION, CHARACTER_FACTS_CHECKER: "verify_facts_v1"}

Verdict = Literal["contradiction", "false_alarm", "needs_author"]
Reason = Literal[
    "misattributed",
    "generic",
    "past_event",
    "flashback",
    "speculative",
    "time_span",
    "value",
    "genuine",
    "ambiguous",
    "other",
]

VERDICT_LABEL = {
    "contradiction": "真矛盾",
    "false_alarm": "误报",
    "needs_author": "需作者确认",
}
REASON_LABEL = {
    "misattributed": "认错人",
    "generic": "泛指 / 规则 / 群体 / 比喻",
    "past_event": "过去某件事发生时的年龄",
    "flashback": "回忆被当成现在",
    "speculative": "推测 / 传闻 / 外表印象",
    "time_span": "时间段理解错误",
    "value": "数值理解错误",
    "genuine": "确实矛盾",
    "ambiguous": "原文有歧义",
    "other": "其他",
}

MAX_CONTEXT = 1500
# One read returns at most this much text: the requested span is cut to MAX_SPAN and the
# context around it shrunk to fit. Reads measured in 4-3: median 709, 99% under 1400 chars;
# the cap only stops runaway requests.
MAX_READ = 3000
MAX_SPAN = 2000
MAX_SEARCH = 5  # passages per search (500 characters each)
MAX_ITEMS = 40


def prompt_for(checker: str) -> str:
    """The prompt version for a checker's issues."""
    if checker not in PROMPTS:
        raise ValueError(f"no verify prompt for checker {checker}")
    return PROMPTS[checker]


def load_prompt(name: str = PROMPT_VERSION) -> str:
    return resources.files("webfic.agent.prompts").joinpath(f"{name}.txt").read_text("utf-8")


# --- tools ---------------------------------------------------------------------------------


class ReadPassageArgs(ToolArgs):
    chapter: int = Field(description="章号")
    start: int = Field(description="起始位置（章内字符偏移，证据里给出的位置）")
    end: int = Field(description="结束位置")
    context: int = Field(default=300, ge=0, le=MAX_CONTEXT, description="前后各多读多少字")

    @model_validator(mode="after")
    def _ordered(self) -> "ReadPassageArgs":
        if self.end < self.start:
            raise ValueError("end 不能小于 start")
        return self


async def read_passage(ctx: ToolContext, args: ReadPassageArgs) -> str:
    end = min(args.end, args.start + MAX_SPAN)
    context = min(args.context, (MAX_READ - (end - args.start)) // 2)
    async with ctx.factory() as session:
        shown = await archival_memory.read_passage(
            session, user_id=ctx.user_id, book_id=ctx.book_id, chapter=args.chapter,
            start=args.start, end=end, context=context,
        )  # fmt: skip
    cut = f"（请求的范围超过 {MAX_SPAN} 字，只返回前 {MAX_SPAN} 字）" if end < args.end else ""
    async with ctx.factory() as session:
        notes = (await _note_ranges(session, ctx, [shown.chapter_number])).get(
            shown.chapter_number, []
        )
    inside = [(a, b) for a, b in notes if a < shown.char_end and shown.char_start < b]
    if inside:
        cut += "（其中 " + "、".join(f"{a}–{b}" for a, b in inside) + " 是作者的话，不是正文）"
    t = shown.text
    marked = t[: shown.focus_start] + "【" + t[shown.focus_start : shown.focus_end] + "】"
    return (
        f"第 {shown.chapter_number} 章「{shown.chapter_title}」{shown.char_start}–{shown.char_end}"
        f"（【】内是请求的位置）{cut}：\n{marked}{t[shown.focus_end :]}"
    )


class SearchTextArgs(ToolArgs):
    query: str = Field(description="要找的内容，写成具体的事实或原文措辞，如「周淳 儿子 十八岁」")
    character: str | None = Field(default=None, description="只看提到这个角色（名字或别名）的段落")
    first_chapter: int | None = Field(default=None, description="章节范围起点")
    last_chapter: int | None = Field(default=None, description="章节范围终点")
    k: int = Field(default=5, ge=1, le=MAX_SEARCH, description="返回几段")


class PassageBrief(BaseModel):
    chapter: int
    start: int
    end: int
    text: str
    author_note: bool = False  # the passage lies in the author's notes, not the story


async def search_text(ctx: ToolContext, args: SearchTextArgs) -> list[PassageBrief]:
    if ctx.archival is None:
        raise ValueError("原文检索不可用（这部作品没有建检索索引）；请用 read_passage")
    chapters = None
    if args.first_chapter is not None or args.last_chapter is not None:
        chapters = (args.first_chapter or 1, args.last_chapter or 100_000)
    async with ctx.factory() as session:
        hits = await archival_memory.search_text(
            session, ctx.archival, user_id=ctx.user_id, book_id=ctx.book_id, query=args.query,
            k=args.k, character=args.character, chapters=chapters,
        )  # fmt: skip
        notes = await _note_ranges(session, ctx, [h.chapter_number for h in hits])

    def noted(h) -> bool:
        return any(
            a <= h.char_start and h.char_end <= b for a, b in notes.get(h.chapter_number, [])
        )

    return [PassageBrief(chapter=h.chapter_number, start=h.char_start, end=h.char_end, text=h.text,
                         author_note=noted(h))
            for h in hits]  # fmt: skip


class CharacterArgs(ToolArgs):
    name: str = Field(description="角色名或别名")
    as_of_chapter: int | None = Field(default=None, description="查看截至第几章的状态")


class CharacterBrief(BaseModel):
    name: str
    aliases: list[str]
    as_of_chapter: int
    stated_age: str | None  # the last present-time age the text states, with its source
    estimated_age: str | None  # carried forward by the time spans (inherits their errors)
    life_stage: str | None


async def get_character(ctx: ToolContext, args: CharacterArgs) -> CharacterBrief:
    async with ctx.factory() as session:
        v = await core.get_character(
            session, user_id=ctx.user_id, book_id=ctx.book_id, name=args.name,
            as_of_chapter=args.as_of_chapter,
        )  # fmt: skip
    stated = None
    if v.age_low is not None:
        stated = f"{fmt_range(v.age_low, v.age_high)} 岁（第 {v.age_chapter} 章「{v.age_quote}」）"
    estimated = None
    if v.estimated_age_low is not None:
        estimated = (
            f"{fmt_range(v.estimated_age_low, v.estimated_age_high)} 岁（按时间段推算，仅供参考）"
        )
    stage = f"{v.life_stage}（第 {v.life_stage_chapter} 章）" if v.life_stage else None
    return CharacterBrief(
        name=v.canonical_name, aliases=v.aliases, as_of_chapter=v.as_of_chapter,
        stated_age=stated, estimated_age=estimated, life_stage=stage,
    )  # fmt: skip


class FactsArgs(ToolArgs):
    character: str = Field(description="角色名或别名")
    first_chapter: int | None = Field(default=None)
    last_chapter: int | None = Field(default=None)


class FactBrief(BaseModel):
    chapter: int
    start: int
    end: int
    text: str  # the quote
    mention: str  # how the text refers to the character there
    reading: str  # what extraction made of it


async def list_facts(ctx: ToolContext, args: FactsArgs) -> list[FactBrief] | str:
    async with ctx.factory() as session:
        # Ages only: the age prompt's tool, whose results must not change as other
        # kinds of facts are added (step 5-1).
        facts = await recall.list_facts(
            session, user_id=ctx.user_id, book_id=ctx.book_id, character=args.character,
            category=AGE.name, chapters=_chapters(args.first_chapter, args.last_chapter),
        )  # fmt: skip
    briefs = [
        FactBrief(chapter=f.chapter_number, start=f.char_start, end=f.char_end, text=f.raw_text,
                  mention=f.mention, reading=describe_fact(f))
        for f in facts
    ]  # fmt: skip
    return _limited(briefs)


class SpansArgs(ToolArgs):
    first_chapter: int = Field(description="章节范围起点")
    last_chapter: int = Field(description="章节范围终点")


class SpanBrief(BaseModel):
    chapter: int
    start: int
    text: str
    reading: str


async def list_time_spans(ctx: ToolContext, args: SpansArgs) -> list[SpanBrief] | str:
    async with ctx.factory() as session:
        spans = await recall.list_time_spans(
            session, user_id=ctx.user_id, book_id=ctx.book_id,
            chapters=(args.first_chapter, args.last_chapter),
        )  # fmt: skip
    briefs = [
        SpanBrief(chapter=s.chapter_number, start=s.char_start, text=s.raw_text,
                  reading=describe_span(s.kind, s.estimated_years, s.is_flashback))
        for s in spans
    ]  # fmt: skip
    return _limited(briefs)


async def _note_ranges(
    session: AsyncSession, ctx: ToolContext, numbers: list[int]
) -> dict[int, list[tuple[int, int]]]:
    """The author's-note ranges of some chapters (step 4.5)."""
    rows = await session.execute(
        select(Chapter.number, ChapterNoteRow.ranges)
        .join(ChapterNoteRow, ChapterNoteRow.chapter_id == Chapter.id)
        .where(
            Chapter.user_id == ctx.user_id,
            Chapter.book_id == ctx.book_id,
            Chapter.number.in_(sorted(set(numbers))),
        )
    )
    return {number: [(a, b) for a, b in ranges] for number, ranges in rows}


# --- the verdict ---------------------------------------------------------------------------


class Quote(BaseModel):
    chapter: int = Field(description="章号")
    quote: str = Field(description="逐字摘自该章原文的片段")


class VerdictArgs(ToolArgs):
    verdict: Verdict = Field(
        description="contradiction：真矛盾；false_alarm：误报；needs_author：原文有歧义，只能由作者判断"
    )
    reason: Reason = Field(description="判断依据的类型，见说明")
    explanation: str = Field(description="给作者看的一两句中文说明")
    evidence: list[Quote] = Field(description="支持结论的原文，至少一条")


READING_TOOLS = ("read_passage", "search_text")


async def check_verdict(ctx: ToolContext, args: VerdictArgs, state: RunState) -> str | None:
    """The code guard on a submitted verdict (A4): quoted evidence must exist in the
    text, a dismissal needs the text to have been read, and the reason must fit."""
    if args.verdict == "contradiction" and args.reason != "genuine":
        return "判为真矛盾时 reason 应为 genuine。"
    if args.verdict != "contradiction" and args.reason == "genuine":
        return "reason 为 genuine 时结论应为真矛盾。"
    if args.verdict == "false_alarm" and not any(t in state.tools_used for t in READING_TOOLS):
        return "判为误报之前必须读过原文（read_passage 或 search_text），不能只凭输入的摘要驳回。"
    if not args.evidence:
        return "evidence 至少要有一条原文。"
    missing = []
    async with ctx.factory() as session:
        for q in args.evidence:
            content = await session.scalar(
                select(Chapter.content).where(
                    Chapter.user_id == ctx.user_id,
                    Chapter.book_id == ctx.book_id,
                    Chapter.number == q.chapter,
                )
            )
            if content is None or locate(content, q.quote) is None:
                missing.append(f"第 {q.chapter} 章「{q.quote}」")
    if missing:
        return f"这些引文在原文中找不到：{'；'.join(missing)}。引文必须逐字摘自所写章节的原文。"
    return None


def verify_tools() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool("read_passage", "读取某章某个位置前后的原文。核实时首先用它读证据的上下文。",
                 ReadPassageArgs, run=read_passage),
            Tool("search_text", "在全书原文中检索段落（向量 + 关键词），可限定角色和章节范围。"
                 "用来找角色的其他描写、身份线索；查询写成具体事实，不要写成抽象问题。",
                 SearchTextArgs, run=search_text),
            Tool("get_character", "查角色的主名、别名、最近写明的年龄及出处、人生阶段。",
                 CharacterArgs, run=get_character),
            Tool("list_facts", "列出某角色在原文中被抽取出的全部年龄表述"
                 "（章节、位置、原文、抽取结果）。", FactsArgs, run=list_facts),
            Tool("list_time_spans", "列出某个章节范围内抽取出的时间段"
                 "（原文、推进 / 回顾 / 将来、年数）。", SpansArgs, run=list_time_spans),
            Tool("submit_verdict", "提交核实结论。提交后本次核实结束。",
                 VerdictArgs, terminal=True, guard=check_verdict),
        ]
    )  # fmt: skip


# --- character facts (step 5-1) -------------------------------------------------------------

FactsReason = Literal[
    "misattributed",
    "equivalent",
    "explained",
    "flashback",
    "speculative",
    "generic",
    "genuine",
    "ambiguous",
    "other",
]
FACTS_REASON_LABEL = {
    "misattributed": "认错人",
    "equivalent": "说法等价",
    "explained": "原文交代了变化",
    "flashback": "回忆 / 梦境",
    "speculative": "推测 / 传闻",
    "generic": "不是血缘 / 泛指",
    "genuine": "确实矛盾",
    "ambiguous": "原文有歧义",
    "other": "其他",
}
REASON_LABEL.update({k: v for k, v in FACTS_REASON_LABEL.items() if k not in REASON_LABEL})


class FactsVerdictArgs(ToolArgs):
    verdict: Verdict = Field(
        description="contradiction：真矛盾；false_alarm：误报；needs_author：原文有歧义，只能由作者判断"
    )
    reason: FactsReason = Field(description="判断依据的类型，见说明")
    explanation: str = Field(description="给作者看的一两句中文说明")
    evidence: list[Quote] = Field(description="支持结论的原文，至少一条")


FactCategory = Literal["appearance", "kinship", "life"]


class CharacterFactsArgs(ToolArgs):
    character: str = Field(description="角色名或别名")
    category: FactCategory | None = Field(
        default=None, description="appearance：外貌；kinship：亲属；life：死亡与出场；不填为全部"
    )
    first_chapter: int | None = Field(default=None)
    last_chapter: int | None = Field(default=None)


async def list_character_facts(ctx: ToolContext, args: CharacterFactsArgs) -> list[FactBrief] | str:
    categories = [args.category] if args.category else [APPEARANCE.name, KINSHIP.name, LIFE.name]
    facts = []
    async with ctx.factory() as session:
        for category in categories:
            facts += await recall.list_facts(
                session, user_id=ctx.user_id, book_id=ctx.book_id, character=args.character,
                category=category, chapters=_chapters(args.first_chapter, args.last_chapter),
            )  # fmt: skip
    facts.sort(key=lambda f: (f.chapter_number, f.char_start))
    briefs = [
        FactBrief(chapter=f.chapter_number, start=f.char_start, end=f.char_end, text=f.raw_text,
                  mention=f.mention, reading=describe_fact(f))
        for f in facts
    ]  # fmt: skip
    return _limited(briefs)


class FactsCharacterBrief(BaseModel):
    name: str
    aliases: list[str]
    as_of_chapter: int
    appearance: list[str]  # the latest stated features, with their source
    died: str | None


_TRAIT_WORD = {"eye_color": "瞳色", "hair_color": "发色", "mark": "身体标记"}


async def get_character_facts(ctx: ToolContext, args: CharacterArgs) -> FactsCharacterBrief:
    async with ctx.factory() as session:
        v = await core.get_character(
            session, user_id=ctx.user_id, book_id=ctx.book_id, name=args.name,
            as_of_chapter=args.as_of_chapter,
        )  # fmt: skip
    appearance = [
        f"{_TRAIT_WORD.get(k, k)} {t.value}（第 {t.chapter} 章「{t.quote}」）"
        for k, t in sorted(v.traits.items())
    ]
    died = f"第 {v.died_chapter} 章「{v.died_quote}」" if v.died_chapter else None
    return FactsCharacterBrief(
        name=v.canonical_name, aliases=v.aliases, as_of_chapter=v.as_of_chapter,
        appearance=appearance, died=died,
    )  # fmt: skip


def facts_tools() -> ToolRegistry:
    """The tools of the character-facts prompt (`verify_facts_v1`)."""
    return ToolRegistry(
        [
            Tool("read_passage", "读取某章某个位置前后的原文。核实时首先用它读证据的上下文。",
                 ReadPassageArgs, run=read_passage),
            Tool("search_text", "在全书原文中检索段落（向量 + 关键词），可限定角色和章节范围。"
                 "用来找变化的交代（染发、易容、复活、身世揭晓）、角色的其他描写；"
                 "查询写成具体事实或原文措辞。", SearchTextArgs, run=search_text),
            Tool("get_character", "查角色的主名、别名、最近写明的外貌及出处、死亡章节。",
                 CharacterArgs, run=get_character_facts),
            Tool("list_facts", "列出某角色被抽取出的外貌、亲属、死亡与出场记录"
                 "（章节、位置、原文、抽取结果），可按类别和章节范围过滤。",
                 CharacterFactsArgs, run=list_character_facts),
            Tool("submit_verdict", "提交核实结论。提交后本次核实结束。",
                 FactsVerdictArgs, terminal=True, guard=check_verdict),
        ]
    )  # fmt: skip


# The tools that go with each prompt version.
TOOLS = {"verify_v1": verify_tools, "verify_v2": verify_tools, "verify_facts_v1": facts_tools}


def tools_for(prompt: str) -> ToolRegistry:
    return TOOLS[prompt]()


def _chapters(first: int | None, last: int | None) -> tuple[int, int] | None:
    if first is None and last is None:
        return None
    return (first or 1, last or 100_000)


def _limited[T](items: list[T]) -> list[T] | str:
    if len(items) <= MAX_ITEMS:
        return items
    return (
        f"共 {len(items)} 条，超过 {MAX_ITEMS} 条的上限；请用 first_chapter / last_chapter "
        "缩小章节范围再查。"
    )
