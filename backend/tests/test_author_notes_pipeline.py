"""Author's notes in the pipeline (step 4.5): left out of extraction, positions kept,
recomputation only from the first chapter whose story text changed, marked for the
verify agent."""

import re
from pathlib import Path

import pytest
from sqlalchemy import select

from tests.fakes import make_archival
from tests.test_chapters import USER, World
from webfic.agent.tools import ToolContext
from webfic.agent.verify import ReadPassageArgs, read_passage
from webfic.db.models import Book, ChapterNoteRow, FactRow
from webfic.extraction.author_notes import load_prompt
from webfic.services import chapters

BOOK = """第1章 起
作者有话说：今天只有一更。
林远今年18岁。
第2章 承
3年后。
林远今年21岁。
PS：说明一下设定，别当真。
林远今年99岁是以前写错的。
第3章 转
林远今年22岁。
Note: 下章再说。
林远今年50岁这件事也是。
第4章 合
苏晚晴今年20岁。
"""


@pytest.fixture
async def world(factory, tmp_path):
    return await World(factory, make_archival(tmp_path, size=40, overlap=10)).load(BOOK)


async def ages(factory):
    async with factory() as session:
        query = select(
            FactRow.chapter_number, FactRow.value_num, FactRow.char_start, FactRow.raw_text
        )
        rows = (await session.execute(query)).all()
    return sorted(rows)


async def test_ages_in_marked_notes_are_not_extracted_and_positions_hold(world):
    found = await ages(world.factory)
    # 99 (a "PS" note) is left out; 50 is too — until "Note" is registered it is story.
    assert [(c, v) for c, v, _, _ in found] == [(1, 18), (2, 21), (3, 22), (3, 50), (4, 20)]
    async with world.factory() as session:
        from webfic.db.models import Chapter

        content = await session.scalar(select(Chapter.content).where(Chapter.number == 1))
    start = next(s for c, _, s, _ in found if c == 1)
    assert content[start : start + len("林远今年18岁")] == "林远今年18岁"  # after the head note


async def test_registering_a_marker_recomputes_from_the_first_changed_chapter(world):
    calls = len(world.backend.calls)
    result = await world.do(chapters.rescan_author_notes, markers=["Note"])
    # Chapter 3's story text changed (its "Note" line is out now): chapters 3 and 4 are
    # recomputed, only chapter 3 is read by the model again.
    assert (result.extracted, result.reused) == (2, 1)
    assert len(world.backend.calls) == calls + 1
    assert [(c, v) for c, v, _, _ in await ages(world.factory)] == [
        (1, 18), (2, 21), (3, 22), (4, 20),
    ]  # fmt: skip
    async with world.factory() as session:
        book = await session.scalar(select(Book))
        assert book.author_note_markers == ["Note"]
        assert await session.scalar(select(ChapterNoteRow).where(ChapterNoteRow.ranges != []))
    # Nothing changed: nothing is recomputed.
    again = await world.do(chapters.rescan_author_notes)
    assert again.extracted == 0


async def test_a_dry_run_leaves_the_markers_alone(world):
    from tests.test_chapters import dump

    before = await dump(world.factory)
    await world.do(chapters.rescan_author_notes, markers=["Note"], dry_run=True)
    assert await dump(world.factory) == before


async def test_the_verify_agent_sees_notes_marked(world):
    ctx = ToolContext(world.factory, USER, world.book_id, world.archival)
    text = await read_passage(ctx, ReadPassageArgs(chapter=2, start=0, end=5, context=200))
    assert "是作者的话，不是正文" in text


def test_the_note_prompt_shares_no_text_with_the_evaluation_data():
    path = Path(__file__).resolve().parents[2] / "eval" / "author_notes" / "snippets.yaml"
    if not path.exists():
        pytest.skip("eval/author_notes/snippets.yaml not present")
    data = path.read_text("utf-8")
    leaks = [f for f in set(re.findall(r"[一-鿿]{6,}", load_prompt())) if f in data]
    assert leaks == []


async def test_a_longer_head_note_moves_reused_facts_with_the_story(world):
    # Step 4.6: the story text is unchanged, so the stored extraction is reused, but the
    # note before it grew; stored positions must follow the story.
    async with world.factory() as session:
        from webfic.db.models import Chapter

        content = await session.scalar(select(Chapter.content).where(Chapter.number == 1))
    longer = content.replace("今天只有一更。", "今天只有一更，感谢各位读者的打赏和月票支持！")
    calls = len(world.backend.calls)
    result = await world.do(chapters.replace_chapter, number=1, content=longer)
    assert result.reused >= 1 and len(world.backend.calls) == calls  # no model call
    async with world.factory() as session:
        from webfic.db.models import Chapter

        rows = await session.execute(
            select(Chapter.content, FactRow.char_start, FactRow.char_end, FactRow.raw_text).join(
                FactRow, FactRow.chapter_id == Chapter.id
            )
        )
        for text, start, end, raw in rows:
            assert text[start:end] == raw
