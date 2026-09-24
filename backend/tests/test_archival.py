"""Archival layer: passage splitting, tokenizing, indexing alongside extraction and
chapter edits, search (SQLite ranking in Python) and reading passages in context."""

import uuid
from itertools import pairwise

import pytest
from sqlalchemy import select

from tests.fakes import make_archival
from tests.test_chapters import USER, World, dump
from webfic.archival.passages import split_passages
from webfic.archival.tokenize import Tokenizer
from webfic.db.models import PassageRow
from webfic.memory.archival import read_passage, search_text
from webfic.services import chapters
from webfic.services.errors import NotFound

# --- splitting -------------------------------------------------------------------------


def test_passages_end_at_sentence_ends_and_overlap():
    text = "".join(f"第{i}句话写在这里，内容不长。" for i in range(40))  # 14 chars each
    spans = split_passages(text, size=60, overlap=20)
    assert all(text[e - 1] == "。" for _, e in spans)
    assert all(e - s <= 60 for s, e in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)
    assert all(b[0] < a[1] for a, b in pairwise(spans))  # overlapping
    assert all(a[0] < b[0] for a, b in pairwise(spans))  # moving forward


def test_long_sentence_without_punctuation_is_cut_hard():
    spans = split_passages("无" * 250, size=100, overlap=20)
    assert [e - s for s, e in spans] == [100, 100, 50]


def test_closing_quotes_stay_with_their_sentence_and_blank_text_has_no_passages():
    text = "他说：“走吧。”她没动。"
    assert split_passages(text, size=8, overlap=2)[0] == (0, text.index("她"))
    assert split_passages("  \n ") == []


def test_tokenizer_keeps_names_whole_and_drops_function_words(tmp_path):
    tokenizer = Tokenizer(tmp_path)
    tokenizer.add_names(["旗木卡卡西"])
    words = tokenizer.tokens("旗木卡卡西的年龄是三十岁。")
    assert "旗木卡卡西" in words and "的" not in words and "。" not in words


# --- indexing --------------------------------------------------------------------------


@pytest.fixture
async def world(factory, tmp_path):
    return await World(factory, make_archival(tmp_path, size=40, overlap=10)).load()


async def passages(factory):
    async with factory() as session:
        rows = (await session.scalars(select(PassageRow))).all()
    return {(r.chapter_number, r.id) for r in rows}


async def test_import_indexes_every_chapter(world):
    assert {n for n, _ in await passages(world.factory)} == {1, 2, 3, 4}


async def test_only_changed_chapters_are_reindexed(world):
    before = await passages(world.factory)
    await world.do(chapters.replace_chapter, number=3, content="林远今年22岁。")
    after = await passages(world.factory)
    assert {p for p in before if p[0] != 3} == {p for p in after if p[0] != 3}
    assert {p for p in before if p[0] == 3}.isdisjoint(after)


async def test_deleting_a_chapter_renumbers_its_successors_passages(world):
    before = {i: n for n, i in await passages(world.factory)}
    await world.do(chapters.delete_chapter, number=2)
    after = {i: n for n, i in await passages(world.factory)}
    assert set(after) == {i for i, n in before.items() if n != 2}  # same passages, kept
    assert all(after[i] == (n if n < 2 else n - 1) for i, n in before.items() if i in after)


async def test_a_dry_run_leaves_the_passages_alone(world):
    before = await dump(world.factory)
    await world.do(chapters.replace_chapter, number=3, content="林远今年22岁。", dry_run=True)
    assert await dump(world.factory) == before


# --- search and reading ----------------------------------------------------------------


async def search(world, query, **filters):
    async with world.factory() as session:
        return await search_text(
            session, world.archival, user_id=USER, book_id=world.book_id, query=query, **filters
        )


async def test_keyword_and_vector_both_find_a_name(world):
    hits = await search(world, "苏晚晴多大", k=1)
    assert hits[0].chapter_number == 4 and "苏晚晴" in hits[0].text
    assert set(hits[0].matched_by) == {"vector", "keyword"}
    for mode in ("vector", "keyword"):
        assert (await search(world, "苏晚晴", k=1, mode=mode))[0].chapter_number == 4


async def test_filters_by_chapter_range_and_character(world):
    hits = await search(world, "今年多少岁", k=10, chapters=(2, 3))
    assert {h.chapter_number for h in hits} <= {2, 3} and hits
    hits = await search(world, "今年多少岁", k=10, character="林远")
    assert {h.chapter_number for h in hits} == {1, 2, 3}  # not 苏晚晴's chapter 4
    with pytest.raises(NotFound):
        await search(world, "年龄", character="没有这个人")


async def test_hits_point_at_the_chapter_text(world):
    hit = (await search(world, "3年后", k=1))[0]
    async with world.factory() as session:
        shown = await read_passage(
            session, user_id=USER, book_id=world.book_id, chapter=hit.chapter_number,
            start=hit.char_start, end=hit.char_end, context=0,
        )  # fmt: skip
    assert shown.text == hit.text


async def test_read_passage_extends_to_whole_sentences(world):
    async with world.factory() as session:
        # "林远今年21岁。" in chapter 2 ("3年后。\n林远今年21岁。"); ask for "21岁" only.
        shown = await read_passage(
            session, user_id=USER, book_id=world.book_id, chapter=2, start=9, end=12, context=20
        )
        clamped = await read_passage(
            session, user_id=USER, book_id=world.book_id, chapter=2, start=-5, end=999
        )
        with pytest.raises(NotFound):
            await read_passage(
                session, user_id=USER, book_id=world.book_id, chapter=9, start=0, end=1
            )
    assert shown.text == "3年后。\n林远今年21岁。"
    assert shown.text[shown.focus_start : shown.focus_end] == "21岁"
    assert clamped.text == "3年后。\n林远今年21岁。"


async def test_nobody_else_can_search_or_read(world):
    async with world.factory() as session:
        with pytest.raises(NotFound):
            await search_text(
                session, world.archival, user_id=uuid.uuid4(), book_id=world.book_id, query="林远"
            )
        with pytest.raises(NotFound):
            await read_passage(
                session, user_id=uuid.uuid4(), book_id=world.book_id, chapter=1, start=0, end=2
            )
