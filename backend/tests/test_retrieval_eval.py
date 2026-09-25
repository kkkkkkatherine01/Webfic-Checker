"""Retrieval evaluation: scoring, DetectiveQA paragraph mapping, questions, and one
evaluation end to end on SQLite with the fake embedder."""

import json
from pathlib import Path

from tests.fakes import make_archival
from tests.story01_ideal import STORY01, needs_golden
from tests.test_chapters import USER, World
from webfic.evaluation.golden import Span, load_story
from webfic.evaluation.retrieval import (
    Question,
    aggregate,
    detectiveqa_questions,
    evaluate,
    golden_questions,
    paragraph_spans,
    score,
)
from webfic.memory.archival import PassageHit


def hit(chapter, start, end):
    return PassageHit(
        chapter_number=chapter, char_start=start, char_end=end, text="", score=0, matched_by=[]
    )


def test_score_ranks_the_first_relevant_hit_and_counts_coverage():
    relevant = (Span(1, 10, 20), Span(3, 0, 5))
    hits = [hit(2, 0, 50), hit(1, 15, 40), hit(1, 30, 60)]
    assert score(hits, relevant) == (2, 0.5)
    assert score([hit(2, 0, 50)], relevant) == (None, 0.0)
    assert score([hit(1, 20, 30)], relevant) == (None, 0.0)  # touching is not overlapping


def test_aggregate():
    archival = make_archival(Path("."))
    q = Question(corpus="c", book="b", query="q", relevant=(Span(1, 0, 1),))
    s = aggregate("c", archival, "hybrid", [(q, 1, 1.0), (q, 7, 0.5), (q, None, 0.0), (q, 2, 1.0)])
    assert (s.questions, s.hit_at_5, s.hit_at_10, s.coverage_at_10) == (4, 0.5, 0.75, 0.625)
    assert s.mrr == round((1 + 1 / 7 + 1 / 2) / 4, 4) and s.missed == ["b q"]


NOVEL = "\n".join([
    "[1]第一部", "[2]第一章", "[3]罗杰在小岛上建房的时候，大家都觉得这人真怪。",
    "[4]他每天清晨都去海边散步，风雨无阻。", "[5]1", "[6]第二章",
    "[7]波洛皱着眉头，看着桌上那封没有署名的信。", "[8]1",
])  # fmt: skip


def test_paragraphs_are_found_in_the_imported_chapters(tmp_path):
    path = tmp_path / "novel_data_zh" / "900-测试-作者.txt"
    path.parent.mkdir()
    path.write_text(NOVEL, "utf-8")
    spans = paragraph_spans(path)
    # The volume and chapter headings are not chapter text; paragraph 5 ("1") is found
    # in chapter 1 only, and paragraph 8 ("1") in chapter 2.
    assert set(spans) == {3, 4, 5, 7, 8}
    assert (spans[3].chapter, spans[7].chapter, spans[5].chapter, spans[8].chapter) == (1, 2, 1, 2)
    assert spans[4].start > spans[3].start


def test_detectiveqa_questions_skip_reasoned_clues(tmp_path):
    (tmp_path / "novel_data_zh").mkdir()
    (tmp_path / "human_anno").mkdir()
    (tmp_path / "novel_data_zh" / "900-测试-作者.txt").write_text(NOVEL, "utf-8")
    anno = [{"novel_id": 900, "questions": [
        {"question": "谁在建房？", "clue_position": [3, -1, 2], "answer_position": 4,
         "reasoning": ["罗杰在岛上建了房子，大家觉得怪", "短", "推理过程：所以是罗杰"]},
        {"question": "只有推理？", "clue_position": [-1], "answer_position": -1},
    ]}]  # fmt: skip
    (tmp_path / "human_anno" / "900.json").write_text(json.dumps(anno, ensure_ascii=False), "utf-8")
    questions, clues, skipped = detectiveqa_questions(tmp_path)
    assert [q.query for q in questions] == ["谁在建房？"]
    # Clue statements become queries of their own (too-short lines and the final
    # reasoning line are not), over the same relevant passages.
    assert [c.query for c in clues] == ["罗杰在岛上建了房子，大家觉得怪"]
    assert clues[0].relevant == questions[0].relevant and clues[0].corpus == "detectiveqa-clues"
    assert len(questions[0].relevant) == 2  # paragraphs 3 and 4; the heading 2 is not text
    assert skipped == 4  # -1, the heading, and the second question's -1 and answer


@needs_golden
def test_golden_questions_point_at_their_quotes():
    story = load_story(STORY01)
    questions = golden_questions([story])
    assert questions and all(q.book == "golden/story01" for q in questions)
    first = questions[0]
    assert (first.query, first.character) == ("林远今年多大", "林远")
    span = first.relevant[0]
    assert story.chapters[span.chapter - 1].content[span.start : span.end] == "他今年十八岁"


async def test_evaluate_end_to_end(factory, tmp_path):
    world = await World(factory, make_archival(tmp_path, size=40, overlap=10)).load()
    async with factory() as session:
        from webfic.memory.archival import read_passage

        where = await read_passage(
            session, user_id=USER, book_id=world.book_id, chapter=4, start=0, end=3, context=0
        )
    question = Question(
        corpus="t",
        book="b",
        query="苏晚晴今年多大",
        relevant=(Span(4, where.char_start, where.char_end),),
    )
    lost = Question(
        corpus="t", book="b", query="苏晚晴", relevant=(Span(9, 0, 5),), character="没有这个人"
    )
    outcomes = await evaluate(
        factory, world.archival, {"b": world.book_id}, [question, lost], "hybrid", user_id=USER
    )
    assert [(rank, coverage) for _, rank, coverage in outcomes] == [(1, 1.0), (None, 0.0)]
