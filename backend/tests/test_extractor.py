from tests.fakes import FakeBackend
from webfic.extraction.extractor import extract_chapter
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

TEXT = (
    "三天之后，他们到了镇上。三年后，那人自称沈砚。“再过五年我就回来。”"
    "若干年后，镇上的孩子年满十岁都要去学堂。"
)


def client(answer):
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    return JsonLLMClient(FakeBackend(lambda m: answer), tiers)


async def extract(answer):
    return await extract_chapter(
        client(answer), chapter_number=1, text=TEXT, known_characters="（暂无）",
        system_prompt="json", chunk_size=8000, chunk_overlap=500,
    )  # fmt: skip


async def test_tiny_advance_is_relabelled_short():
    result = await extract(
        {
            "elapsed_time_statements": [
                {"raw_text": "三天之后", "estimated_years": 0.008, "kind": "advance"},
                {"raw_text": "三年后", "estimated_years": 3, "kind": "advance"},
            ]
        }
    )
    assert [e.statement.kind for e in result.elapsed] == ["short", "advance"]


async def test_advance_with_a_future_marker_is_relabelled_future():
    result = await extract(
        {
            "elapsed_time_statements": [
                {"raw_text": "三年后", "estimated_years": 3, "kind": "advance"},
                {"raw_text": "再过五年", "estimated_years": 5, "kind": "advance"},
                {"raw_text": "若干年后", "estimated_years": None, "kind": "advance"},
            ]
        }
    )
    # "三年后" in narration and "若干年后" (若 but not a condition) stay advances.
    assert [e.statement.kind for e in result.elapsed] == ["advance", "future", "advance"]


async def test_generic_ages_are_dropped():
    age = {"mention": "孩子", "resolved_name": None, "raw_text": "年满十岁都要去学堂",
           "statement_type": "absolute_age", "value": 10}  # fmt: skip
    result = await extract({"age_statements": [{**age, "generic": True}]})
    assert result.ages == [] and result.dropped == ["年满十岁都要去学堂"]
    result = await extract({"age_statements": [age]})
    assert len(result.ages) == 1


async def test_revealed_name_must_appear_in_text():
    result = await extract(
        {
            "revealed_names": [
                {"known_as": "那人", "real_name": "沈砚"},
                {"known_as": "那人", "real_name": "顾长风"},  # not in the text
            ]
        }
    )
    assert [r.real_name for r in result.revealed_names] == ["沈砚"]
    assert result.dropped == ["顾长风"]


FLASHBACK_TEXT = "十五年前，十三岁的他第一次进城。父亲去世时，他才九岁。"


async def extract_flashbacks(ages):
    return await extract_chapter(
        client({"age_statements": ages}), chapter_number=1, text=FLASHBACK_TEXT,
        known_characters="- 周屿", system_prompt="json", chunk_size=8000, chunk_overlap=500,
    )  # fmt: skip


def flashback(raw, value, ybp, quote):
    return {"mention": "他", "resolved_name": "周屿", "raw_text": raw,
            "statement_type": "absolute_age", "value": value, "is_flashback": True,
            "years_before_present": ybp, "years_before_present_quote": quote}  # fmt: skip


async def test_offset_is_kept_only_when_its_quote_is_in_the_text():
    result = await extract_flashbacks(
        [
            flashback("十三岁的他", 13, 15, "十五年前"),  # stated in the text
            flashback("他才九岁", 9, 19, None),  # the model's own guess
            flashback("他才九岁", 9, 19, "十九年前"),  # a quote that is not in the text
        ]
    )
    offsets = [a.statement.years_before_present for a in result.ages]
    assert offsets == [15, None]  # the third is the same statement: deduplicated


def test_upper_bounds_alone_are_not_ages():
    from webfic.extraction.extractor import _valid_age
    from webfic.extraction.schemas import AgeStatement

    def age(raw, value):
        return AgeStatement(mention="他", raw_text=raw, statement_type="absolute_age", value=value)

    assert not _valid_age(age("连一千岁都不到", 999))
    assert not _valid_age(age("还没到三十岁", 29))
    assert not _valid_age(age("不到二十岁的年纪", 19))
    assert _valid_age(age("这家伙今年还未满十八周岁", 17))
    assert _valid_age(age("他三十岁不到就当上了掌门", 29)) is False  # an upper bound too


def test_durations_are_not_ages():
    from webfic.extraction.extractor import _valid_age
    from webfic.extraction.schemas import AgeStatement

    def ok(raw):
        return _valid_age(AgeStatement(mention="x", raw_text=raw,
                                       statement_type="absolute_age", value=18))  # fmt: skip

    for raw in ["迎接迟来十八年的长眠", "守了整整三十年", "等了他二十年"]:
        assert not ok(raw), raw
    for raw in ["十八岁的林远", "今年二十五", "年方二八", "五水都小三十了吧", "今年十八岁",
                "年纪二十出头", "十八年前，他才十八岁"]:  # fmt: skip
        assert ok(raw), raw


def test_a_stored_extraction_round_trips():
    from webfic.extraction.extractor import (
        ChapterExtraction,
        LocatedAge,
        LocatedElapsed,
        dump_extraction,
        extraction_version,
        load_extraction,
    )
    from webfic.extraction.schemas import AgeStatement, ElapsedTimeStatement, RevealedName

    original = ChapterExtraction(
        ages=[LocatedAge(AgeStatement(mention="林远", raw_text="林远三十来岁",
                                      statement_type="absolute_age", value=30, value_max=39,
                                      is_flashback=True, years_before_present=15,
                                      years_before_present_quote="十五年前"), 3, 9)],
        elapsed=[LocatedElapsed(ElapsedTimeStatement(raw_text="三年后", estimated_years=3), 0, 3)],
        revealed_names=[RevealedName(known_as="疤脸刀客", real_name="沈砚")],
        dropped=["幻觉"],
    )  # fmt: skip
    again = load_extraction(dump_extraction(original))
    assert again.ages == original.ages and again.elapsed == original.elapsed
    assert again.revealed_names == original.revealed_names and again.dropped == ["幻觉"]
    assert again.llm_calls == 0
    assert extraction_version("p", 8000, 500) != extraction_version("p", 8000, 400)
    # Stored in story coordinates (step 4.6): read for a story that now starts later,
    # every position moves with it; a row from before 4.6 keeps its positions.
    stored = dump_extraction(original, offset=3)
    moved = load_extraction(stored, offset=10)
    assert [(a.char_start, a.char_end) for a in moved.ages] == [(10, 16)]
    assert [(e.char_start, e.char_end) for e in moved.elapsed] == [(7, 10)]
    old_row = {k: v for k, v in dump_extraction(original).items() if k != "coords"}
    assert load_extraction(old_row, offset=10).ages == original.ages


def test_stored_positions_are_checked_against_the_chapter():
    from webfic.extraction.extractor import ChapterExtraction, LocatedAge, positions_hold
    from webfic.extraction.schemas import AgeStatement

    reading = ChapterExtraction(
        ages=[LocatedAge(AgeStatement(mention="林远", raw_text="林远今年十八岁",
                                      statement_type="absolute_age", value=18), 2, 10)]
    )  # fmt: skip
    assert positions_hold(reading, "序。林远，今年十八岁。")  # punctuation aside
    assert not positions_hold(reading, "这是序言。林远今年十八岁。")
    assert not positions_hold(reading, "短")


# --- step 3.9 -------------------------------------------------------------------------

SHARED = "林远和苏晴是同学，两人都是十八岁。"


def shared_age(name):
    return {"mention": name, "raw_text": "两人都是十八岁", "statement_type": "absolute_age",
            "value": 18}  # fmt: skip


async def test_two_characters_sharing_a_quote_are_both_kept():
    result = await extract_chapter(
        client({"age_statements": [shared_age("林远"), shared_age("苏晴")]}),
        chapter_number=1, text=SHARED, known_characters="（暂无）", system_prompt="json",
        chunk_size=8000, chunk_overlap=500,
    )  # fmt: skip
    assert [a.statement.mention for a in result.ages] == ["林远", "苏晴"]


async def test_a_statement_read_again_in_the_overlap_is_kept_once():
    # Two chunks that both contain the shared sentence (the overlap) and both report
    # the two characters: two facts, not four.
    text = "甲" * 90 + "\n" + SHARED + "\n" + "乙" * 90
    result = await extract_chapter(
        client({"age_statements": [shared_age("林远"), shared_age("苏晴")]}),
        chapter_number=1, text=text, known_characters="（暂无）", system_prompt="json",
        chunk_size=150, chunk_overlap=60,
    )  # fmt: skip
    assert result.llm_calls > 1
    assert sorted(a.statement.mention for a in result.ages) == ["林远", "苏晴"]


def test_future_markers_leave_real_advances_alone():
    from webfic.extraction.extractor import _FUTURE_MARKER

    future = ["再过三年", "还有两年半", "还有三个月就要出发", "至少还要五年", "少说也得三年"]
    advance = ["三年过去了，他还有些不适应", "至少过去了五年", "少说也过了三年", "三年后"]
    assert all(_FUTURE_MARKER.search(x) for x in future)
    assert not any(_FUTURE_MARKER.search(x) for x in advance)
