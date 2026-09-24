from tests.fakes import FakeBackend
from webfic.extraction.extractor import extract_chapter
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

TEXT = "三天之后，他们到了镇上。三年后，那人自称沈砚。"


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
