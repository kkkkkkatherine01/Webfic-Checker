import pytest

from tests.fakes import FakeBackend
from tests.story01_ideal import a, elapsed
from webfic.config import Settings
from webfic.evaluation.matching import ModelAge, ModelCharacter, ModelElapsed, Observation
from webfic.evaluation.realtext import (
    AnnotatedChapter,
    AnnotationError,
    Book,
    ChapterScore,
    Pick,
    Target,
    aggregate,
    plan_jobs,
    run_mode,
    score_chapter,
)
from webfic.evaluation.runner import ReplayCallStore, open_cache
from webfic.llm.base import Tier
from webfic.llm.client import JsonLLMClient, TierConfig

CH1 = "林远在山上练剑。\n他练了整整一天。"
CH2 = "第二天，林远今年十八岁了。\n他估计师父七十岁左右。\n这一年很快就会过去。\n三天后，林远下山。"
CH3 = "林远到了城里。"
BOOK = Book("demo", "示例", f"第1章 上山\n{CH1}\n第2章 生日\n{CH2}\n第3章 下山\n{CH3}")
PICK = Pick("demo", "示例", 2, "dense", 1)


def annotation(**extra) -> AnnotatedChapter:
    return AnnotatedChapter.model_validate(
        {
            "id": "01",
            "book": "demo",
            "chapter": 2,
            "characters": {"林远": ["他"], "师父": []},
            "ages": [
                {
                    "quote": "林远今年十八岁",
                    "character": "林远",
                    "type": "absolute_age",
                    "value": 18,
                }
            ],
            "acceptable": [{"quote": "师父七十岁左右"}],
            "elapsed": [{"quote": "第二天", "kind": "short"}],
            "not_elapsed": [{"quote": "这一年很快就会过去"}],
            **extra,
        }
    )


def target(ann: AnnotatedChapter | None = None) -> Target:
    return Target(ann or annotation(), "dense", 2, CH2)


def at(quote: str) -> tuple[int, int]:
    start = CH2.index(quote)
    return start, start + len(quote)


def age(quote: str, value: float, *, character="c1", value_max=None, flashback=False):
    return ModelAge(character, 2, *at(quote), "absolute_age", value, None, flashback, None,
                    value_max=value_max)  # fmt: skip


def span(quote: str, kind: str, years=None) -> ModelElapsed:
    return ModelElapsed(2, *at(quote), kind, years, False)


def observe(ages=(), elapsed_=()) -> Observation:
    chars = [ModelCharacter("c1", {"林远"}), ModelCharacter("c2", {"师父"})]
    return Observation(chars, list(ages), list(elapsed_), [])


# --- annotations and jobs ------------------------------------------------------------


def test_annotation_rejects_unknown_character():
    with pytest.raises(ValueError, match="不在 characters 里"):
        annotation(characters={"师父": []})


def test_plan_jobs_context_and_single():
    [job] = plan_jobs([annotation()], [PICK], {"demo": BOOK}, "context")
    assert "第3章" not in job.text  # only up to the annotated chapter
    [t] = job.targets
    assert (t.chapter, t.content) == (2, CH2)

    [single] = plan_jobs([annotation()], [PICK], {"demo": BOOK}, "single")
    assert "第1章" not in single.text
    assert single.targets[0].chapter == 1


def test_plan_jobs_rejects_quotes_not_in_the_chapter():
    ann = annotation(not_ages=[{"quote": "不存在的句子"}])
    with pytest.raises(AnnotationError, match="找不到"):
        plan_jobs([ann], [PICK], {"demo": BOOK}, "context")


# --- scoring -------------------------------------------------------------------------


def test_perfect_extraction():
    s = score_chapter(target(), observe([age("林远今年十八岁", 18)], [span("第二天", "short")]))
    assert (s.ages_found, s.ages_correct, s.ages_extracted_ok) == (1, 1, 1)
    assert (s.elapsed_found, s.elapsed_correct, s.elapsed_extracted_ok) == (1, 1, 1)
    assert (s.traps, s.traps_passed) == (1, 1)
    assert s.errors == []


def test_acceptable_is_not_a_false_extraction_but_unlabelled_is():
    s = score_chapter(
        target(),
        observe(
            [
                age("林远今年十八岁", 18),
                age("师父七十岁左右", 70, character="c2"),
                age("三天后", 3),
            ],
            [span("这一年很快就会过去", "short")],
        ),
    )
    assert (s.ages_extracted, s.ages_extracted_ok) == (3, 2)
    assert (s.elapsed_extracted, s.elapsed_extracted_ok) == (1, 0)
    # A future duration taken as a "short" span: counted, but passes the trap and is no
    # false advance (only "advance" affects the age check).
    assert (s.future, s.future_taken, s.traps_passed, s.advances) == (1, 1, 1, 0)
    assert "01 误抽年龄：三天后" in s.errors
    assert "01 漏抽时间段：第二天" in s.errors
    assert "01 将来时长标错（short）：这一年很快就会过去" in s.errors
    # Labelled "future" it is right.
    s = score_chapter(target(), observe([], [span("这一年很快就会过去", "future")]))
    assert (s.future, s.future_taken) == (1, 0)


def test_false_advances():
    obs = observe([], [span("第二天", "advance", 1), span("这一年很快就会过去", "advance", 1)])
    s = score_chapter(target(), obs)
    # An advance on a required "short" quote is wrong too.
    assert (s.advances, s.advances_ok) == (2, 0)
    assert (s.elapsed_found, s.elapsed_correct) == (1, 0)
    assert "01 时间段「第二天」：推进判断 advance≠short错误" in s.errors
    assert s.traps_passed == 0
    # "short" versus "retrospective" does not matter.
    s = score_chapter(target(), observe([], [span("第二天", "retrospective")]))
    assert s.elapsed_correct == 1


def test_attribute_errors():
    s = score_chapter(
        target(), observe([age("林远今年十八岁", 18, character="c2", flashback=True)])
    )
    assert (s.ages_found, s.ages_correct) == (1, 0)
    assert "01 年龄「林远今年十八岁」：角色、回忆标记错误" in s.errors
    assert s.verdicts["01/年龄/林远今年十八岁"] == ("回忆标记", "角色")


def test_merged_character_is_wrong_attribution():
    obs = observe([age("林远今年十八岁", 18)])
    obs.characters = [ModelCharacter("c1", {"林远", "师父"})]
    s = score_chapter(target(), obs)
    assert (s.ages_correct, s.merges) == (0, 1)
    assert "01 年龄「林远今年十八岁」：角色（与师父合并）错误" in s.errors

    obs.characters = [ModelCharacter("c1", {"林远"}), ModelCharacter("c3", {"他"})]
    assert score_chapter(target(), obs).splits == 1


def test_approximate_values_and_either_flags():
    ann = annotation(
        ages=[{"quote": "师父七十岁左右", "character": "师父", "type": "absolute_age",
               "value": 70, "approx": True, "flashback": "any"}],
        acceptable=[],
    )  # fmt: skip
    for value, value_max, ok in [(70, None, True), (65, 75, True), (60, 65, False)]:
        obs = observe([age("师父七十岁左右", value, character="c2", value_max=value_max,
                           flashback=value == 65)])  # fmt: skip
        assert score_chapter(target(ann), obs).ages_correct == int(ok), (value, value_max)

    exact = annotation()  # 18 exactly: a range is wrong
    obs = observe([age("林远今年十八岁", 18, value_max=19)])
    assert score_chapter(target(exact), obs).ages_correct == 0


def test_consistency_across_samples():
    right = observe([age("林远今年十八岁", 18)], [span("第二天", "short")])
    # Same verdicts: a different non-advance kind does not matter.
    same = observe([age("林远今年十八岁", 18)], [span("第二天", "retrospective")])
    # The age judged differently; an extra age extraction.
    wrong = observe(
        [age("林远今年十八岁", 18, flashback=True), age("三天后", 3)], [span("第二天", "short")]
    )
    m = aggregate("全部", [[score_chapter(target(), o)] for o in (right, same, wrong)])
    assert m.found_in == {3: 2}
    assert (m.stable.num, m.stable.den) == (1, 2)  # the time span
    assert (m.age_accuracy.num, m.age_accuracy.den) == (2, 3)
    # Age extractions: the required one in every sample, "三天后" in one of three.
    assert (m.age_consistency.num, m.age_consistency.den) == (1, 2)

    # Found in some samples only: not stable; never found: a recall problem only.
    missed = observe([age("林远今年十八岁", 18)])
    m = aggregate("全部", [[score_chapter(target(), o)] for o in (right, missed)])
    assert m.found_in == {2: 1, 1: 1} and (m.stable.num, m.stable.den) == (1, 2)
    assert m.errors["01 漏抽时间段：第二天"] == 1
    never = aggregate("x", [[score_chapter(target(), observe())] for _ in range(2)])
    assert never.found_in == {0: 2} and never.stable.den == 0


def test_empty_group_has_no_ratios():
    m = aggregate("random", [[], []])
    assert m.age_recall.value is None and m.found_in == {}
    assert isinstance(ChapterScore("x", "dense").errors, list)


# --- running -------------------------------------------------------------------------


async def test_run_mode_with_fake_model_and_replay(tmp_path):
    answers = iter([18, 19, 18])  # single: one per sample; then context

    def respond(messages):
        user = messages[1].content
        if "十八岁" in user:
            return {
                "age_statements": [a("林远", "林远", "林远今年十八岁", next(answers))],
                "elapsed_time_statements": [elapsed("第二天", None, "short")],
            }
        return {"age_statements": [], "elapsed_time_statements": []}

    backend = FakeBackend(respond)
    tiers = {Tier.EXTRACT: TierConfig("deepseek-flash"), Tier.REASON: TierConfig("x")}
    cache = await open_cache(tmp_path / "cache.sqlite")

    def make_client(store):
        return JsonLLMClient(backend, tiers, store=store)

    def replay(sample):
        store = ReplayCallStore(cache, sample)
        return lambda _: JsonLLMClient(backend, tiers, store=store)

    jobs = plan_jobs([annotation()], [PICK], {"demo": BOOK}, "single")
    run = await run_mode(jobs, "single", Settings(), lambda _: make_client, cache,
                         samples=2, fresh=True, concurrency=1)  # fmt: skip
    assert len(backend.calls) == 2
    overall = run.rows[0]
    assert [r.name for r in run.rows] == ["全部", "dense", "random", "shushan", "01"]
    assert overall.age_accuracy.value == 0.5 and overall.stable.value == 0.5

    # Replaying gives both recorded answers back, without calling the model.
    again = await run_mode(jobs, "single", Settings(), replay, cache, samples=2, fresh=False)
    assert len(backend.calls) == 2
    assert again.rows[0].age_accuracy.value == 0.5 and again.rows[0].stable.value == 0.5

    context = plan_jobs([annotation()], [PICK], {"demo": BOOK}, "context")
    run = await run_mode(context, "context", Settings(), lambda _: make_client, cache,
                         samples=1, fresh=True)  # fmt: skip
    assert len(backend.calls) == 4  # chapters 1 and 2; chapter 3 is not needed
