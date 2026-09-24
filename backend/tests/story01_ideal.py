"""A hand-written, fully correct extraction of eval/golden/story01, used as a fake LLM
answer: with it, every downstream step (checker, eval scoring) should be perfect."""

from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parents[2] / "eval" / "golden"
STORY01 = GOLDEN_DIR / "story01"

# The golden set is kept out of git; on a fresh clone these tests are skipped.
needs_golden = pytest.mark.skipif(
    not (STORY01 / "text.txt").exists(), reason="eval/golden/ is local only"
)


def a(mention, resolved, raw, value, *, flashback=False, ybp=None, ybp_quote=None):
    return {"mention": mention, "resolved_name": resolved, "raw_text": raw,
            "statement_type": "absolute_age", "value": value, "life_stage": None,
            "is_flashback": flashback, "years_before_present": ybp,
            "years_before_present_quote": ybp_quote}  # fmt: skip


def stage(mention, resolved, raw, life_stage, *, flashback=False, ybp=None, ybp_quote=None):
    return {"mention": mention, "resolved_name": resolved, "raw_text": raw,
            "statement_type": "life_stage", "value": None, "life_stage": life_stage,
            "is_flashback": flashback, "years_before_present": ybp,
            "years_before_present_quote": ybp_quote}  # fmt: skip


def rel(mention, resolved, raw, value):
    return {**a(mention, resolved, raw, value), "statement_type": "relative_age"}


def elapsed(raw, years, kind="advance"):
    return {"raw_text": raw, "estimated_years": years, "kind": kind, "is_flashback": False}


IDEAL = {
    1: {
        "age_statements": [
            a("他", "林远", "他今年十八岁", 18),
            rel("她", "苏晚晴", "她比林远大一岁", 1),
            a("苏晚晴", None, "今年十九", 19),
            stage("赵无极", None, "白发老者", "elderly"),
        ],
        "elapsed_time_statements": [
            elapsed("在山上整整待了十年", 10, "retrospective"),
            elapsed("已经来了好些年", None, "retrospective"),
        ],
    },
    2: {
        "age_statements": [
            a("林远", "林远", "十年前，八岁的林远", 8, flashback=True, ybp=10, ybp_quote="十年前"),
            stage("他", "林远", "瘦弱的孩童", "child", flashback=True, ybp=10, ybp_quote="十年前"),
        ],
        "elapsed_time_statements": [],
    },
    3: {
        "age_statements": [
            a("林远", "林远", "二十岁的林远", 20),
            a("苏晚晴", "苏晚晴", "二十五岁的苏晚晴", 25),
        ],
        "elapsed_time_statements": [
            elapsed("两年后", 2),
            elapsed("这两年她奔波在外", 2, "retrospective"),
        ],
    },
    4: {
        "age_statements": [
            a("赵无极", "赵无极", "年方三十的赵无极", 30),
            a("他", "林远", "他今年二十一岁", 21),
        ],
        "elapsed_time_statements": [elapsed("不知不觉又过了一年", 1)],
    },
    5: {
        "age_statements": [
            a("林远", "林远", "十九岁的林远", 19),
            a("她", "苏晚晴", "二十六岁的她", 26),
        ],
        "elapsed_time_statements": [],
    },
}


def respond(messages):
    user = messages[1].content
    for number, answer in IDEAL.items():
        if f"第 {number} 章" in user:
            return answer
    raise AssertionError(user)
