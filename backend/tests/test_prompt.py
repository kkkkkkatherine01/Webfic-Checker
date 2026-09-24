"""Guards on the extraction prompt itself. Step 1 showed that an example contradicting
a rule silently wrecks extraction, and test stories leaking into the prompt would turn
the eval into an open-book exam."""

import json
import re

import pytest

from tests.story01_ideal import GOLDEN_DIR, needs_golden
from webfic.extraction.extractor import load_prompt
from webfic.extraction.schemas import AgeExtraction

PROMPT = load_prompt()
EXAMPLES = re.findall(r"## 示例 (\d+)\n.*?正文：\n(.*?)\n\n输出：\n(\{.*?\]\})", PROMPT, re.S)


def test_prompt_has_examples():
    assert len(EXAMPLES) >= 3


@pytest.mark.parametrize(("number", "text", "output"), EXAMPLES)
def test_example_output_is_valid_and_quotes_its_own_text(number, text, output):
    data = AgeExtraction.model_validate(json.loads(output))
    quotes = [s.raw_text for s in data.age_statements]
    quotes += [e.raw_text for e in data.elapsed_time_statements]
    quotes += [r.real_name for r in data.revealed_names]
    assert [q for q in quotes if q not in text] == [], f"示例 {number}"


@needs_golden
def test_no_golden_story_text_leaks_into_prompt():
    fragments = set(re.findall(r"[一-鿿]{6,}", PROMPT))
    leaks = []
    for story in sorted(GOLDEN_DIR.glob("story*/text.txt")):
        text = story.read_text("utf-8-sig")
        leaks += [(story.parent.name, f) for f in fragments if f in text]
    assert leaks == []
