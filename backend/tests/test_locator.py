from webfic.extraction.locator import locate

TEXT = "那一年，林远十八岁。\n他说：“我叫林远。” 苏晚晴笑了。"


def test_exact_match():
    start, end = locate(TEXT, "林远十八岁")
    assert TEXT[start:end] == "林远十八岁"


def test_match_ignoring_punctuation_and_width():
    # Model changed Chinese quotes to ASCII and dropped the colon.
    start, end = locate(TEXT, '他说"我叫林远。"')
    assert TEXT[start:end] == "他说：“我叫林远"


def test_match_ignoring_whitespace_across_lines():
    start, end = locate(TEXT, "十八岁。他说")
    assert TEXT[start:end] == "十八岁。\n他说"


def test_not_found_returns_none():
    assert locate(TEXT, "林远二十岁") is None
    assert locate(TEXT, "  ") is None
    assert locate(TEXT, "。。") is None


def test_start_from_prefers_later_occurrence():
    text = "他十八岁。后来他又说自己十八岁。"
    first = locate(text, "十八岁")
    later = locate(text, "十八岁", start_from=5)
    assert first[0] < later[0]


def test_start_from_also_applies_to_the_normalized_match():
    text = "他说：“十八，岁。”后来又说：“十八，岁。”"
    first = locate(text, "十八岁")
    later = locate(text, "十八岁", start_from=first[1])
    assert later[0] > first[0]
    assert text[later[0] : later[1]] == "十八，岁"
