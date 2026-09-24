from itertools import pairwise

import pytest

from webfic.ingest.chunker import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("短文。", size=100, overlap=10)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, 3)


def test_chunks_cover_text_overlap_and_offsets_are_consistent():
    paragraph = "这是一段测试文字。" * 5 + "\n"  # 46 chars
    text = paragraph * 40
    chunks = chunk_text(text, size=300, overlap=60)

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 300
    for prev, nxt in pairwise(chunks):
        assert nxt.start < prev.end  # overlap, no gap
        assert prev.end - nxt.start >= 60
        assert nxt.start > prev.start  # always makes progress


def test_prefers_paragraph_boundaries():
    text = ("甲" * 90 + "\n") * 10
    chunks = chunk_text(text, size=300, overlap=50)
    for chunk in chunks[:-1]:
        assert chunk.text.endswith("\n")


def test_text_without_newlines_still_chunks():
    text = "字" * 1000
    chunks = chunk_text(text, size=300, overlap=50)
    assert chunks[-1].end == 1000
    assert all(len(c.text) <= 300 for c in chunks)


def test_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)
