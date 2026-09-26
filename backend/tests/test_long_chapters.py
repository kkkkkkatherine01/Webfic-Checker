"""Long-chapter bases (step 4-3): merging chapters and placing the original chapters'
text in the merged ones."""

import json

from webfic.evaluation import long_chapters as lc
from webfic.ingest.splitter import split_chapters


def dataset(tmp_path, chapters: int):
    novel = {
        "novel": "测试",
        "chapters": [
            f"第{i}章 标题{i}\n第{i}章的正文，一共有几句话。\n又一句。"
            for i in range(1, chapters + 1)
        ],
    }
    folder = tmp_path / "webnovelbench"
    folder.mkdir()
    (folder / "novel_data_subset_d_100.json").write_text(
        json.dumps([novel], ensure_ascii=False), "utf-8"
    )
    return tmp_path


def test_three_chapters_become_one_and_a_remainder_joins_the_last(tmp_path):
    (long,) = lc.long_bases(dataset(tmp_path, 10), count=1)
    merged = split_chapters(long.base.text).chapters
    assert [c.title for c in merged] == [
        "第1章 合并（原第1至3章）", "第2章 合并（原第4至6章）", "第3章 合并（原第7至10章）",
    ]  # fmt: skip
    assert long.base.name == "long/000" and long.original == "webnovelbench/000"


def test_every_original_chapter_is_found_at_its_placement(tmp_path):
    root = dataset(tmp_path, 10)
    (long,) = lc.long_bases(root, count=1)
    merged = {c.number: c.content for c in split_chapters(long.base.text).chapters}
    novel = json.loads((root / "webnovelbench" / "novel_data_subset_d_100.json").read_text("utf-8"))
    originals = split_chapters("\n".join(novel[0]["chapters"])).chapters
    assert len(long.placement) == len(originals) == 10
    for chapter, (target, offset) in zip(originals, long.placement, strict=True):
        assert merged[target][offset : offset + len(chapter.content)] == chapter.content
