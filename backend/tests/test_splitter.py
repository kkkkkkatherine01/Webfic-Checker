from webfic.ingest.splitter import split_chapters


def test_splits_numbered_chapters_and_keeps_narrative_order():
    text = "第一章 下山\n林远下山了。\n\n第2章 入城\n他进了城。\n"
    result = split_chapters(text)

    assert [(c.number, c.title) for c in result.chapters] == [(1, "第一章 下山"), (2, "第2章 入城")]
    assert result.chapters[0].content == "林远下山了。"
    assert result.chapters[1].content == "他进了城。"
    assert result.warnings == []


def test_heading_without_space_and_special_headings():
    text = "楔子\n序幕。\n第一章风起\n正文。\n番外一 旧事\n往事。\n尾声\n完。"
    titles = [c.title for c in split_chapters(text).chapters]
    assert titles == ["楔子", "第一章风起", "番外一 旧事", "尾声"]


def test_volume_headings_are_dropped_and_numbering_is_sequential():
    text = "第一卷 风起\n第一章 甲\n甲。\n第二卷 云涌\n第一章 乙\n乙。"
    chapters = split_chapters(text).chapters
    assert [(c.number, c.title) for c in chapters] == [(1, "第一章 甲"), (2, "第一章 乙")]
    assert chapters[0].content == "甲。"


def test_volume_heading_without_separator():
    text = "第一卷风起云涌\n第一章 甲\n甲。"
    chapters = split_chapters(text).chapters
    assert [(c.title, c.content) for c in chapters] == [("第一章 甲", "甲。")]


def test_body_lines_that_look_like_volume_headings_are_kept():
    body = "第二部电影上映那天，他没去。\n第三卷的书稿还在桌上\n第一集结束了。"
    chapters = split_chapters(f"第一章 甲\n{body}").chapters
    assert chapters[0].content == body


def test_body_lines_that_look_like_headings_are_not_split():
    text = (
        "第一章 开端\n尾声渐远，钟声停了。\n"
        "这一句很长很长很长很长，第三章里提到的那件事终于有了结果，大家都松了一口气。"
    )
    chapters = split_chapters(text).chapters
    assert len(chapters) == 1


def test_preamble_is_ignored_with_warning():
    text = "《剑来》\n作者：某某\n\n第一章 始\n正文。"
    result = split_chapters(text)
    assert result.preamble == "《剑来》\n作者：某某"
    assert len(result.chapters) == 1
    assert any("已忽略" in w for w in result.warnings)


def test_no_heading_treats_whole_text_as_one_chapter():
    result = split_chapters("只有一段文字。")
    assert len(result.chapters) == 1
    assert result.chapters[0].content == "只有一段文字。"
    assert result.warnings


def test_crlf_and_bom_are_normalized():
    result = split_chapters("﻿第一章 始\r\n正文。\r\n")
    assert result.chapters[0].title == "第一章 始"
    assert "\r" not in result.chapters[0].content


def test_empty_chapter_is_skipped():
    result = split_chapters("第一章 空\n\n第二章 实\n正文。")
    assert [c.title for c in result.chapters] == ["第二章 实"]
    assert result.chapters[0].number == 1


def test_parse_number():
    from webfic.ingest.splitter import parse_number

    cases = {"12": 12, "十": 10, "十二": 12, "二十": 20, "一百零五": 105, "两千": 2000,
             "一万二千": 12000, "三十六": 36, "〇": 0}  # fmt: skip
    for text, n in cases.items():
        assert parse_number(text) == n, text
    assert parse_number("甲") is None


def test_long_heading_accepted_when_it_continues_the_numbering():
    long_title = (
        "第2章 【番外/论坛体01】理性讨论：这一章的标题特别特别长，远远超过了四十个字的限制（上）"
    )
    assert len(long_title) > 40
    chapters = split_chapters(f"第1章 开始\n甲。\n{long_title}\n乙。").chapters
    assert [c.title for c in chapters] == ["第1章 开始", long_title]


def test_body_line_that_restarts_numbering_is_not_a_heading():
    text = "第29章 赛前\n甲。\n第30章 比赛\n第一回合比赛，开始！\n乙。\n第一章 来暗杀他的人。\n丙。"
    chapters = split_chapters(text).chapters
    assert [c.title for c in chapters] == ["第29章 赛前", "第30章 比赛"]
    assert "第一回合比赛" in chapters[1].content


def test_repeated_and_skipped_numbers_are_accepted_when_short():
    text = "第5章 上\n甲。\n第5章 下\n乙。\n第7章 跳号\n丙。"
    assert len(split_chapters(text).chapters) == 3


def test_far_jump_is_not_a_heading():
    text = "第3章 甲\n甲。\n第九十九章\n乙。"
    assert [c.title for c in split_chapters(text).chapters] == ["第3章 甲"]
