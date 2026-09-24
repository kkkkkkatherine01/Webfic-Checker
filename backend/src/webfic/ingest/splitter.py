"""Split a whole-book text file into chapters by web-novel chapter headings."""

import re
from dataclasses import dataclass, field

_NUM = r"[零〇一二两三四五六七八九十百千万\d]+"

# Keyword headings and short numbered headings. Longer lines are only accepted as
# numbered headings when their number continues the sequence (see split_chapters).
_MAX_SHORT_HEADING = 40
_MAX_HEADING = 100
# A short numbered heading may skip a few numbers (authors do), but not jump far ahead.
_MAX_SKIP = 10

NUMBERED_HEADING = re.compile(rf"^第({_NUM})[章节回]")
# "尾声渐远" in the body must not match: keyword headings need a separator or a number.
KEYWORD_HEADING = re.compile(
    rf"^(?:楔子|序章|序|引子|番外|尾声|后记)(?:$|[\s:：·、—\-．.（(【\[]|{_NUM})"
)
# "第三卷的书稿" is a noun phrase, never a volume title.
VOLUME_HEADING = re.compile(rf"^第{_NUM}[卷部集](?!的)")
# Volume lines are dropped, so a body sentence that starts like one ("第二部电影上映那天，
# 他没去。") must not match: after the volume word comes a separator or the end of the
# line, or else the line has no sentence punctuation ("第一卷风起云涌").
_VOLUME_SEPARATED = re.compile(rf"^第{_NUM}[卷部集](?:$|[\s:：·、—\-．.（(【\[])")
_SENTENCE_PUNCT = re.compile(r"[，。！？；…,!?;]")
# "第一卷 第七章 浮游世界", "第三卷 途漫漫 第三六七章 久违了": a chapter heading carrying
# its volume. Group 1 is the volume number, group 2 the chapter heading proper.
_VOLUME_PREFIXED = re.compile(
    rf"^第({_NUM})[卷部集](?:[\s:：·、—\-．.]\S{{0,12}})?\s*(第{_NUM}[章节回].*)$"
)
# A heading in its full form: number, then a separator, then a title ("第一百零二章 权限
# 问题"). Such a line is a heading even when its number is off (an author's typo, or a new
# volume restarting at 1 without a volume line), unless it reads like a sentence: the body
# line "第一章 来暗杀他的人。" must stay body text.
_TITLED_HEADING = re.compile(rf"^第{_NUM}[章节回][ \t　]+\S")
_BODY_PUNCT = re.compile(r"[，。；,;]")

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}  # fmt: skip
_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def parse_number(text: str) -> int | None:
    """Arabic or Chinese numerals ("12", "十二", "一百零五") -> int. Chinese digits
    written place by place without units ("三六七", "二七零") are read positionally."""
    if text.isdigit():
        return int(text)
    if len(text) > 1 and all(ch in _DIGITS for ch in text):
        return int("".join(str(_DIGITS[ch]) for ch in text))
    total, section, digit = 0, 0, None
    for ch in text:
        if ch in _DIGITS:
            digit = _DIGITS[ch]
        elif ch in _UNITS:
            unit = _UNITS[ch]
            if unit == 10000:
                total += (section + (digit or 0)) * unit
                section = 0
            else:
                section += (digit if digit is not None else 1) * unit
            digit = None
        else:
            return None
    return total + section + (digit or 0)


@dataclass
class RawChapter:
    number: int
    title: str
    content: str


@dataclass
class SplitResult:
    chapters: list[RawChapter]
    preamble: str = ""
    warnings: list[str] = field(default_factory=list)


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")


def _is_volume_heading(line: str) -> bool:
    return (
        len(line) <= _MAX_SHORT_HEADING
        and bool(VOLUME_HEADING.match(line))
        and not NUMBERED_HEADING.match(line)
        and (bool(_VOLUME_SEPARATED.match(line)) or not _SENTENCE_PUNCT.search(line))
    )


class _HeadingTracker:
    """Decides whether a line is a chapter heading, using the numbering so far.

    Length alone is a poor signal: real titles can be long ("第66章 【番外】……（上）")
    and body sentences can start like a heading ("第一回合比赛，开始！"). A numbered
    line counts as a heading when its number fits the sequence: the next number (any
    length up to _MAX_HEADING), or, for short lines, a repeat (上/下 parts) or a small
    skip. Numbering may restart at 1 right after a volume heading."""

    def __init__(self) -> None:
        self.last: int | None = None
        self.after_volume = False

    def volume(self) -> None:
        self.after_volume = True

    def accept(self, line: str) -> bool:
        if len(line) > _MAX_HEADING:
            return False
        if KEYWORD_HEADING.match(line):
            return len(line) <= _MAX_SHORT_HEADING
        match = NUMBERED_HEADING.match(line)
        if not match:
            return False
        n = parse_number(match.group(1))
        if n is None:
            return False

        short = len(line) <= _MAX_SHORT_HEADING
        if self.last is None or n == self.last + 1 or (self.after_volume and n == 1):
            ok = True
        else:
            ok = short and self.last <= n <= self.last + _MAX_SKIP
        if ok:
            self.last, self.after_volume = n, False
            return True

        if not (short and _TITLED_HEADING.match(line) and not _BODY_PUNCT.search(line)):
            return False
        # Out of sequence, but clearly a heading. 1 restarts the numbering (a new volume);
        # a smaller number is a typo ("第一八十三章" after 182), so the sequence carries on
        # as if it were the next one; a larger one is a real jump.
        if n == 1 or n > self.last:
            self.last = n
        else:
            self.last += 1
        self.after_volume = False
        return True


def split_chapters(text: str) -> SplitResult:
    """Chapters are numbered 1..N in narrative order, not by the number in the title,
    since titles skip numbers, repeat them per volume, or have none (楔子, 番外)."""
    lines = normalize_newlines(text).split("\n")

    preamble: list[str] = []
    chapters: list[tuple[str, list[str]]] = []
    warnings: list[str] = []
    tracker = _HeadingTracker()

    volume: int | None = None
    for line in lines:
        stripped = line.strip()
        prefixed = _VOLUME_PREFIXED.match(stripped)
        if prefixed:
            # Judge the chapter part; a change of volume may restart the numbering.
            if parse_number(prefixed.group(1)) != volume:
                volume = parse_number(prefixed.group(1))
                tracker.volume()
            is_heading = tracker.accept(prefixed.group(2))
        elif _is_volume_heading(stripped):
            tracker.volume()
            continue
        else:
            is_heading = tracker.accept(stripped)
        if is_heading:
            chapters.append((stripped, []))
        elif chapters:
            chapters[-1][1].append(line)
        else:
            preamble.append(line)

    preamble_text = "\n".join(preamble).strip()

    if not chapters:
        warnings.append("未识别到任何章节标题，整本作为一章处理。")
        return SplitResult(
            chapters=[RawChapter(number=1, title="全文", content=preamble_text)],
            warnings=warnings,
        )

    if preamble_text:
        warnings.append(f"第一个章节标题前有 {len(preamble_text)} 字内容（如书名、简介），已忽略。")

    result: list[RawChapter] = []
    for title, body in chapters:
        content = "\n".join(body).strip("\n")
        if not content.strip():
            warnings.append(f"章节「{title}」没有正文，已跳过。")
            continue
        result.append(RawChapter(number=len(result) + 1, title=title, content=content))

    return SplitResult(chapters=result, preamble=preamble_text, warnings=warnings)
