"""Cut a chapter into short, overlapping passages for search.

Passages end at sentence ends (or line ends), so a hit reads as whole sentences, and
neighbours overlap so a statement spanning a cut is whole in at least one passage.
Unrelated to the 8000-character chunks used for extraction.
"""

import re

# A sentence ends after 。！？… (and any closing quotes / brackets right after), or at a
# line break.
_SENTENCE_END = re.compile(r"[。！？!?…]+[”’」』\"'）)]*|\n")


def sentence_ends(text: str) -> list[int]:
    """Offsets right after each sentence end, and the end of the text."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    return ends


def split_passages(text: str, size: int = 500, overlap: int = 100) -> list[tuple[int, int]]:
    """(start, end) offsets of passages of about `size` characters, cut at sentence ends,
    each starting about `overlap` characters before the previous one ended. A sentence
    longer than `size` is cut hard, so no passage is much longer than `size`."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    if not text.strip():
        return []
    ends = sentence_ends(text)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        # The last sentence end that keeps the passage within `size`; if the next
        # sentence alone is longer, cut it hard.
        fitting = [e for e in ends if start < e <= start + size]
        end = fitting[-1] if fitting else min(start + size, len(text))
        spans.append((start, end))
        if end >= len(text):
            break
        # Start the next passage at a sentence end about `overlap` characters back, but
        # always move forward.
        back = [e for e in ends if end - overlap <= e < end and e > start]
        start = back[0] if back else end
    # Leading whitespace (blank lines between paragraphs) is not part of a passage.
    trimmed = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            trimmed.append((s, e))
    return trimmed
