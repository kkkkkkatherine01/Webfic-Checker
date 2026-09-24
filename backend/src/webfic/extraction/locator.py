"""Find where a quoted `raw_text` sits in the source text.

Statements that cannot be located are treated as hallucinated and dropped, which is the
main guard against the model inventing facts.
"""

import bisect
import unicodedata


def _canonical(ch: str) -> str | None:
    """Fold width variants and drop whitespace/punctuation; None means 'skip'."""
    folded = unicodedata.normalize("NFKC", ch)
    if len(folded) != 1:
        folded = ch
    category = unicodedata.category(folded)
    if folded.isspace() or category.startswith(("P", "Z", "C")):
        return None
    return folded.lower()


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        c = _canonical(ch)
        if c is not None:
            chars.append(c)
            index_map.append(i)
    return "".join(chars), index_map


def locate(haystack: str, needle: str, start_from: int = 0) -> tuple[int, int] | None:
    """Return (start, end) offsets in `haystack`, or None if not found.

    Tries an exact match first, then a match that ignores whitespace, punctuation and
    full/half-width differences (models often normalise quotes or drop spaces)."""
    needle = needle.strip()
    if not needle:
        return None

    pos = haystack.find(needle, start_from)
    if pos == -1 and start_from:
        pos = haystack.find(needle)
    if pos != -1:
        return pos, pos + len(needle)

    norm_needle, _ = _normalize_with_map(needle)
    if not norm_needle:
        return None
    norm_hay, index_map = _normalize_with_map(haystack)
    # Honour start_from here too, or two identical quotes would collapse into one.
    norm_from = bisect.bisect_left(index_map, start_from)
    i = norm_hay.find(norm_needle, norm_from)
    if i == -1 and norm_from:
        i = norm_hay.find(norm_needle)
    if i == -1:
        return None
    return index_map[i], index_map[i + len(norm_needle) - 1] + 1
