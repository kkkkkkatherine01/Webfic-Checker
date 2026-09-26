"""Split long chapters into overlapping chunks for extraction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int  # offset of this chunk within the chapter text
    end: int
    text: str


def chunk_text(text: str, size: int = 8000, overlap: int = 500) -> list[Chunk]:
    """Cut on paragraph boundaries where possible; consecutive chunks share ~`overlap`
    characters so a statement spanning a cut is fully contained in at least one chunk."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")

    n = len(text)
    if n <= size:
        return [Chunk(index=0, start=0, end=n, text=text)]

    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + size, n)
        if end < n:
            # Prefer ending right after a newline in the back half of the window.
            cut = text.rfind("\n", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(Chunk(index=len(chunks), start=start, end=end, text=text[start:end]))
        if end >= n:
            return chunks

        # Start the next chunk on a line within 2x overlap of the end, else overlap hard.
        # Looking further back could land on a line whose chunk ends at this same cut and
        # so lies entirely inside this one (long paragraphs), a wasted model call.
        lowest = max(start + 1, end - 2 * overlap)
        next_start = text.rfind("\n", lowest, end - overlap) + 1
        if next_start <= start:
            next_start = end - overlap
        start = next_start
