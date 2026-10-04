"""Split long notes into overlapping chunks at line boundaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    start: int  # character offset of the chunk in the note
    text: str

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def chunk_text(text: str, max_chars: int, overlap: int) -> list[Chunk]:
    """Greedy line packing. Each chunk ends on a newline when possible and the next
    chunk starts `overlap` characters earlier (snapped to a line start), so an item
    cut at a boundary is seen whole in one of the two chunks."""
    if len(text) <= max_chars:
        return [Chunk(0, text)]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", start + max_chars // 2, end)
            if nl != -1:
                end = nl + 1
        chunks.append(Chunk(start, text[start:end]))
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        nl = text.find("\n", nxt, end)
        start = nl + 1 if nl != -1 else nxt
    return chunks
