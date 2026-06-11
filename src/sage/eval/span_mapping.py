"""Map gold character spans to retrieved chunks (chunker-agnostic relevance).

Benchmark gold evidence is annotated as character spans in the source document. A
retrieved chunk counts as relevant if it sufficiently overlaps a gold span. This
makes retrieval metrics independent of how any system chunks documents -- the key
fairness mechanism for comparing systems with different chunk sizes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["ChunkSpan", "GoldSpan", "build_qrels", "relevant_chunk_ids"]


@dataclass(frozen=True, slots=True)
class GoldSpan:
    document_id: str
    char_start: int
    char_end: int
    grade: int = 1


@dataclass(frozen=True, slots=True)
class ChunkSpan:
    chunk_id: str
    document_id: str
    char_start: int
    char_end: int


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def relevant_chunk_ids(
    gold_spans: Sequence[GoldSpan],
    chunks: Sequence[ChunkSpan],
    *,
    min_overlap: float = 0.5,
) -> dict[str, int]:
    """Return ``{chunk_id: grade}`` for chunks that cover a gold span.

    A chunk is relevant if its overlap with a gold span is at least ``min_overlap``
    of that span's length (or it fully contains the span). The grade is the maximum
    grade of any covered span.
    """
    relevant: dict[str, int] = {}
    for span in gold_spans:
        span_len = max(1, span.char_end - span.char_start)
        for chunk in chunks:
            if chunk.document_id != span.document_id:
                continue
            overlap = _overlap(span.char_start, span.char_end, chunk.char_start, chunk.char_end)
            if overlap / span_len >= min_overlap:
                relevant[chunk.chunk_id] = max(relevant.get(chunk.chunk_id, 0), span.grade)
    return relevant


def build_qrels(
    gold_by_query: Mapping[str, Sequence[GoldSpan]],
    chunks: Sequence[ChunkSpan],
    *,
    min_overlap: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Build an ir_measures-style qrels mapping from gold spans and chunk spans."""
    return {
        qid: relevant_chunk_ids(spans, chunks, min_overlap=min_overlap)
        for qid, spans in gold_by_query.items()
    }
