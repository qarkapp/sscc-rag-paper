"""Retrieval-aware adaptive chunking (RAAC).

Standard chunking optimizes for a fixed size, ignoring downstream retrieval. RAAC
maintains per-chunk statistics and, over re-indexing cycles, plans boundary edits:

* **split** a frequently-retrieved chunk with low precision (relevant and irrelevant
  content co-located).
* **merge** adjacent chunks that are almost always co-retrieved (redundant split).
* **re-anchor** a frequently-retrieved chunk the generator does not use, by
  prepending hierarchical context.

This module plans operations from statistics; applying them is the indexer's job.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import StrEnum

from sage.config.schema import RaacCfg

__all__ = ["ChunkStats", "RaacOpType", "RaacOperation", "plan_operations"]


class RaacOpType(StrEnum):
    SPLIT = "split"
    MERGE = "merge"
    RE_ANCHOR = "re_anchor"


@dataclass(frozen=True, slots=True)
class ChunkStats:
    """Observed retrieval statistics for one chunk."""

    chunk_id: str
    document_id: str
    chunk_index: int
    hit_rate: float  # fraction of queries retrieving this chunk
    precision: float  # average relevance when retrieved
    co_retrieval_entropy: float  # entropy of co-retrieved chunks (low = redundant)
    generation_utility: float  # fraction of generated output grounded in this chunk


@dataclass(frozen=True, slots=True)
class RaacOperation:
    op_type: RaacOpType
    chunk_ids: tuple[str, ...]
    reason: str


def plan_operations(stats: list[ChunkStats], cfg: RaacCfg) -> list[RaacOperation]:
    """Plan split/merge/re-anchor operations from per-chunk statistics."""
    operations: list[RaacOperation] = []
    by_doc: dict[str, list[ChunkStats]] = {}
    for s in stats:
        by_doc.setdefault(s.document_id, []).append(s)

    for s in stats:
        if s.hit_rate >= 0.5 and s.precision < cfg.split_precision_threshold:
            operations.append(
                RaacOperation(RaacOpType.SPLIT, (s.chunk_id,), "high hit-rate, low precision")
            )
        elif s.hit_rate >= 0.5 and s.generation_utility < 0.1:
            operations.append(
                RaacOperation(
                    RaacOpType.RE_ANCHOR, (s.chunk_id,), "retrieved but not used by generator"
                )
            )

    for chunks in by_doc.values():
        ordered = sorted(chunks, key=lambda c: c.chunk_index)
        for a, b in itertools.pairwise(ordered):
            if (
                b.chunk_index == a.chunk_index + 1
                and a.co_retrieval_entropy < cfg.merge_coretrieval_threshold
                and b.co_retrieval_entropy < cfg.merge_coretrieval_threshold
            ):
                operations.append(
                    RaacOperation(RaacOpType.MERGE, (a.chunk_id, b.chunk_id), "always co-retrieved")
                )
    return operations
