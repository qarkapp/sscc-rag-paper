"""Result fusion for multi-path retrieval.

``merge_deduplicate`` reproduces the reference behaviour (union by chunk id, then
re-sort by score). ``reciprocal_rank_fusion`` is the rank-based alternative used by
dual-path hypothesis fusion.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from sage.core.types import ScoreSource, SearchResult

__all__ = ["merge_deduplicate", "reciprocal_rank_fusion"]


def merge_deduplicate(rankings: Sequence[Sequence[SearchResult]]) -> list[SearchResult]:
    """Union results by ``chunk_id`` (first occurrence wins), then sort by score."""
    seen: dict[str, SearchResult] = {}
    for ranking in rankings:
        for result in ranking:
            if result.chunk_id not in seen:
                seen[result.chunk_id] = result
    # Tie-break on chunk_id so equal scores order identically across processes
    # (LanceDB may return equal-distance hits in process-dependent order).
    return sorted(seen.values(), key=lambda r: (-r.relevance_score, r.chunk_id))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], *, k: int = 60
) -> list[SearchResult]:
    """Fuse ranked lists by reciprocal rank: ``score(c) = sum_L 1 / (k + rank_L(c))``."""
    scores: dict[str, float] = defaultdict(float)
    representative: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking):
            scores[result.chunk_id] += 1.0 / (k + rank + 1)
            representative.setdefault(result.chunk_id, result)
    fused = [
        replace(representative[cid], relevance_score=score, score_source=ScoreSource.RRF)
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda r: (-r.relevance_score, r.chunk_id))
    return fused
