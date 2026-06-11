"""Cross-encoder reranking of candidate results."""

from __future__ import annotations

from sage.core.protocols import Reranker
from sage.core.types import ScoreSource, SearchResult

__all__ = ["apply_reranker"]


async def apply_reranker(
    reranker: Reranker, query: str, results: list[SearchResult], top_n: int
) -> list[SearchResult]:
    """Rerank ``results`` with a cross-encoder and keep the top ``top_n``.

    The reranked score replaces the bi-encoder score and the source is marked as
    cross-encoder, so downstream correction applies the correct calibrated threshold.
    """
    if not results:
        return results
    ranked = await reranker.rerank(query, [r.content for r in results], top_n)
    return [results[idx].with_score(score, ScoreSource.CROSS_ENCODER) for idx, score in ranked]
