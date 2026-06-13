"""Facet-coverage selection: pick the chunk *set* that covers the query's sub-claims.

Top-k similarity and RRF rank each chunk independently, so they fill the context with
several near-duplicate chunks about the most salient facet while leaving other required
facets uncovered -- exactly the failure mode of multi-hop and cross-document questions,
where the answer needs *diverse* evidence. The cross-encoder reranker cannot fix this:
it scores (query, chunk) pairs and is blind to what the rest of the selected set covers.

This module instead:

1. decomposes the query into facets (sub-claims / required pieces of evidence) with the
   LLM (cached);
2. scores every candidate chunk against every facet (embedding cosine);
3. selects chunks by greedy maximization of total facet coverage
   ``F(S) = sum_f max_{c in S} rel(c, f)`` -- a monotone submodular objective, so the
   greedy order has the standard ``1 - 1/e`` guarantee.

The selection order is returned as the ranking. Chunks that add no new coverage are
appended by raw relevance, so on single-facet (single-hop) queries this reduces to the
relevance ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sage.core.protocols import Generator

__all__ = ["coverage_select", "decompose_query", "facet_relevance"]

_SYSTEM = (
    "You decompose a search question into the distinct pieces of evidence needed to "
    "answer it. Output each piece as a short standalone phrase on its own line. Output "
    "only the phrases, no numbering or commentary."
)


async def decompose_query(query: str, generator: Generator, *, max_facets: int = 5) -> list[str]:
    """Return up to ``max_facets`` facet phrases for the query (the query itself if none).

    The LLM call is content-addressed by the caching client, so decomposition is paid
    once per query across the whole experiment suite.
    """
    prompt = (
        f"Question: {query}\n\nList the distinct pieces of evidence needed to answer it "
        f"(at most {max_facets}), one per line."
    )
    text = await generator.complete(_SYSTEM, prompt, max_tokens=160)
    facets = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
    facets = [f for f in facets if len(f) > 2][:max_facets]
    return facets or [query]


def facet_relevance(chunk_embeddings: np.ndarray, facet_embeddings: np.ndarray) -> np.ndarray:
    """Non-negative cosine relevance matrix ``rel[chunk, facet]``."""
    c = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-12)
    f = facet_embeddings / (np.linalg.norm(facet_embeddings, axis=1, keepdims=True) + 1e-12)
    rel: np.ndarray = np.maximum(c @ f.T, 0.0)
    return rel


def _unit(x: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1] so relevance and coverage are blended on one scale."""
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def coverage_select(
    candidate_ids: Sequence[str],
    relevance: np.ndarray,
    base_scores: Sequence[float],
    *,
    k: int,
    relevance_weight: float = 0.0,
) -> list[str]:
    """Greedy facet-coverage ordering of candidates (MMR-style relevance + coverage).

    At each step a candidate is scored by
    ``relevance_weight * norm_relevance + (1 - relevance_weight) * marginal_coverage``,
    both min-max normalized per query. ``relevance_weight = 0`` is pure coverage;
    ``= 1`` is the raw relevance ranking. ``relevance`` is ``rel[i, f]`` for
    ``candidate_ids[i]`` and facet ``f``; ``base_scores`` is each candidate's retrieval
    score. Returns candidate ids, best first.
    """
    n = len(candidate_ids)
    if n == 0:
        return []
    k = min(k, n)
    base = np.asarray(base_scores, dtype=np.float64)
    norm_base = _unit(base)
    lam = relevance_weight
    covered = np.zeros(relevance.shape[1], dtype=np.float64)
    remaining = set(range(n))
    order: list[int] = []
    while len(order) < k and remaining:
        rem = list(remaining)
        gains = np.array([np.maximum(covered, relevance[i]).sum() - covered.sum() for i in rem])
        norm_gain = _unit(gains)
        # Tiny relevance term so that, once coverage is exhausted, ties resolve by score.
        blended = (1.0 - lam) * norm_gain + (lam + 1e-6) * norm_base[rem]
        best_i = rem[int(np.argmax(blended))]
        order.append(best_i)
        covered = np.maximum(covered, relevance[best_i])
        remaining.remove(best_i)
    order.extend(sorted(remaining, key=lambda i: float(base[i]), reverse=True))
    return [candidate_ids[i] for i in order]
