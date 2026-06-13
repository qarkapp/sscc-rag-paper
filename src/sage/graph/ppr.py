"""Personalized PageRank result expansion over the chunk graph.

After initial retrieval, the seed results define a restart distribution; chunks with
high personalized PageRank that are not already retrieved are added, up to a budget.
This surfaces structurally-related chunks (a definition's call sites, a referenced
section) that a pure similarity search misses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sage.graph.build import ChunkGraph

__all__ = ["expand_by_ppr", "personalized_pagerank"]


def _transition_matrix(graph: ChunkGraph, edge_types: Sequence[str]):  # type: ignore[no-untyped-def]
    """Column-stochastic sparse transition matrix for the active edges, memoized.

    Built once per edge-type set and cached on the graph: query-time PPR is then a
    handful of sparse matrix-vector products rather than a full graph reconstruction.
    """
    from scipy import sparse  # type: ignore[import-untyped]

    key = tuple(sorted(edge_types))
    cached = graph._ppr_cache.get(key)
    if cached is not None:
        return cached
    n = len(graph.chunk_ids)
    edges = graph.active_edges(edge_types)
    if edges:
        a = np.fromiter((e[0] for e in edges), dtype=np.int64, count=len(edges))
        b = np.fromiter((e[1] for e in edges), dtype=np.int64, count=len(edges))
        rows = np.concatenate([a, b])  # undirected: both directions
        cols = np.concatenate([b, a])
        adj = sparse.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    else:
        adj = sparse.csr_matrix((n, n))
    degree = np.asarray(adj.sum(axis=0)).ravel()
    degree[degree == 0.0] = 1.0
    transition = (adj @ sparse.diags(1.0 / degree)).tocsr()  # column-normalized
    graph._ppr_cache[key] = transition
    return transition


def personalized_pagerank(
    graph: ChunkGraph,
    seed_ids: Sequence[str],
    edge_types: Sequence[str],
    *,
    alpha: float = 0.15,
    steps: int = 20,
) -> dict[str, float]:
    """Return PPR scores per chunk id, restarting at the seed set.

    ``alpha`` is the restart probability; ``steps`` power-iteration steps. Computed by
    sparse power iteration ``p <- (1-alpha) M p + alpha r`` over a cached transition
    matrix, so cost is O(steps * nnz) per query rather than a per-query graph rebuild.
    """
    index_of = graph.index_of
    seed_idx = [index_of[s] for s in seed_ids if s in index_of]
    if not seed_idx:
        return {}

    transition = _transition_matrix(graph, edge_types)
    n = len(graph.chunk_ids)
    restart = np.zeros(n, dtype=np.float64)
    restart[seed_idx] = 1.0 / len(seed_idx)
    scores = restart.copy()
    for _ in range(max(1, steps)):
        scores = (1.0 - alpha) * (transition @ scores) + alpha * restart
    return {graph.chunk_ids[i]: float(scores[i]) for i in np.nonzero(scores)[0]}


def expand_by_ppr(
    graph: ChunkGraph,
    seed_ids: Sequence[str],
    edge_types: Sequence[str],
    *,
    budget: int,
    alpha: float = 0.15,
    steps: int = 20,
    min_score: float = 0.0,
) -> list[str]:
    """Return up to ``budget`` new chunk ids ranked by PPR, excluding the seeds.

    Only chunks with PPR score above ``min_score`` are added, which keeps weakly
    connected (off-topic) chunks out of the candidate set.
    """
    if budget <= 0:
        return []
    scores = personalized_pagerank(graph, seed_ids, edge_types, alpha=alpha, steps=steps)
    seeds = set(seed_ids)
    threshold = max(0.0, min_score)
    candidates = [(cid, s) for cid, s in scores.items() if cid not in seeds and s > threshold]
    candidates.sort(key=lambda p: p[1], reverse=True)
    return [cid for cid, _ in candidates[:budget]]
