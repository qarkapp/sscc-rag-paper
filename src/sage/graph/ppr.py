"""Personalized PageRank result expansion over the chunk graph.

After initial retrieval, the seed results define a restart distribution; chunks with
high personalized PageRank that are not already retrieved are added, up to a budget.
This surfaces structurally-related chunks (a definition's call sites, a referenced
section) that a pure similarity search misses.
"""

from __future__ import annotations

from collections.abc import Sequence

from sage.graph.build import ChunkGraph

__all__ = ["expand_by_ppr", "personalized_pagerank"]


def personalized_pagerank(
    graph: ChunkGraph,
    seed_ids: Sequence[str],
    edge_types: Sequence[str],
    *,
    alpha: float = 0.15,
    steps: int = 20,
) -> dict[str, float]:
    """Return PPR scores per chunk id, restarting at the seed set.

    ``alpha`` is the restart probability; ``steps`` power-iteration steps.
    """
    import networkx as nx

    index_of = graph.index_of
    seeds = [s for s in seed_ids if s in index_of]
    if not seeds:
        return {}

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(len(graph.chunk_ids)))
    nx_graph.add_edges_from(graph.active_edges(edge_types))

    personalization = {index_of[s]: 1.0 / len(seeds) for s in seeds}
    scores = nx.pagerank(
        nx_graph,
        alpha=1.0 - alpha,  # networkx alpha is the damping factor (follow-edge prob)
        personalization=personalization,
        max_iter=max(steps * 10, 100),  # ensure convergence on small graphs
        tol=1e-6,
    )
    return {graph.chunk_ids[i]: float(s) for i, s in scores.items()}


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
