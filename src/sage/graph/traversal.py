"""Entailment-chain traversal for multi-hop retrieval.

From seed chunks, follow entailment/elaboration edges up to a hop limit, scoring each
chain by the product of its edge confidences. High-scoring chains connect evidence
across documents for multi-hop questions.
"""

from __future__ import annotations

from collections.abc import Sequence

from sage.graph.nli_edges import EntailmentEdge, EntailmentLabel

__all__ = ["EntailmentChain", "entailment_chains"]

EntailmentChain = tuple[list[str], float]

_TRAVERSABLE = {EntailmentLabel.ENTAILMENT, EntailmentLabel.ELABORATION}


def entailment_chains(
    edges: Sequence[EntailmentEdge],
    seed_ids: Sequence[str],
    *,
    max_hops: int = 3,
    top_k: int = 10,
) -> list[EntailmentChain]:
    """Return chains from the seeds, ranked by cumulative edge confidence."""
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for edge in edges:
        if edge.label in _TRAVERSABLE:
            adjacency.setdefault(edge.src, []).append((edge.dst, edge.confidence))

    chains: list[EntailmentChain] = []

    def walk(node: str, path: list[str], score: float, hops: int) -> None:
        if hops >= max_hops:
            return
        for nxt, confidence in adjacency.get(node, []):
            if nxt in path:  # avoid cycles
                continue
            new_path = [*path, nxt]
            new_score = score * confidence
            chains.append((new_path, new_score))
            walk(nxt, new_path, new_score, hops + 1)

    for seed in seed_ids:
        walk(seed, [seed], 1.0, 0)

    chains.sort(key=lambda c: c[1], reverse=True)
    return chains[:top_k]
