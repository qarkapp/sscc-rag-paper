"""Shared nearest-neighbour distance features for routing."""

from __future__ import annotations

import numpy as np

from sage.core.protocols import VectorStore
from sage.routing.egr import routing_entropy

__all__ = ["distance_features", "knn_distances"]


async def knn_distances(query_vector: np.ndarray, store: VectorStore, k: int) -> np.ndarray:
    """Recover the L2 distances to the top-``k`` neighbours from their scores."""
    hits = await store.search(query_vector, k)
    return np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits])


def distance_features(distances: np.ndarray, temperature: float) -> np.ndarray:
    """A small feature vector summarizing the neighbour-distance distribution."""
    if distances.size == 0:
        return np.zeros(5, dtype=np.float64)
    return np.array(
        [
            routing_entropy(distances, temperature),
            float(distances.mean()),
            float(distances.std()),
            float(distances.min()),
            float(distances.max()),
        ]
    )
