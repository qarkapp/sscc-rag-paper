"""Prefetch buffer keyed by query embedding."""

from __future__ import annotations

import numpy as np

from sage.core.types import SearchResult

__all__ = ["PrefetchBuffer"]


class PrefetchBuffer:
    """Stores speculatively-retrieved results, looked up by embedding similarity."""

    def __init__(self, threshold: float = 0.8, capacity: int = 64) -> None:
        self._threshold = threshold
        self._capacity = capacity
        self._entries: list[tuple[np.ndarray, list[SearchResult]]] = []

    def put(self, query_vector: np.ndarray, results: list[SearchResult]) -> None:
        vec = _normalize(query_vector)
        self._entries.append((vec, results))
        if len(self._entries) > self._capacity:
            self._entries.pop(0)

    def lookup(self, query_vector: np.ndarray) -> list[SearchResult] | None:
        if not self._entries:
            return None
        q = _normalize(query_vector)
        best_sim, best_results = -1.0, None
        for vec, results in self._entries:
            sim = float(q @ vec)
            if sim > best_sim:
                best_sim, best_results = sim, results
        return best_results if best_sim >= self._threshold else None

    def __len__(self) -> int:
        return len(self._entries)


def _normalize(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)
