"""Deterministic fakes implementing the backend protocols.

The embedder maps text to a stable hashed vector, so semantically unrelated texts
get unrelated vectors while identical texts always map to the same vector. This is
enough to exercise routing, fusion, correction, and indexing logic without a
backend.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence

import numpy as np

__all__ = ["FakeEmbedder", "FakeGenerator", "FakeReranker"]


class FakeEmbedder:
    """Hash-based deterministic embedder."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def probe(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dim).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


class FakeReranker:
    """Lexical-overlap reranker: scores by shared-token fraction with the query."""

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        q = set(query.lower().split())
        scored: list[tuple[int, float]] = []
        for i, doc in enumerate(documents):
            tokens = set(doc.lower().split())
            overlap = len(q & tokens) / (len(q) + 1e-9)
            scored.append((i, float(overlap)))
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_n]


class FakeGenerator:
    """Template generator that echoes a deterministic transform of its input."""

    def __init__(self, response: str | None = None) -> None:
        self._response = response

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        if self._response is not None:
            return self._response
        return f"[{system[:16]}] {user}"

    async def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        text = await self.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        for token in text.split():
            yield token + " "
