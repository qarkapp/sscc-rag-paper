"""Structural interfaces for every swappable component.

Everything in the pipeline depends only on these protocols, never on concrete
classes. This is what lets the call-cache wrap backends transparently, lets tests
substitute deterministic fakes, and lets an ablation be a config diff rather than a
code change.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from sage.core.types import (
    Chunk,
    CorrectionOutcome,
    DocumentSection,
    SearchResult,
    StoreRow,
    StrategyDecision,
)

__all__ = [
    "Chunker",
    "Corrector",
    "Embedder",
    "Generator",
    "Parser",
    "Reranker",
    "RetrievalStrategy",
    "Router",
    "VectorStore",
]


@runtime_checkable
class Embedder(Protocol):
    """Maps text to dense vectors. Backed by oMLX in production."""

    @property
    def dim(self) -> int:
        """Embedding dimensionality (resolved via runtime probing)."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of passages. Returns an ``(n, dim)`` float32 array."""
        ...

    async def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query. Returns a ``(dim,)`` float32 array."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder reranker exposed via an OpenAI-style ``/rerank`` endpoint."""

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` pairs, ordered best-first."""
        ...


@runtime_checkable
class Generator(Protocol):
    """Chat-completion generator (oMLX local or OpenRouter)."""

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Return a single completion string."""
        ...

    def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield completion tokens as they arrive (used by speculative prefetch)."""
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent dense index over chunks and summary nodes."""

    async def upsert(self, rows: Sequence[StoreRow]) -> None:
        """Insert or replace rows by ``chunk_id``."""
        ...

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        document_filter: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Nearest-neighbour search over leaf chunks (level 0)."""
        ...

    async def search_by_level(
        self, query_vector: np.ndarray, top_k: int, level: int
    ) -> list[SearchResult]:
        """Nearest-neighbour search restricted to a tree level."""
        ...

    async def get_by_ids(self, ids: Sequence[str]) -> list[StoreRow]:
        """Fetch rows (including embeddings) by id."""
        ...

    async def all_leaf_rows(self) -> list[StoreRow]:
        """Return every leaf row (level 0), e.g. for building the chunk graph."""
        ...


@runtime_checkable
class Parser(Protocol):
    """Turns a raw document into ordered sections prior to chunking."""

    def parse(self, data: bytes, filename: str) -> list[DocumentSection]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits parsed sections into indexable chunks."""

    def chunk(
        self, document_id: str, sections: Sequence[DocumentSection], filename: str
    ) -> list[Chunk]: ...


@runtime_checkable
class Router(Protocol):
    """Selects a first-stage retrieval strategy for a query."""

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision: ...


@runtime_checkable
class RetrievalStrategy(Protocol):
    """Produces an initial ranked candidate set for a query."""

    async def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator | None,
        top_k: int,
    ) -> list[SearchResult]: ...


@runtime_checkable
class Corrector(Protocol):
    """Filters/augments results and assigns a confidence (CRAG / SSCC)."""

    async def correct(
        self,
        query: str,
        results: list[SearchResult],
        *,
        generator: Generator,
        embedder: Embedder,
        store: VectorStore,
    ) -> CorrectionOutcome: ...
