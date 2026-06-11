"""Direct semantic (dense) retrieval strategy."""

from __future__ import annotations

import numpy as np

from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.registry import register
from sage.core.types import SearchResult

__all__ = ["SemanticStrategy"]


@register("strategy", "semantic")
class SemanticStrategy:
    """Implements :class:`sage.core.protocols.RetrievalStrategy`."""

    async def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator | None,
        top_k: int,
    ) -> list[SearchResult]:
        return await store.search(query_vector, top_k)
