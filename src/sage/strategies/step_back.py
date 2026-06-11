"""Step-back retrieval: abstract the query, then search with both forms."""

from __future__ import annotations

import numpy as np

from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.types import SearchResult
from sage.strategies.fusion import merge_deduplicate

__all__ = ["STEP_BACK_SYSTEM", "StepBackStrategy"]

STEP_BACK_SYSTEM = (
    "You generate abstract search queries. Given a specific question, produce a "
    "broader, more general version of it. Return ONLY the query, no explanation."
)


class StepBackStrategy:
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
        if generator is None:
            return await store.search(query_vector, top_k)

        abstract = await generator.complete(STEP_BACK_SYSTEM, query)
        abstract_vector = await embedder.embed_query(abstract)
        query_results = await store.search(query_vector, top_k)
        abstract_results = await store.search(abstract_vector, max(1, top_k // 2))
        return merge_deduplicate([query_results, abstract_results])
