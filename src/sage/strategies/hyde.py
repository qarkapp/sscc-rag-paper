"""Hypothetical-document retrieval (HyDE) with optional dual-path fusion.

The fusion variant selects the behaviour:

* ``single``     -- search with the hypothetical document only (single-path HyDE).
* ``merge_dedup`` -- search with both the query and the hypothetical document and
  union the results (the reference behaviour).
* ``rrf``        -- search with both and fuse by reciprocal rank (dual-path
  hypothesis fusion, DPHF).
"""

from __future__ import annotations

import numpy as np

from sage.config.schema import FusionCfg
from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.types import SearchResult
from sage.strategies.fusion import merge_deduplicate, reciprocal_rank_fusion

__all__ = ["HYDE_SYSTEM", "HydeStrategy"]

HYDE_SYSTEM = (
    "Given a question, write a 2-3 paragraph answer as if you had the perfect "
    "document in front of you. Be specific and factual."
)


class HydeStrategy:
    """Implements :class:`sage.core.protocols.RetrievalStrategy`."""

    def __init__(self, fusion: FusionCfg) -> None:
        self._fusion = fusion

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
        if generator is None:  # no generator -> fall back to dense search
            return await store.search(query_vector, top_k)

        hypothesis = await generator.complete(HYDE_SYSTEM, f"Question: {query}")
        hypo_vector = await embedder.embed_query(hypothesis)
        hypo_results = await store.search(hypo_vector, top_k)
        if self._fusion.variant == "single":
            return hypo_results

        query_results = await store.search(query_vector, top_k)
        if self._fusion.variant == "rrf":
            return reciprocal_rank_fusion([query_results, hypo_results], k=self._fusion.rrf_k)
        return merge_deduplicate([query_results, hypo_results])
