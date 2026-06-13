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

        if self._fusion.hyde_expansion != "none":
            return await self._expanded(query, query_vector, store, embedder, generator, top_k)

        hypothesis = await generator.complete(HYDE_SYSTEM, f"Question: {query}")
        hypo_vector = await embedder.embed_query(hypothesis)
        hypo_results = await store.search(hypo_vector, top_k)
        if self._fusion.variant == "single":
            return hypo_results

        query_results = await store.search(query_vector, top_k)
        if self._fusion.variant == "rrf":
            return reciprocal_rank_fusion([query_results, hypo_results], k=self._fusion.rrf_k)
        return merge_deduplicate([query_results, hypo_results])

    async def _expanded(
        self,
        query: str,
        query_vector: np.ndarray,
        store: VectorStore,
        embedder: Embedder,
        generator: Generator,
        top_k: int,
    ) -> list[SearchResult]:
        """Multi-hypothetical HyDE: union the query path with one path per hypothetical.

        ``modality`` uses one hypothetical per content modality (the method); ``multi_prose``
        uses the same number of prose hypotheticals (the ensemble control).
        """
        from sage.strategies.modality_hyde import modality_hypotheticals, prose_hypotheticals

        if self._fusion.hyde_expansion == "modality":
            texts = list(
                (await modality_hypotheticals(
                    query, generator, modalities=self._fusion.modality_kinds
                )).values()
            )
        else:
            texts = await prose_hypotheticals(
                query, generator, n=len(self._fusion.modality_kinds)
            )
        pools = [await store.search(query_vector, top_k)]
        for text in texts:
            pools.append(await store.search(await embedder.embed_query(text), top_k))
        return merge_deduplicate(pools)
