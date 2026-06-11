"""Speculative retrieval prefetching (SRP).

During multi-turn generation, entities mentioned in the model's output stream often
anticipate the user's next information need. SRP extracts entities from the streamed
tokens, speculatively retrieves for them in the background, and serves a later query
from the buffer when it is similar enough to a prefetched one -- turning a retrieval
round trip into a near-instant buffer hit.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

import numpy as np

from sage.config.schema import PrefetchCfg
from sage.core.protocols import Embedder
from sage.core.types import SearchResult
from sage.prefetch.buffer import PrefetchBuffer
from sage.prefetch.metrics import PrefetchMetrics

__all__ = ["SpeculativeRetrievalPrefetcher", "extract_entities"]

# Proper-noun-like spans: capitalized words, optionally multi-word.
_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b")
_RetrieveFn = Callable[[str], Awaitable[list[SearchResult]]]


def extract_entities(text: str) -> list[str]:
    """Extract candidate entities (proper-noun-like spans) from text."""
    seen: dict[str, None] = {}
    for match in _ENTITY.findall(text):
        if len(match) >= 3:
            seen.setdefault(match, None)
    return list(seen)


class SpeculativeRetrievalPrefetcher:
    """Prefetches retrievals for entities seen in a generation stream."""

    def __init__(self, cfg: PrefetchCfg, embedder: Embedder, retrieve: _RetrieveFn) -> None:
        self._cfg = cfg
        self._embedder = embedder
        self._retrieve = retrieve
        self._buffer = PrefetchBuffer(threshold=cfg.hit_cosine_threshold)
        self._seen: set[str] = set()
        self.metrics = PrefetchMetrics()

    async def observe(self, text: str) -> int:
        """Prefetch for any new entities in ``text``. Returns the count prefetched."""
        prefetched = 0
        for entity in extract_entities(text):
            if entity in self._seen:
                continue
            self._seen.add(entity)
            results = await self._retrieve(entity)
            self._buffer.put(await self._embedder.embed_query(entity), results)
            prefetched += 1
        return prefetched

    async def maybe_hit(self, query_vector: np.ndarray) -> list[SearchResult] | None:
        """Return buffered results if the query matches a prefetched one."""
        hit = self._buffer.lookup(query_vector)
        self.metrics.record(hit is not None)
        return hit
