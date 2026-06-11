"""Speculative retrieval prefetching."""

from sage.prefetch.buffer import PrefetchBuffer
from sage.prefetch.metrics import PrefetchMetrics
from sage.prefetch.srp import SpeculativeRetrievalPrefetcher, extract_entities

__all__ = [
    "PrefetchBuffer",
    "PrefetchMetrics",
    "SpeculativeRetrievalPrefetcher",
    "extract_entities",
]
