"""Entropy-gated routing (EGR).

The distance distribution from the query to its nearest neighbours carries a signal
about how well the query is anchored in the corpus. A sharply peaked distribution
(low entropy) means a confident match -> direct semantic search; a flat distribution
(high entropy) means the query is diffuse/abstract -> step-back abstraction; in
between, dual-path hypothesis fusion. The thresholds are calibrated against an oracle
on a held-out split (see :mod:`sage.routing.calibration`).
"""

from __future__ import annotations

import re

import numpy as np

from sage.config.schema import RouterCfg
from sage.core.protocols import VectorStore
from sage.core.registry import register
from sage.core.types import Strategy, StrategyDecision

__all__ = ["EntropyGatedRouter", "routing_entropy"]

_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_MULTI_HOP_CUES = (" and ", " vs ", " versus ", "compare", "difference between", "both")


def routing_entropy(distances: np.ndarray, temperature: float) -> float:
    """Entropy of the softmax over negated nearest-neighbour distances."""
    logits = -np.asarray(distances, dtype=np.float64) / temperature
    logits -= logits.max()  # numerical stability
    weights = np.exp(logits)
    p = weights / weights.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def _entity_count(query: str) -> int:
    return len(set(_CAPITALIZED.findall(query)))


@register("router", "egr")
class EntropyGatedRouter:
    """Implements :class:`sage.core.protocols.Router`."""

    def __init__(self, cfg: RouterCfg) -> None:
        self._cfg = cfg

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision:
        hits = await store.search(query_vector, self._cfg.egr_k)
        if not hits:
            return StrategyDecision(strategy=Strategy.SEMANTIC, rationale="empty index")

        # Recover L2 distance from the stored relevance score (1 / (1 + d)).
        distances = np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits])
        entropy = routing_entropy(distances, self._cfg.egr_temperature)

        if entropy < self._cfg.egr_tau_low:
            strategy = Strategy.SEMANTIC
        elif entropy < self._cfg.egr_tau_high:
            strategy = Strategy.DPHF
        else:
            strategy = Strategy.STEP_BACK

        lowered = query.lower()
        multi_hop = strategy is Strategy.STEP_BACK and (
            _entity_count(query) >= 2 or any(cue in lowered for cue in _MULTI_HOP_CUES)
        )
        return StrategyDecision(
            strategy=strategy,
            entropy=entropy,
            knn_distances=distances,
            is_multi_hop=multi_hop,
            rationale=f"H={entropy:.3f}",
        )
