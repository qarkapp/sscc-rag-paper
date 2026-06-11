"""Oracle router: an evaluation-only upper bound.

Given a per-query mapping to the strategy that maximizes that query's retrieval
metric (computed offline by trying each strategy), the oracle router returns it. It
quantifies the maximum achievable routing gain and is never used in deployment.
"""

from __future__ import annotations

import numpy as np

from sage.config.schema import RouterCfg
from sage.core.protocols import VectorStore
from sage.core.registry import register
from sage.core.types import Strategy, StrategyDecision

__all__ = ["OracleRouter"]


@register("router", "oracle")
class OracleRouter:
    """Implements :class:`sage.core.protocols.Router`."""

    def __init__(self, cfg: RouterCfg | None = None) -> None:
        self._cfg = cfg
        self.decisions: dict[str, Strategy] = {}

    def set_decisions(self, decisions: dict[str, Strategy]) -> None:
        """Provide the offline-computed best strategy per query."""
        self.decisions = dict(decisions)

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision:
        strategy = self.decisions.get(query, Strategy.SEMANTIC)
        return StrategyDecision(strategy=strategy, rationale="oracle")
