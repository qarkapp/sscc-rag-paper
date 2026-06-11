"""Learned router: a small supervised classifier over distance features.

This is the supervised comparison point for EGR. It is trained on a dev split where
the best strategy per query is known, and shows how close EGR's label-free routing
comes to a trained classifier. Until fitted it falls back to semantic search.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sage.config.schema import RouterCfg
from sage.core.protocols import VectorStore
from sage.core.registry import register
from sage.core.types import Strategy, StrategyDecision
from sage.routing.features import distance_features, knn_distances

__all__ = ["LearnedRouter"]


@register("router", "learned")
class LearnedRouter:
    """Implements :class:`sage.core.protocols.Router`."""

    def __init__(self, cfg: RouterCfg) -> None:
        self._cfg = cfg
        self._model: object | None = None

    def fit(self, features: np.ndarray, labels: Sequence[Strategy]) -> None:
        """Train the classifier on distance features and oracle strategy labels."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, multi_class="auto")
        model.fit(features, [str(label) for label in labels])
        self._model = model

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision:
        if self._model is None:
            return StrategyDecision(strategy=Strategy.SEMANTIC, rationale="unfitted")
        distances = await knn_distances(query_vector, store, self._cfg.egr_k)
        features = distance_features(distances, self._cfg.egr_temperature).reshape(1, -1)
        predicted = self._model.predict(features)[0]  # type: ignore[attr-defined]
        return StrategyDecision(strategy=Strategy(predicted), rationale="learned")
