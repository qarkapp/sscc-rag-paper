"""Compositional router: per-strategy reward regression over query-aware features.

The entropy-gated router (:mod:`sage.routing.egr`) routes on a single scalar -- the
entropy of the kNN distance distribution. On normalized embeddings that scalar is
degenerate: the in-query distances are nearly equal, so the softmax is uniform and the
entropy pins to its ``log K`` ceiling for *every* query (between/within class
dispersion ~0). It cannot route.

This router instead predicts, for each candidate strategy, the *retrieval reward* that
strategy would earn on the query, and routes to the arg-max. The reward is regressed
from features that actually vary across queries and carry intent:

* **intent**     -- question type, comparative/multi-hop cues, entity and clause counts
                    (what the keyword heuristic reads, but as continuous features);
* **anchoring**  -- the *scale* of the kNN distances (min / mean / spread / gap), not
                    their entropy: how well the query is grounded in the corpus;
* **semantic**   -- a low-rank projection of the query embedding, so the regressor can
                    pick up intent the lexical cues miss.

Training targets are the per-query retrieval metric under each forced strategy, so the
objective is the metric itself -- not label agreement, which is misleading here (the
most-often-best strategy can still have the worst mean metric). The model is fit on a
disjoint split; :func:`cross_fit_decisions` provides leakage-free decisions for every
query via k-fold cross-fitting.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from sage.config.schema import RouterCfg
from sage.core.protocols import VectorStore
from sage.core.registry import register
from sage.core.types import Strategy, StrategyDecision

__all__ = ["CompositionalRouter", "RouterFeatures", "cross_fit_decisions"]

# Strategies the router chooses among (HyDE's dual-path form is DPHF).
ROUTED_STRATEGIES: tuple[Strategy, ...] = (Strategy.SEMANTIC, Strategy.DPHF, Strategy.STEP_BACK)

_CAPITALIZED = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_QUESTION_PREFIXES = (
    "what", "why", "how", "who", "when", "where", "which",
    "is", "are", "does", "do", "can", "list", "name",
)
_BROAD_TERMS = ("overview", "summary", "summarize", "general", "concept", "explain", "describe")
_MULTI_HOP_CUES = (" and ", " vs ", " versus ", "compare", "difference between", "both", "as well as")

# Length of the intent block; keep in sync with ``_intent_features``.
_N_INTENT = len(_QUESTION_PREFIXES) + 6
_N_ANCHOR = 6


def _intent_features(query: str) -> np.ndarray:
    """Continuous query-intent features (lexical signal, the keyword router generalized)."""
    q = query.strip()
    lower = q.lower()
    tokens = lower.split()
    feats: list[float] = []
    first = tokens[0] if tokens else ""
    feats.extend(1.0 if first == p else 0.0 for p in _QUESTION_PREFIXES)
    feats.append(1.0 if "?" in q else 0.0)
    feats.append(1.0 if any(t in lower for t in _BROAD_TERMS) else 0.0)
    feats.append(1.0 if any(c in lower for c in _MULTI_HOP_CUES) else 0.0)
    feats.append(float(len(set(_CAPITALIZED.findall(q)))))          # entity count
    feats.append(float(len(tokens)))                                # query length
    feats.append(float(lower.count(",") + lower.count(" and ")))    # clause count
    return np.asarray(feats, dtype=np.float64)


def _anchoring_features(distances: np.ndarray) -> np.ndarray:
    """Scale (not entropy) of the kNN distance distribution -- how grounded the query is."""
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        return np.zeros(_N_ANCHOR, dtype=np.float64)
    dmin, dmean, dmax = float(d.min()), float(d.mean()), float(d.max())
    nearest_gap = float(np.sort(d)[1] - dmin) if d.size > 1 else 0.0
    return np.asarray(
        [dmin, dmean, dmax - dmin, float(d.std()), nearest_gap, dmin / (dmean + 1e-9)],
        dtype=np.float64,
    )


class RouterFeatures:
    """Assembles and standardizes [intent || anchoring || query-embedding-PCA] vectors.

    Scaler and PCA are fit on the training queries only and reused at inference, so the
    transform never sees held-out data.
    """

    def __init__(self, pca_dims: int) -> None:
        self._pca_dims = pca_dims
        self._scaler: object | None = None
        self._pca: object | None = None

    def fit(
        self, queries: Sequence[str], distances: Sequence[np.ndarray], embeddings: np.ndarray
    ) -> np.ndarray:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        lexical = np.vstack(
            [np.concatenate([_intent_features(q), _anchoring_features(d)])
             for q, d in zip(queries, distances, strict=True)]
        )
        self._scaler = StandardScaler().fit(lexical)
        scaled = self._scaler.transform(lexical)  # type: ignore[union-attr]
        dims = min(self._pca_dims, embeddings.shape[1], max(1, embeddings.shape[0] - 1))
        self._pca = PCA(n_components=dims, random_state=0).fit(embeddings)
        emb = self._pca.transform(embeddings)  # type: ignore[union-attr]
        return np.hstack([scaled, emb])

    def transform(
        self, query: str, distances: np.ndarray, embedding: np.ndarray
    ) -> np.ndarray:
        assert self._scaler is not None and self._pca is not None, "RouterFeatures not fitted"
        lexical = np.concatenate([_intent_features(query), _anchoring_features(distances)])
        scaled = self._scaler.transform(lexical.reshape(1, -1))  # type: ignore[attr-defined]
        emb = self._pca.transform(embedding.reshape(1, -1))  # type: ignore[attr-defined]
        return np.hstack([scaled, emb])


@register("router", "compositional")
class CompositionalRouter:
    """Implements :class:`sage.core.protocols.Router`. Falls back to DPHF until fitted."""

    def __init__(self, cfg: RouterCfg) -> None:
        self._cfg = cfg
        self._features = RouterFeatures(cfg.comp_pca_dims)
        self._models: dict[Strategy, object] = {}

    def fit(
        self,
        queries: Sequence[str],
        distances: Sequence[np.ndarray],
        embeddings: np.ndarray,
        rewards: dict[Strategy, np.ndarray],
    ) -> None:
        """Fit one ridge reward-regressor per strategy on a training split."""
        from sklearn.linear_model import Ridge

        x = self._features.fit(queries, distances, embeddings)
        for strat in ROUTED_STRATEGIES:
            model = Ridge(alpha=self._cfg.comp_alpha)
            model.fit(x, np.asarray(rewards[strat], dtype=np.float64))
            self._models[strat] = model

    def predict_strategy(
        self, query: str, distances: np.ndarray, embedding: np.ndarray
    ) -> Strategy:
        """Arg-max predicted reward over routed strategies."""
        x = self._features.transform(query, distances, embedding)
        scores = {s: float(m.predict(x)[0]) for s, m in self._models.items()}  # type: ignore[attr-defined]
        return max(ROUTED_STRATEGIES, key=lambda s: scores.get(s, -np.inf))

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision:
        if not self._models:
            return StrategyDecision(strategy=Strategy.DPHF, rationale="unfitted")
        hits = await store.search(query_vector, self._cfg.egr_k)
        distances = np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits])
        strategy = self.predict_strategy(query, distances, query_vector)
        lower = query.lower()
        multi_hop = strategy is Strategy.STEP_BACK and any(c in lower for c in _MULTI_HOP_CUES)
        return StrategyDecision(
            strategy=strategy, knn_distances=distances, is_multi_hop=multi_hop,
            rationale="compositional",
        )


def cross_fit_decisions(
    cfg: RouterCfg,
    qids: Sequence[str],
    queries: Sequence[str],
    distances: Sequence[np.ndarray],
    embeddings: np.ndarray,
    rewards: dict[Strategy, np.ndarray],
) -> dict[str, Strategy]:
    """Leakage-free per-query decisions via k-fold cross-fitting.

    Each query is routed by a model trained only on the *other* folds, so the resulting
    decisions can be evaluated on the full set without train/test contamination.
    """
    from sklearn.model_selection import KFold

    n = len(qids)
    folds = max(2, min(cfg.comp_folds, n))
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    decisions: dict[str, Strategy] = {}
    for train_idx, test_idx in kf.split(np.arange(n)):
        router = CompositionalRouter(cfg)
        router.fit(
            [queries[i] for i in train_idx],
            [distances[i] for i in train_idx],
            embeddings[train_idx],
            {s: rewards[s][train_idx] for s in ROUTED_STRATEGIES},
        )
        for i in test_idx:
            decisions[qids[i]] = router.predict_strategy(queries[i], distances[i], embeddings[i])
    return decisions
