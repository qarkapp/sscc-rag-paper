"""Keyword-heuristic router (the reference baseline).

Routes interrogative queries to HyDE, broad/thematic queries to step-back, and the
rest to direct semantic search. This is the baseline the entropy-gated router is
measured against.
"""

from __future__ import annotations

import numpy as np

from sage.config.schema import RouterCfg
from sage.core.protocols import VectorStore
from sage.core.registry import register
from sage.core.types import Strategy, StrategyDecision

__all__ = ["KeywordRouter"]

_QUESTION_PREFIXES = (
    "what ",
    "why ",
    "how ",
    "who ",
    "when ",
    "where ",
    "which ",
    "is ",
    "are ",
    "does ",
    "do ",
    "can ",
)
_BROAD_TERMS = ("overview", "summary", "summarize", "general", "concept", "explain")


@register("router", "keyword")
class KeywordRouter:
    """Implements :class:`sage.core.protocols.Router`."""

    def __init__(self, cfg: RouterCfg | None = None) -> None:
        self._cfg = cfg

    async def route(
        self, query: str, query_vector: np.ndarray, store: VectorStore
    ) -> StrategyDecision:
        q = query.lower()
        is_question = "?" in q or q.startswith(_QUESTION_PREFIXES)
        is_broad = any(term in q for term in _BROAD_TERMS)
        if is_question and not is_broad:
            strategy = Strategy.HYDE
        elif is_broad:
            strategy = Strategy.STEP_BACK
        else:
            strategy = Strategy.SEMANTIC
        return StrategyDecision(strategy=strategy, rationale="keyword-heuristic")
