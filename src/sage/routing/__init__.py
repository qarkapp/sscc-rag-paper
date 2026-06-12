"""Query routing: keyword baseline, entropy-gated, oracle, and learned routers."""

from sage.routing.calibration import calibrate_thresholds, routing_agreement
from sage.routing.compositional import CompositionalRouter, cross_fit_decisions
from sage.routing.egr import EntropyGatedRouter, routing_entropy
from sage.routing.keyword import KeywordRouter
from sage.routing.learned import LearnedRouter
from sage.routing.oracle import OracleRouter

__all__ = [
    "CompositionalRouter",
    "EntropyGatedRouter",
    "KeywordRouter",
    "LearnedRouter",
    "OracleRouter",
    "calibrate_thresholds",
    "cross_fit_decisions",
    "routing_agreement",
    "routing_entropy",
]
