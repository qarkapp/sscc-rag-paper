"""Calibrate entropy-gated routing thresholds against an oracle.

A grid search over candidate ``(tau_low, tau_high)`` pairs maximizes agreement
between the entropy-router's decision and the oracle's best strategy on a held-out
calibration split. This produces the honest analogue of the reference paper's
asserted routing-agreement number.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sage.core.types import Strategy

__all__ = ["calibrate_thresholds", "routing_agreement"]


def _decide(entropy: float, tau_low: float, tau_high: float) -> Strategy:
    if entropy < tau_low:
        return Strategy.SEMANTIC
    if entropy < tau_high:
        return Strategy.DPHF
    return Strategy.STEP_BACK


def routing_agreement(
    entropies: Sequence[float],
    oracle: Sequence[Strategy],
    tau_low: float,
    tau_high: float,
) -> float:
    """Fraction of queries where the entropy decision matches the oracle."""
    if not entropies:
        return 0.0
    # HyDE and DPHF are the same dual-path strategy for routing purposes.
    matches = 0
    for h, target in zip(entropies, oracle, strict=True):
        decision = _decide(h, tau_low, tau_high)
        if decision is target or {decision, target} == {Strategy.HYDE, Strategy.DPHF}:
            matches += 1
    return matches / len(entropies)


def calibrate_thresholds(
    entropies: Sequence[float],
    oracle: Sequence[Strategy],
    *,
    grid: int = 40,
) -> tuple[float, float, float]:
    """Return ``(tau_low, tau_high, agreement)`` maximizing oracle agreement."""
    if not entropies:
        return 1.8, 2.6, 0.0
    lo, hi = float(np.min(entropies)), float(np.max(entropies))
    candidates = np.linspace(lo, hi, grid)
    best = (1.8, 2.6, -1.0)
    for i, tau_low in enumerate(candidates):
        for tau_high in candidates[i + 1 :]:
            agreement = routing_agreement(entropies, oracle, tau_low, tau_high)
            if agreement > best[2]:
                best = (float(tau_low), float(tau_high), agreement)
    return best
