"""Calibrate per-source SSCC thresholds on a labelled dev split.

For each score source, choose the threshold that maximizes balanced accuracy of the
keep/drop decision against gold relevance labels. Bi-encoder and cross-encoder
thresholds are fit independently because their score scales differ.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["calibrate_threshold"]


def calibrate_threshold(
    scores: Sequence[float], labels: Sequence[bool], *, grid: int = 100
) -> float:
    """Return the threshold maximizing balanced accuracy of ``score >= tau``."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.size == 0 or y.all() or (~y).all():
        return float(np.median(s)) if s.size else 0.0

    candidates = np.linspace(s.min(), s.max(), grid)
    pos, neg = y.sum(), (~y).sum()
    best_tau, best_score = float(candidates[0]), -1.0
    for tau in candidates:
        predicted = s >= tau
        tpr = float((predicted & y).sum()) / pos
        tnr = float((~predicted & ~y).sum()) / neg
        balanced = 0.5 * (tpr + tnr)
        if balanced > best_score:
            best_tau, best_score = float(tau), balanced
    return best_tau
