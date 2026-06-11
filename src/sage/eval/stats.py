"""Statistical tools for honest comparisons.

Bootstrap confidence intervals, paired bootstrap significance tests over per-query
scores, and multiple-comparison correction (Holm-Bonferroni and Benjamini-Hochberg).
A result counts as an improvement only when its corrected CI excludes zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "bootstrap_ci",
    "holm_bonferroni",
    "paired_bootstrap_test",
    "paired_diff_ci",
]


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` for a percentile bootstrap CI of the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(arr.mean()), float(lo), float(hi)


def paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI of the paired mean difference ``a - b`` (same queries, aligned)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_ci(diff, confidence=confidence, n_resamples=n_resamples, seed=seed)


def paired_bootstrap_test(
    a: Sequence[float], b: Sequence[float], *, n_resamples: int = 10000, seed: int = 0
) -> float:
    """Two-sided paired bootstrap p-value for ``mean(a) == mean(b)``.

    Resamples the per-query differences and measures how often the bootstrap mean
    falls on the opposite side of zero from the observed mean.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if diff.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_resamples, diff.size))
    boot_means = diff[idx].mean(axis=1)
    # Center on zero (the null) and compare to the observed mean.
    observed = diff.mean()
    centered = boot_means - boot_means.mean()
    p = float(np.mean(np.abs(centered) >= abs(observed)))
    return min(1.0, max(p, 1.0 / n_resamples))


def holm_bonferroni(pvalues: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down: return a reject mask in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        threshold = alpha / (m - rank)
        if p[i] <= threshold:
            reject[i] = True
        else:
            break  # once one fails, all higher p-values fail too
    return reject.tolist()


def benjamini_hochberg(pvalues: Sequence[float], *, q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR control: return a reject mask in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    reject = np.zeros(m, dtype=bool)
    max_rank = -1
    for rank, i in enumerate(order, start=1):
        if p[i] <= q * rank / m:
            max_rank = rank
    if max_rank > 0:
        for rank, i in enumerate(order, start=1):
            if rank <= max_rank:
                reject[i] = True
    return reject.tolist()
