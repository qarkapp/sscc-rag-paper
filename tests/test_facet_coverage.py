"""Contract tests for facet-coverage selection."""

from __future__ import annotations

import numpy as np

from sage.strategies.facet_coverage import coverage_select, facet_relevance


def test_coverage_prefers_diversity_over_redundant_relevance():
    # Two facets. A and B both cover facet 0 strongly (B is a near-duplicate of A);
    # C is the only chunk covering facet 1. Plain top-2-by-relevance returns [A, B]
    # and misses facet 1; coverage selection must return A then C.
    ids = ["A", "B", "C"]
    rel = np.array([[1.0, 0.0],   # A -> facet 0
                    [0.9, 0.0],   # B -> facet 0 (redundant)
                    [0.0, 0.8]])  # C -> facet 1
    base = [0.95, 0.90, 0.50]     # raw relevance favors A, B

    order = coverage_select(ids, rel, base, k=2)
    assert order[:2] == ["A", "C"]  # diverse coverage, not the redundant duplicate
    # B still appears in the tail (full ranking preserved).
    assert set(order) == {"A", "B", "C"}


def test_single_facet_reduces_to_relevance_order():
    ids = ["x", "y", "z"]
    rel = np.array([[0.3], [0.9], [0.6]])  # one facet
    base = [0.3, 0.9, 0.6]
    # With a single facet, the first pick is the max-coverage chunk; the rest fall back
    # to raw relevance -> the overall order is the relevance order.
    assert coverage_select(ids, rel, base, k=3) == ["y", "z", "x"]


def test_facet_relevance_is_nonnegative_cosine():
    chunks = np.array([[1.0, 0.0], [0.0, 1.0]])
    facets = np.array([[1.0, 0.0], [-1.0, 0.0]])
    rel = facet_relevance(chunks, facets)
    assert rel.shape == (2, 2)
    assert rel.min() >= 0.0  # negatives clipped
    assert rel[0, 0] > 0.9   # chunk 0 aligns with facet 0
