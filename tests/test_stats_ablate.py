"""Tests for statistics and the ablation runner."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sage.config import full
from sage.config.schema import PipelineConfig
from sage.core.types import StoreRow
from sage.eval.ablate import compare_to_reference, run_ablations
from sage.eval.dataset import QAExample, RetrievalDataset
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap_test,
    paired_diff_ci,
)
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder, FakeGenerator

warnings.filterwarnings("ignore")


# -- statistics ------------------------------------------------------------


def test_bootstrap_ci_brackets_mean():
    values = list(np.random.default_rng(0).normal(0.7, 0.1, 200))
    mean, lo, hi = bootstrap_ci(values, seed=1)
    assert lo < mean < hi
    assert abs(mean - 0.7) < 0.05


def test_paired_test_detects_consistent_difference():
    a = [0.8] * 50
    b = [0.5] * 50
    delta, lo, _hi = paired_diff_ci(a, b, seed=1)
    assert delta == pytest.approx(0.3) and lo > 0  # CI excludes zero
    assert paired_bootstrap_test(a, b, seed=1) < 0.05


def test_paired_test_nonsignificant_for_noise():
    rng = np.random.default_rng(0)
    a = list(rng.normal(0.5, 0.1, 100))
    b = list(rng.normal(0.5, 0.1, 100))
    assert paired_bootstrap_test(a, b, seed=2) > 0.05


def test_holm_and_bh_corrections():
    pvalues = [0.001, 0.02, 0.5, 0.9]
    holm = holm_bonferroni(pvalues, alpha=0.05)
    assert holm[0] is True and holm[2] is False
    bh = benjamini_hochberg(pvalues, q=0.05)
    assert bh[0] is True and bh[3] is False


# -- ablation runner -------------------------------------------------------


async def _indexed_store(tmp_path):
    embedder = FakeEmbedder(dim=32)
    store = LanceDBStore(tmp_path / "db", dim=32)
    passages = {f"p{i}": f"passage about unique topic {i}" for i in range(12)}
    embs = await embedder.embed_documents(list(passages.values()))
    await store.upsert(
        [
            StoreRow(
                chunk_id=pid,
                document_id=pid,
                chunk_index=0,
                content=passages[pid],
                embedding=embs[k],
                level=0,
            )
            for k, pid in enumerate(passages)
        ]
    )
    examples = [QAExample(qid=f"q{i}", question=passages[f"p{i}"]) for i in range(12)]
    qrels = {f"q{i}": {f"p{i}": 1} for i in range(12)}
    dataset = RetrievalDataset("syn", examples, passages, qrels)
    return embedder, store, dataset


async def test_run_ablations_and_compare(tmp_path):
    embedder, store, dataset = await _indexed_store(tmp_path)
    base = full(PipelineConfig())
    base.raptor.enabled = False  # keep ablations query-time only on a flat index
    base.graph.enabled = False

    outcomes = await run_ablations(
        ["full", "semantic_only", "wo_rerank"],
        base,
        dataset,
        embedder=embedder,
        store=store,
        generator=FakeGenerator(response="80"),
        reranker=None,
        top_k=5,
        primary_metric="Success@5",
    )
    assert {o.name for o in outcomes} == {"full", "semantic_only", "wo_rerank"}
    assert all(o.per_query for o in outcomes)

    comparisons = compare_to_reference(outcomes, "full")
    assert {c.name for c in comparisons} == {"semantic_only", "wo_rerank"}
    for c in comparisons:
        assert c.ci_low <= c.delta <= c.ci_high


def test_per_query_metric_alignment():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {"q1": {"d1": 0.9}, "q2": {"d9": 0.9}}
    scores = retrieval_metrics_per_query(qrels, run, "Success@5")
    assert scores["q1"] == 1.0
    assert scores["q2"] == 0.0
