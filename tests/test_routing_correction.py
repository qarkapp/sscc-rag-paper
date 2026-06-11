"""Tests for entropy-gated routing, the routing triad, and SSCC."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sage.config.schema import CorrectionCfg, PipelineConfig, RouterCfg
from sage.core.types import Confidence, ScoreSource, SearchResult, StoreRow, Strategy
from sage.correction.calibration import calibrate_threshold
from sage.correction.sscc import SsccCorrector
from sage.routing.calibration import calibrate_thresholds, routing_agreement
from sage.routing.egr import EntropyGatedRouter, routing_entropy
from sage.routing.oracle import OracleRouter
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder, FakeGenerator

warnings.filterwarnings("ignore")


# -- entropy ---------------------------------------------------------------


def test_entropy_low_for_peaked_high_for_flat():
    peaked = routing_entropy(np.array([0.0, 5.0, 6.0, 7.0]), temperature=1.0)
    flat = routing_entropy(np.array([1.0, 1.0, 1.0, 1.0]), temperature=1.0)
    assert flat > peaked
    # uniform distribution over k reaches log k
    assert flat == pytest.approx(np_log_k(4), abs=1e-9)


def np_log_k(k: int) -> float:
    return float(np.log(k))


async def test_egr_routes_by_entropy(tmp_path):
    embedder = FakeEmbedder(dim=16)
    store = LanceDBStore(tmp_path / "db", dim=16)
    rows = [
        StoreRow(
            chunk_id=f"c{i}",
            document_id="d",
            chunk_index=i,
            content=f"passage {i}",
            embedding=(await embedder.embed_documents([f"passage {i}"]))[0],
            level=0,
        )
        for i in range(30)
    ]
    await store.upsert(rows)
    router = EntropyGatedRouter(RouterCfg(egr_k=20))
    decision = await router.route("a query", await embedder.embed_query("a query"), store)
    assert decision.strategy in {Strategy.SEMANTIC, Strategy.DPHF, Strategy.STEP_BACK}
    assert decision.entropy is not None and decision.entropy >= 0.0


# -- routing triad / calibration ------------------------------------------


async def test_oracle_router_returns_assigned():
    router = OracleRouter()
    router.set_decisions({"q1": Strategy.STEP_BACK})
    d = await router.route("q1", np.zeros(4, dtype=np.float32), None)  # type: ignore[arg-type]
    assert d.strategy is Strategy.STEP_BACK


def test_threshold_calibration_maximizes_agreement():
    # Low entropy -> semantic; high -> step_back; middle -> dphf.
    entropies = [0.1, 0.2, 1.5, 1.6, 2.9, 3.0]
    oracle = [
        Strategy.SEMANTIC,
        Strategy.SEMANTIC,
        Strategy.DPHF,
        Strategy.DPHF,
        Strategy.STEP_BACK,
        Strategy.STEP_BACK,
    ]
    tau_low, tau_high, agreement = calibrate_thresholds(entropies, oracle)
    assert tau_low < tau_high
    assert agreement >= 0.8
    # the calibrated thresholds reproduce that agreement
    assert routing_agreement(entropies, oracle, tau_low, tau_high) == agreement


# -- SSCC ------------------------------------------------------------------


def _result(cid: str, score: float, source: ScoreSource) -> SearchResult:
    return SearchResult(
        chunk_id=cid, document_id="d", content=cid, relevance_score=score, score_source=source
    )


async def test_sscc_uses_per_source_thresholds(tmp_path):
    cfg = CorrectionCfg(sscc_tau_bi=0.5, sscc_tau_cross=0.0)
    corrector = SsccCorrector(cfg)
    assert corrector.threshold_for(ScoreSource.BI_ENCODER) == 0.5
    assert corrector.threshold_for(ScoreSource.CROSS_ENCODER) == 0.0

    # A cross-encoder logit of 0.3 is kept (>= 0.0) where a bi-encoder 0.3 would be dropped.
    cross = [_result("c1", 0.3, ScoreSource.CROSS_ENCODER)]
    outcome = await corrector.correct(
        "q",
        cross,
        generator=FakeGenerator(),
        embedder=FakeEmbedder(8),
        store=LanceDBStore(tmp_path / "db", dim=8),
    )
    assert [r.chunk_id for r in outcome.results] == ["c1"]
    assert outcome.confidence in {Confidence.HIGH, Confidence.MEDIUM}


async def test_sscc_rewrites_when_all_filtered(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=8)
    embedder = FakeEmbedder(dim=8)
    await store.upsert(
        [
            StoreRow(
                chunk_id="x0",
                document_id="d",
                chunk_index=0,
                content="target",
                embedding=await embedder.embed_query("target"),
                level=0,
            )
        ]
    )
    corrector = SsccCorrector(CorrectionCfg(sscc_tau_bi=0.99))
    outcome = await corrector.correct(
        "q",
        [_result("c1", 0.1, ScoreSource.BI_ENCODER)],  # below threshold -> all filtered
        generator=FakeGenerator(response="better query"),
        embedder=embedder,
        store=store,
    )
    assert outcome.rewritten_query == "better query"


def test_sscc_threshold_calibration_separates_classes():
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [False, False, False, True, True, True]
    tau = calibrate_threshold(scores, labels)
    assert 0.3 < tau <= 0.7


def test_full_preset_assembles(tmp_path):
    # egr + sscc are now registered, so the full preset can be assembled.
    from sage.config import full
    from sage.pipeline import build_retrieval_pipeline

    cfg = full(PipelineConfig())
    pipeline = build_retrieval_pipeline(
        cfg,
        embedder=FakeEmbedder(8),
        store=LanceDBStore(tmp_path / "db", dim=8),
        generator=FakeGenerator(),
        reranker=None,
    )
    assert pipeline.router.__class__.__name__ == "EntropyGatedRouter"
    assert pipeline.corrector.__class__.__name__ == "SsccCorrector"
