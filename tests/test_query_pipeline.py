"""Tests for Phase-3 query components and the assembled retrieval pipeline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sage.config import baseline
from sage.config.schema import PipelineConfig
from sage.core.types import ScoreSource, SearchResult, StoreRow, Strategy
from sage.correction.crag import CragCorrector
from sage.pipeline import build_retrieval_pipeline
from sage.routing import KeywordRouter
from sage.store import LanceDBStore
from sage.strategies.fusion import merge_deduplicate, reciprocal_rank_fusion
from sage.testing import FakeEmbedder, FakeGenerator, FakeReranker

warnings.filterwarnings("ignore")


def _results(ids: list[str], scores: list[float]) -> list[SearchResult]:
    return [
        SearchResult(chunk_id=c, document_id="d", content=c, relevance_score=s)
        for c, s in zip(ids, scores, strict=True)
    ]


# -- fusion ---------------------------------------------------------------


def test_merge_deduplicate_no_duplicates_and_sorted():
    a = _results(["c1", "c2"], [0.9, 0.5])
    b = _results(["c2", "c3"], [0.4, 0.8])
    out = merge_deduplicate([a, b])
    ids = [r.chunk_id for r in out]
    assert sorted(ids) == ["c1", "c2", "c3"]
    assert [r.relevance_score for r in out] == sorted(
        (r.relevance_score for r in out), reverse=True
    )


def test_rrf_matches_hand_computation():
    a = _results(["c1", "c2", "c3"], [0.9, 0.8, 0.7])
    b = _results(["c2", "c1"], [0.6, 0.5])
    out = reciprocal_rank_fusion([a, b], k=60)
    by_id = {r.chunk_id: r.relevance_score for r in out}
    # c1: rank0 in a (1/61) + rank1 in b (1/62); c2: rank1 in a (1/62) + rank0 in b (1/61)
    assert by_id["c1"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["c2"] == pytest.approx(1 / 62 + 1 / 61)
    assert all(r.score_source is ScoreSource.RRF for r in out)


def test_rrf_never_duplicates():
    a = _results(["c1", "c1"], [0.9, 0.8])  # degenerate input
    out = reciprocal_rank_fusion([a], k=60)
    assert len({r.chunk_id for r in out}) == len(out)


# -- keyword router -------------------------------------------------------


async def test_keyword_router_branches():
    router = KeywordRouter()
    qvec = np.zeros(4, dtype=np.float32)
    assert (await router.route("What is X?", qvec, None)).strategy is Strategy.HYDE  # type: ignore[arg-type]
    assert (
        await router.route("overview of the methodology", qvec, None)  # type: ignore[arg-type]
    ).strategy is Strategy.STEP_BACK
    assert (
        await router.route("section 3.2 error handling", qvec, None)  # type: ignore[arg-type]
    ).strategy is Strategy.SEMANTIC


# -- CRAG -----------------------------------------------------------------


async def test_crag_high_confidence(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=8)
    corrector = CragCorrector(PipelineConfig().correction)
    results = _results(["c1"], [0.9])
    outcome = await corrector.correct(
        "q",
        results,
        generator=FakeGenerator(response="95"),
        embedder=FakeEmbedder(dim=8),
        store=store,
    )
    assert outcome.raw_score == 95.0
    assert outcome.confidence.value == "high"


async def test_crag_low_triggers_rewrite(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=8)
    embedder = FakeEmbedder(dim=8)
    await store.upsert(
        [
            StoreRow(
                chunk_id="x0",
                document_id="d",
                chunk_index=0,
                content="rewritten target",
                embedding=await embedder.embed_query("rewritten target"),
                level=0,
            )
        ]
    )
    corrector = CragCorrector(PipelineConfig().correction)
    outcome = await corrector.correct(
        "q",
        _results(["c1"], [0.1]),
        generator=FakeGenerator(response="5"),  # low score -> rewrite
        embedder=embedder,
        store=store,
    )
    assert outcome.rewritten_query is not None


# -- full assembled pipeline ----------------------------------------------


async def test_baseline_pipeline_end_to_end(tmp_path):
    cfg = baseline(PipelineConfig())
    cfg.raptor.enabled = False  # keep the test focused on the query path
    embedder = FakeEmbedder(dim=16)
    store = LanceDBStore(tmp_path / "db", dim=16)

    from sage.core.types import StoreRow

    texts = [f"passage {i} about topic {i % 3}" for i in range(12)]
    embs = await embedder.embed_documents(texts)
    await store.upsert(
        [
            StoreRow(
                chunk_id=f"doc:{i}",
                document_id="doc",
                chunk_index=i,
                content=texts[i],
                embedding=embs[i],
                level=0,
            )
            for i in range(12)
        ]
    )

    pipeline = build_retrieval_pipeline(
        cfg,
        embedder=embedder,
        store=store,
        generator=FakeGenerator(response="80"),  # CRAG judge -> high/medium
        reranker=FakeReranker(),
    )
    results, trace = await pipeline.run("What is topic 1?", top_k=5)
    assert results
    assert len(results) <= 5
    stages = [s["stage"] for s in trace.stages]
    assert stages[0] == "route"
    assert "retrieve" in stages
    assert "rerank" in stages
    # reranked results carry the cross-encoder source
    assert any(r.score_source is ScoreSource.CROSS_ENCODER for r in results)
