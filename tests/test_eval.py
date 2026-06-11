"""Tests for the evaluation harness: metrics, span mapping, and the runner."""

from __future__ import annotations

import warnings

from sage.config import semantic_only
from sage.config.schema import PipelineConfig
from sage.eval import (
    ChunkSpan,
    GoldSpan,
    build_qrels,
    evaluate_retrieval,
    exact_match,
    index_passages,
    relevant_chunk_ids,
    retrieval_metrics,
    token_f1,
)
from sage.eval.dataset import QAExample, RetrievalDataset
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder

warnings.filterwarnings("ignore")


# -- retrieval metrics -----------------------------------------------------


def test_perfect_run_scores_one():
    qrels = {"q1": {"d1": 1, "d2": 1}}
    run = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.1}}
    m = retrieval_metrics(qrels, run, ["nDCG@10", "Success@5", "R@2"])
    assert m["nDCG@10"] == 1.0
    assert m["Success@5"] == 1.0
    assert m["R@2"] == 1.0


def test_irrelevant_run_scores_zero():
    qrels = {"q1": {"d1": 1}}
    run = {"q1": {"d9": 0.9, "d8": 0.8}}
    m = retrieval_metrics(qrels, run, ["Success@5"])
    assert m["Success@5"] == 0.0


# -- answer metrics --------------------------------------------------------


def test_exact_match_and_f1_normalization():
    assert exact_match("The Answer.", "answer") == 1.0
    assert token_f1("the quick brown fox", "quick brown fox") > 0.8
    assert token_f1("completely different", "quick brown fox") == 0.0


# -- span mapping ----------------------------------------------------------


def test_span_to_chunk_overlap():
    gold = [GoldSpan("doc", 100, 200, grade=1)]
    chunks = [
        ChunkSpan("doc:0", "doc", 0, 90),  # no overlap
        ChunkSpan("doc:1", "doc", 90, 210),  # contains the span
        ChunkSpan("doc:2", "doc", 180, 300),  # 20% overlap -> below threshold
    ]
    relevant = relevant_chunk_ids(gold, chunks, min_overlap=0.5)
    assert relevant == {"doc:1": 1}


def test_span_mapping_is_document_scoped():
    gold = [GoldSpan("docA", 0, 100)]
    chunks = [ChunkSpan("docB:0", "docB", 0, 100)]  # right offsets, wrong document
    assert relevant_chunk_ids(gold, chunks) == {}


def test_build_qrels_per_query():
    gold = {"q1": [GoldSpan("doc", 0, 50)]}
    chunks = [ChunkSpan("doc:0", "doc", 0, 60)]
    qrels = build_qrels(gold, chunks)
    assert qrels == {"q1": {"doc:0": 1}}


# -- end-to-end runner -----------------------------------------------------


async def test_evaluate_retrieval_end_to_end(tmp_path):
    # Build a dataset where each question text equals its gold passage text, so the
    # deterministic FakeEmbedder maps them to the same vector and retrieves it.
    passages = {f"p{i}": f"passage about distinct topic number {i}" for i in range(8)}
    examples = [QAExample(qid=f"q{i}", question=passages[f"p{i}"]) for i in range(8)]
    qrels = {f"q{i}": {f"p{i}": 1} for i in range(8)}
    dataset = RetrievalDataset("synthetic", examples, passages, qrels)

    cfg = semantic_only(PipelineConfig())
    embedder = FakeEmbedder(dim=32)
    store = LanceDBStore(tmp_path / "db", dim=32)
    await index_passages(store, embedder, dataset.corpus)

    pipeline = build_retrieval_pipeline(cfg, embedder=embedder, store=store)
    result = await evaluate_retrieval(pipeline, dataset, top_k=5)
    assert result.n_queries == 8
    # Each query's gold passage has a unique embedding -> retrieved rank 1.
    assert result.metrics["Success@5"] == 1.0
    assert result.metrics["nDCG@10"] == 1.0
