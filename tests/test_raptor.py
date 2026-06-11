"""Tests for the RAPTOR subsystem (clustering, tree, retrieval, cross-document)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from sage.config.schema import RaptorCfg
from sage.core.types import StoreRow
from sage.raptor import (
    CORPUS_DOCUMENT_ID,
    build_cross_document_tier,
    build_tree,
    hierarchical_cluster,
    membership_hash,
    raptor_retrieve,
    select_within_budget,
)
from sage.raptor.summarize import summarize_cluster
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder, FakeGenerator

warnings.filterwarnings("ignore")  # silence UMAP/numba runtime warnings in tests


def _blobs(n_per: int = 25, dim: int = 48, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((3, dim)) * 5
    return np.vstack([centers[i] + rng.standard_normal((n_per, dim)) for i in range(3)]).astype(
        "float32"
    )


def test_hierarchical_cluster_recovers_blobs():
    cfg = RaptorCfg(umap_target_dim=5, umap_n_epochs=200, max_clusters=10)
    clusters = hierarchical_cluster(_blobs(), cfg=cfg, seed=42)
    assert 2 <= len(clusters) <= 6
    covered = set().union(*clusters)
    assert len(covered) == 75  # every point assigned


def test_clustering_is_deterministic():
    cfg = RaptorCfg(umap_target_dim=5, umap_n_epochs=200, max_clusters=10)
    a = hierarchical_cluster(_blobs(), cfg=cfg, seed=42)
    b = hierarchical_cluster(_blobs(), cfg=cfg, seed=42)
    assert [sorted(c) for c in a] == [sorted(c) for c in b]


def test_membership_hash_is_order_independent():
    assert membership_hash(["a", "b", "c"]) == membership_hash(["c", "a", "b"])
    assert membership_hash(["a", "b"]) != membership_hash(["a", "c"])


def test_select_within_budget_respects_budget():
    from sage.core.types import SearchResult

    results = [
        SearchResult(
            chunk_id=f"c{i}", document_id="d", content="x" * 400, relevance_score=1.0 - i * 0.1
        )
        for i in range(10)
    ]
    kept = select_within_budget(results, token_budget=250)  # each ~100 tokens
    assert 1 <= len(kept) <= 3


async def test_summarize_cluster_uses_generator():
    gen = FakeGenerator(response="SUMMARY")
    out = await summarize_cluster(["a", "b"], level=1, generator=gen)
    assert out == "SUMMARY"


async def _leaves(store: LanceDBStore, embedder: FakeEmbedder, n: int = 24) -> list[StoreRow]:
    texts = [f"Document about topic {i % 3}: detail number {i}." for i in range(n)]
    embs = await embedder.embed_documents(texts)
    rows = [
        StoreRow(
            chunk_id=f"doc:{i}",
            document_id="doc",
            chunk_index=i,
            content=texts[i],
            embedding=embs[i],
            level=0,
        )
        for i in range(n)
    ]
    await store.upsert(rows)
    return rows


async def test_build_tree_and_retrieve(tmp_path):
    embedder = FakeEmbedder(dim=48)
    store = LanceDBStore(tmp_path / "db", dim=48)
    leaves = await _leaves(store, embedder)
    cfg = RaptorCfg(umap_target_dim=5, umap_n_epochs=100, max_clusters=8, max_levels=3)
    summaries = await build_tree(
        leaves,
        document_id="doc",
        embedder=embedder,
        generator=FakeGenerator(response="cluster summary"),
        store=store,
        cfg=cfg,
        seed=42,
    )
    assert summaries  # at least one summary level was built
    assert all(s.level >= 1 for s in summaries)
    assert all(s.child_ids for s in summaries)

    qvec = await embedder.embed_query("topic 1")
    collapsed = await raptor_retrieve(store, qvec, cfg=cfg, top_k=5)
    assert collapsed
    traversal_cfg = cfg.model_copy(update={"retrieval_mode": "tree_traversal"})
    traversal = await raptor_retrieve(store, qvec, cfg=traversal_cfg, top_k=5)
    assert traversal


async def test_cross_document_tier_caches_and_skips_single_doc(tmp_path):
    embedder = FakeEmbedder(dim=48)
    store = LanceDBStore(tmp_path / "db", dim=48)
    cfg = RaptorCfg(umap_target_dim=5, max_clusters=8, min_nodes_for_level=4)

    # Build per-document top summaries for several documents.
    texts = [f"Doc {d} summary covering theme {d % 2}." for d in range(8)]
    embs = await embedder.embed_documents(texts)
    tops = [
        StoreRow(
            chunk_id=f"d{d}:top",
            document_id=f"d{d}",
            chunk_index=0,
            content=texts[d],
            embedding=embs[d],
            level=1,
        )
        for d in range(8)
    ]
    nodes = await build_cross_document_tier(
        tops,
        embedder=embedder,
        generator=FakeGenerator(response="corpus summary"),
        store=store,
        cfg=cfg,
        seed=42,
    )
    assert nodes
    assert all(n.document_id == CORPUS_DOCUMENT_ID for n in nodes)

    # Single-document input produces no cross-document tier.
    empty = await build_cross_document_tier(
        tops[:1],
        embedder=embedder,
        generator=FakeGenerator(response="x"),
        store=store,
        cfg=cfg,
        seed=42,
    )
    assert empty == []


@pytest.mark.parametrize("covariance", ["full", "spherical"])
def test_covariance_ablation_runs(covariance):
    cfg = RaptorCfg(
        umap_target_dim=5, umap_n_epochs=150, max_clusters=10, cluster_covariance=covariance
    )
    clusters = hierarchical_cluster(_blobs(), cfg=cfg, seed=42)
    assert len(set().union(*clusters)) == 75
