"""Tests for the typed chunk graph, PPR expansion, and GraphSAGE refinement."""

from __future__ import annotations

import warnings

import numpy as np

from sage.config.schema import GraphCfg
from sage.core.types import Chunk
from sage.graph import build_chunk_graph, expand_by_ppr, personalized_pagerank
from sage.graph.gnn import late_fusion_score, refine_embeddings

warnings.filterwarnings("ignore")


def _chunk(cid: str, doc: str, idx: int, content: str, language: str | None = None) -> Chunk:
    return Chunk(chunk_id=cid, document_id=doc, chunk_index=idx, content=content, language=language)


def test_sequential_and_semantic_edges():
    chunks = [
        _chunk("d:0", "d", 0, "alpha beta"),
        _chunk("d:1", "d", 1, "beta gamma"),
        _chunk("e:0", "e", 0, "unrelated text"),
    ]
    # d:0 and d:1 nearly identical embeddings -> semantic edge; e:0 orthogonal.
    embeddings = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32)
    graph = build_chunk_graph(chunks, embeddings, semantic_threshold=0.7)
    assert (0, 1) in graph.edges["sequential"]  # consecutive in doc d
    assert (0, 1) in graph.edges["semantic"]  # high cosine
    assert (0, 2) not in graph.edges["semantic"]  # orthogonal


def test_xref_edges_share_marker():
    chunks = [
        _chunk("d:0", "d", 0, "As shown in Section 3 the result holds."),
        _chunk("d:1", "d", 1, "Recall Section 3 again here."),
        _chunk("d:2", "d", 2, "Nothing relevant."),
    ]
    embeddings = np.eye(3, dtype=np.float32)[:, :2]
    graph = build_chunk_graph(chunks, embeddings)
    assert (0, 1) in graph.edges["xref"]
    assert (0, 2) not in graph.edges["xref"]


def test_ast_edges_share_identifier():
    chunks = [
        _chunk("c:0", "c", 0, "def compute_total(items): return sum(items)", language="python"),
        _chunk("c:1", "c", 1, "result = compute_total(values)", language="python"),
        _chunk("c:2", "c", 2, "x = 1", language="python"),
    ]
    embeddings = np.eye(3, dtype=np.float32)[:, :2]
    graph = build_chunk_graph(chunks, embeddings)
    assert (0, 1) in graph.edges["ast"]  # share compute_total


def test_ppr_expands_to_connected_chunks():
    chunks = [_chunk(f"d:{i}", "d", i, f"chunk {i}") for i in range(5)]
    embeddings = np.random.default_rng(0).standard_normal((5, 4)).astype("float32")
    graph = build_chunk_graph(chunks, embeddings, semantic_threshold=2.0)  # no semantic edges
    # sequential chain 0-1-2-3-4; seed at d:0 should rank d:1 highly.
    added = expand_by_ppr(graph, ["d:0"], ["sequential"], budget=2)
    assert "d:0" not in added
    assert "d:1" in added


def test_ppr_scores_sum_positive():
    chunks = [_chunk(f"d:{i}", "d", i, f"chunk {i}") for i in range(4)]
    embeddings = np.random.default_rng(1).standard_normal((4, 4)).astype("float32")
    graph = build_chunk_graph(chunks, embeddings, semantic_threshold=2.0)
    scores = personalized_pagerank(graph, ["d:0"], ["sequential"])
    assert scores["d:0"] > 0
    assert all(s >= 0 for s in scores.values())


def test_late_fusion_beta_zero_recovers_baseline():
    q = np.array([1.0, 0.0, 0.0])
    base = np.array([0.9, 0.1, 0.0])
    refined = np.array([0.0, 0.0, 1.0])
    baseline = late_fusion_score(q, base, base, beta=0.0)
    with_graph = late_fusion_score(q, base, refined, beta=0.0)
    assert baseline == with_graph  # beta=0 ignores the refined embedding


def test_graphsage_refine_shapes_and_identity():
    rng = np.random.default_rng(0)
    chunks = [_chunk(f"d:{i}", "d", i, f"chunk {i}") for i in range(20)]
    centers = rng.standard_normal((2, 8))
    embeddings = np.vstack(
        [centers[i % 2] + 0.1 * rng.standard_normal(8) for i in range(20)]
    ).astype("float32")
    graph = build_chunk_graph(chunks, embeddings, semantic_threshold=0.5)

    # L=0 is the identity (ablation baseline).
    identity = refine_embeddings(graph, GraphCfg(gnn_layers=0), ["semantic", "sequential"])
    np.testing.assert_array_equal(identity, embeddings)

    # L=2 refines into the same dimensionality.
    refined = refine_embeddings(
        graph, GraphCfg(gnn_layers=2, gnn_hidden=16), ["semantic", "sequential"], seed=42
    )
    assert refined.shape == embeddings.shape
    assert not np.allclose(refined, embeddings)  # refinement changed the embeddings


def test_graphsage_is_deterministic():
    rng = np.random.default_rng(2)
    chunks = [_chunk(f"d:{i}", "d", i, f"chunk {i}") for i in range(15)]
    embeddings = rng.standard_normal((15, 8)).astype("float32")
    graph = build_chunk_graph(chunks, embeddings, semantic_threshold=0.3)
    cfg = GraphCfg(gnn_layers=2, gnn_hidden=16)
    a = refine_embeddings(graph, cfg, ["semantic", "sequential"], seed=7)
    b = refine_embeddings(graph, cfg, ["semantic", "sequential"], seed=7)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
