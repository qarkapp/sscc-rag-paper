"""Test that graph PPR expansion is wired into the retrieval pipeline."""

from __future__ import annotations

import warnings

from sage.config import semantic_only
from sage.config.schema import PipelineConfig
from sage.core.types import StoreRow
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder

warnings.filterwarnings("ignore")


async def _index(tmp_path, embedder):
    store = LanceDBStore(tmp_path / "db", dim=24)
    texts = [f"doc A paragraph {i} discussing subtopic {i}" for i in range(8)]
    embs = await embedder.embed_documents(texts)
    await store.upsert(
        [
            StoreRow(
                chunk_id=f"a:{i}",
                document_id="a",
                chunk_index=i,
                content=texts[i],
                embedding=embs[i],
                level=0,
            )
            for i in range(8)
        ]
    )
    return store


async def test_graph_expansion_runs_and_adds_chunks(tmp_path):
    embedder = FakeEmbedder(dim=24)
    store = await _index(tmp_path, embedder)

    cfg = semantic_only(PipelineConfig())
    cfg.graph.enabled = True
    cfg.graph.edges = ["sequential", "semantic"]  # structural edges over a single doc
    cfg.expansion.enabled = False

    pipeline = build_retrieval_pipeline(cfg, embedder=embedder, store=store)
    _, trace = await pipeline.run("subtopic 2", top_k=3)

    expand = trace.last("graph_expand")
    assert expand is not None  # the graph stage ran
    assert expand["added"] >= 1  # PPR surfaced structurally-related (non-seed) chunks
    # the graph is built once and reused across queries
    assert pipeline.graph is not None


async def test_graph_disabled_has_no_expand_stage(tmp_path):
    embedder = FakeEmbedder(dim=24)
    store = await _index(tmp_path, embedder)
    cfg = semantic_only(PipelineConfig())  # graph disabled by default
    pipeline = build_retrieval_pipeline(cfg, embedder=embedder, store=store)
    _, trace = await pipeline.run("subtopic 2", top_k=3)
    assert trace.last("graph_expand") is None


async def test_srp_short_circuits_on_buffer_hit(tmp_path):
    from sage.config.schema import PrefetchCfg
    from sage.core.types import SearchResult
    from sage.prefetch import SpeculativeRetrievalPrefetcher

    embedder = FakeEmbedder(dim=24)
    store = await _index(tmp_path, embedder)

    async def _retrieve(_query: str) -> list[SearchResult]:
        return []

    prefetcher = SpeculativeRetrievalPrefetcher(
        PrefetchCfg(hit_cosine_threshold=0.99), embedder, _retrieve
    )
    qvec = await embedder.embed_query("subtopic 2")
    cached = [SearchResult(chunk_id="cached", document_id="d", content="x", relevance_score=1.0)]
    prefetcher._buffer.put(qvec, cached)

    cfg = semantic_only(PipelineConfig())
    cfg.prefetch.enabled = True
    pipeline = build_retrieval_pipeline(cfg, embedder=embedder, store=store)
    pipeline.prefetcher = prefetcher

    results, trace = await pipeline.run("subtopic 2", top_k=3)
    assert trace.last("prefetch_hit") is not None
    assert results[0].chunk_id == "cached"  # served from the buffer, not retrieval
