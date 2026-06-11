"""End-to-end indexing pipeline test (offline via fakes)."""

from __future__ import annotations

import warnings

from sage.chunking import ChonkieChunker
from sage.config.schema import PipelineConfig
from sage.parsing import TextParser
from sage.pipeline import Document, IndexingPipeline
from sage.raptor.cross_doc import CORPUS_DOCUMENT_ID
from sage.store import LanceDBStore
from sage.testing import FakeEmbedder, FakeGenerator

warnings.filterwarnings("ignore")


def _make_pipeline(tmp_path, cfg: PipelineConfig):
    embedder = FakeEmbedder(dim=48)
    store = LanceDBStore(tmp_path / "db", dim=48)
    pipeline = IndexingPipeline(
        embedder=embedder,
        generator=FakeGenerator(response="summary"),
        store=store,
        chunker=ChonkieChunker(cfg.chunking),
        parser=TextParser(),
        config=cfg,
    )
    return pipeline, store, embedder


def _doc(doc_id: str, topic: str, n: int = 30) -> Document:
    body = " ".join(f"{topic} sentence number {i} with specific detail {i}." for i in range(n))
    return Document(document_id=doc_id, filename=f"{doc_id}.md", data=body.encode())


async def test_index_single_document(tmp_path):
    cfg = PipelineConfig()
    cfg.chunking.prose_target_chars = 200
    cfg.raptor.umap_target_dim = 5
    cfg.raptor.umap_n_epochs = 100
    cfg.raptor.max_clusters = 6
    pipeline, store, embedder = _make_pipeline(tmp_path, cfg)

    await pipeline.index_document(_doc("d1", "retrieval"))
    assert await store.count() > 0
    results = await store.search(await embedder.embed_query("retrieval detail"), top_k=5)
    assert results
    assert all(r.level == 0 for r in results)  # leaf search


async def test_index_corpus_builds_cross_document_tier(tmp_path):
    cfg = PipelineConfig()
    cfg.chunking.prose_target_chars = 200
    cfg.raptor.umap_target_dim = 5
    cfg.raptor.umap_n_epochs = 100
    cfg.raptor.max_clusters = 6
    cfg.raptor.min_nodes_for_level = 4
    pipeline, store, embedder = _make_pipeline(tmp_path, cfg)

    docs = [_doc(f"d{i}", topic) for i, topic in enumerate(["alpha", "beta", "gamma", "delta"])]
    await pipeline.index_corpus(docs)

    qvec = await embedder.embed_query("alpha")
    # Fetch broadly: corpus nodes coexist with per-document summaries at level 1.
    corpus_hits = await store.search_by_level(qvec, top_k=100, level=1)
    corpus_nodes = [r for r in corpus_hits if r.document_id == CORPUS_DOCUMENT_ID]
    assert corpus_nodes  # a cross-document tier was created


async def test_indexing_without_raptor(tmp_path):
    cfg = PipelineConfig()
    cfg.chunking.prose_target_chars = 200
    cfg.raptor.enabled = False
    pipeline, store, embedder = _make_pipeline(tmp_path, cfg)

    await pipeline.index_document(_doc("d1", "retrieval"))
    # Only leaves exist (no summary levels).
    summaries = await store.search_by_level(await embedder.embed_query("x"), top_k=10, level=1)
    assert summaries == []
