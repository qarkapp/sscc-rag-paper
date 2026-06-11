"""Tests for the deterministic offline fakes."""

from __future__ import annotations

import numpy as np

from sage.testing import FakeEmbedder, FakeGenerator, FakeReranker


async def test_embedder_is_deterministic_and_normalized():
    emb = FakeEmbedder(dim=32)
    a = await emb.embed_query("hello")
    b = await emb.embed_query("hello")
    c = await emb.embed_query("different")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (32,)
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, rtol=1e-5)


async def test_embed_documents_shape():
    emb = FakeEmbedder(dim=16)
    mat = await emb.embed_documents(["a", "b", "c"])
    assert mat.shape == (3, 16)
    empty = await emb.embed_documents([])
    assert empty.shape == (0, 16)


async def test_reranker_orders_by_overlap():
    rr = FakeReranker()
    ranked = await rr.rerank("quick brown fox", ["a brown fox", "nothing here"], top_n=2)
    assert ranked[0][0] == 0
    assert ranked[0][1] >= ranked[1][1]


async def test_generator_stream_concatenates_to_complete():
    gen = FakeGenerator()
    streamed = "".join([tok async for tok in gen.stream("sys", "hello world")]).strip()
    full = await gen.complete("sys", "hello world")
    assert streamed == full
