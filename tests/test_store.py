"""Tests for the LanceDB vector store."""

from __future__ import annotations

import numpy as np

from sage.core.types import StoreRow
from sage.store import LanceDBStore


def _row(cid: str, vec: list[float], level: int = 0, doc: str = "d1") -> StoreRow:
    return StoreRow(
        chunk_id=cid,
        document_id=doc,
        chunk_index=int(cid.split(":")[-1]) if ":" in cid else 0,
        content=f"content of {cid}",
        embedding=np.asarray(vec, dtype=np.float32),
        level=level,
        child_ids=("a", "b") if level > 0 else (),
    )


async def test_upsert_search_and_scoring(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=3)
    await store.upsert(
        [
            _row("c0", [1.0, 0.0, 0.0]),
            _row("c1", [0.0, 1.0, 0.0]),
            _row("c2", [0.9, 0.1, 0.0]),
        ]
    )
    results = await store.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)
    assert results[0].chunk_id == "c0"  # exact match ranks first
    # scores are 1/(1+L2) so descending and in (0, 1]
    scores = [r.relevance_score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert 0.0 < scores[0] <= 1.0


async def test_level_filter(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=3)
    await store.upsert([_row("c0", [1.0, 0.0, 0.0], level=0)])
    await store.upsert([_row("s0", [1.0, 0.0, 0.0], level=1)])
    leaves = await store.search(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=5)
    assert {r.chunk_id for r in leaves} == {"c0"}
    summaries = await store.search_by_level(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), top_k=5, level=1
    )
    assert {r.chunk_id for r in summaries} == {"s0"}


async def test_get_by_ids_preserves_order_and_embeddings(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=3)
    await store.upsert([_row("c0", [1.0, 0.0, 0.0]), _row("c1", [0.0, 1.0, 0.0])])
    rows = await store.get_by_ids(["c1", "c0"])
    assert [r.chunk_id for r in rows] == ["c1", "c0"]
    assert rows[0].embedding.shape == (3,)


async def test_upsert_is_idempotent(tmp_path):
    store = LanceDBStore(tmp_path / "db", dim=3)
    await store.upsert([_row("c0", [1.0, 0.0, 0.0])])
    await store.upsert([_row("c0", [0.5, 0.5, 0.0])])  # same id -> update, not duplicate
    assert await store.count() == 1
