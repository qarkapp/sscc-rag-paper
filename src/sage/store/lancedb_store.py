"""LanceDB-backed vector store.

Chunks and RAPTOR summary nodes live in a single table keyed by ``chunk_id`` and
tagged with a ``level`` (0 = leaf). Hierarchy links (``parent_id``, ``child_ids``)
are stored inline, so no separate metadata database is required. Search uses L2
distance, and relevance scores follow the reference convention
``score = 1 / (1 + distance)``.

LanceDB's synchronous client is used and offloaded to worker threads so the store
satisfies the async :class:`sage.core.protocols.VectorStore` interface without
blocking the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from sage.core.types import ScoreSource, SearchResult, StoreRow

__all__ = ["LanceDBStore"]


def _schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("document_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("content", pa.string()),
            pa.field("level", pa.int32()),
            pa.field("filename", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("section_name", pa.string()),
            pa.field("language", pa.string()),
            pa.field("parent_id", pa.string()),
            pa.field("child_ids", pa.list_(pa.string())),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]
    )


def _row_to_record(row: StoreRow) -> dict[str, Any]:
    return {
        "chunk_id": row.chunk_id,
        "document_id": row.document_id,
        "chunk_index": int(row.chunk_index),
        "content": row.content,
        "level": int(row.level),
        "filename": row.filename,
        "page_number": row.page_number,
        "section_name": row.section_name,
        "language": row.language,
        "parent_id": row.parent_id,
        "child_ids": list(row.child_ids),
        "vector": np.asarray(row.embedding, dtype=np.float32).tolist(),
    }


def _sql_quote(value: str) -> str:
    """Quote a string for a LanceDB SQL filter, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def _record_to_result(rec: dict[str, Any]) -> SearchResult:
    distance = float(rec.get("_distance", 0.0))
    return SearchResult(
        chunk_id=rec["chunk_id"],
        document_id=rec["document_id"],
        content=rec["content"],
        relevance_score=1.0 / (1.0 + distance),
        chunk_index=int(rec["chunk_index"]),
        level=int(rec["level"]),
        score_source=ScoreSource.BI_ENCODER,
        filename=rec.get("filename"),
        page_number=rec.get("page_number"),
        section_name=rec.get("section_name"),
        embedding=np.asarray(rec["vector"], dtype=np.float32),
    )


def _record_to_row(rec: dict[str, Any]) -> StoreRow:
    return StoreRow(
        chunk_id=rec["chunk_id"],
        document_id=rec["document_id"],
        chunk_index=int(rec["chunk_index"]),
        content=rec["content"],
        embedding=np.asarray(rec["vector"], dtype=np.float32),
        level=int(rec["level"]),
        filename=rec.get("filename"),
        page_number=rec.get("page_number"),
        section_name=rec.get("section_name"),
        language=rec.get("language"),
        parent_id=rec.get("parent_id"),
        child_ids=tuple(rec.get("child_ids") or ()),
    )


class LanceDBStore:
    """Implements :class:`sage.core.protocols.VectorStore`."""

    def __init__(self, path: str | Path, dim: int, *, table: str = "chunks") -> None:
        self._path = str(path)
        self._dim = dim
        self._table_name = table
        self._table: Any | None = None

    # -- table lifecycle ---------------------------------------------------

    def _ensure_table(self) -> Any:
        if self._table is None:
            import lancedb

            db = lancedb.connect(self._path)
            if self._table_name in db.list_tables():
                self._table = db.open_table(self._table_name)
            else:
                self._table = db.create_table(self._table_name, schema=_schema(self._dim))
        return self._table

    # -- VectorStore API ---------------------------------------------------

    async def upsert(self, rows: Sequence[StoreRow]) -> None:
        if not rows:
            return
        records = [_row_to_record(r) for r in rows]
        await asyncio.to_thread(self._upsert_sync, records)

    def _upsert_sync(self, records: list[dict[str, Any]]) -> None:
        table = self._ensure_table()
        (
            table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        *,
        document_filter: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        where = "level = 0"
        if document_filter:
            ids = ", ".join(_sql_quote(d) for d in document_filter)
            where += f" AND document_id IN ({ids})"
        return await asyncio.to_thread(self._search_sync, query_vector, top_k, where)

    async def search_by_level(
        self, query_vector: np.ndarray, top_k: int, level: int
    ) -> list[SearchResult]:
        return await asyncio.to_thread(
            self._search_sync, query_vector, top_k, f"level = {int(level)}"
        )

    def _search_sync(self, query_vector: np.ndarray, top_k: int, where: str) -> list[SearchResult]:
        table = self._ensure_table()
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        records = table.search(vec).metric("l2").where(where).limit(top_k).to_list()
        return [_record_to_result(r) for r in records]

    async def get_by_ids(self, ids: Sequence[str]) -> list[StoreRow]:
        if not ids:
            return []
        return await asyncio.to_thread(self._get_by_ids_sync, list(ids))

    def _get_by_ids_sync(self, ids: list[str]) -> list[StoreRow]:
        table = self._ensure_table()
        quoted = ", ".join(_sql_quote(i) for i in ids)
        records = table.search().where(f"chunk_id IN ({quoted})").limit(len(ids)).to_list()
        by_id = {r["chunk_id"]: _record_to_row(r) for r in records}
        return [by_id[i] for i in ids if i in by_id]

    async def count(self) -> int:
        return await asyncio.to_thread(lambda: self._ensure_table().count_rows())

    async def all_leaf_rows(self) -> list[StoreRow]:
        """Return every leaf row (level 0), used to build the chunk graph."""
        return await asyncio.to_thread(self._all_leaf_rows_sync)

    def _all_leaf_rows_sync(self) -> list[StoreRow]:
        table = self._ensure_table()
        records = table.search().where("level = 0").limit(table.count_rows()).to_list()
        rows = [_record_to_row(r) for r in records]
        rows.sort(key=lambda r: (r.document_id, r.chunk_index))
        return rows
