"""RAPTOR retrieval over the constructed tree.

Two modes mirror the reference system:

* ``collapsed`` -- flatten all tree levels, rank by similarity, and greedily fill a
  token budget. Simple and robust; the common default.
* ``tree_traversal`` -- start at the top level, select the best nodes, descend into
  their children, and repeat down to the leaves, then fill the token budget.

A retrieved chunk's token estimate follows the ``len // 4`` convention.
"""

from __future__ import annotations

import numpy as np

from sage.config.schema import RaptorCfg
from sage.core.protocols import VectorStore
from sage.core.types import SearchResult, StoreRow

__all__ = ["raptor_retrieve", "select_within_budget"]


def select_within_budget(results: list[SearchResult], token_budget: int) -> list[SearchResult]:
    """Greedily keep the highest-scoring results that fit the token budget."""
    selected: list[SearchResult] = []
    used = 0
    for result in results:
        cost = max(1, len(result.content) // 4)
        if selected and used + cost > token_budget:
            break
        selected.append(result)
        used += cost
    return selected


def _dedup(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    out: list[SearchResult] = []
    for r in sorted(results, key=lambda x: x.relevance_score, reverse=True):
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            out.append(r)
    return out


async def _max_level(store: VectorStore, query_vector: np.ndarray, cfg: RaptorCfg) -> int:
    for level in range(cfg.max_levels, 0, -1):
        hits = await store.search_by_level(query_vector, 1, level)
        if hits:
            return level
    return 0


async def raptor_retrieve(
    store: VectorStore,
    query_vector: np.ndarray,
    *,
    cfg: RaptorCfg,
    top_k: int,
) -> list[SearchResult]:
    """Retrieve from the RAPTOR tree using the configured mode."""
    if cfg.retrieval_mode == "collapsed":
        return await _collapsed(store, query_vector, cfg=cfg, top_k=top_k)
    return await _traversal(store, query_vector, cfg=cfg, top_k=top_k)


async def _collapsed(
    store: VectorStore, query_vector: np.ndarray, *, cfg: RaptorCfg, top_k: int
) -> list[SearchResult]:
    candidates: list[SearchResult] = []
    fetch = max(top_k * 3, cfg.traversal_top_k)
    for level in range(cfg.max_levels + 1):
        candidates.extend(await store.search_by_level(query_vector, fetch, level))
    ranked = _dedup(candidates)
    return select_within_budget(ranked, cfg.retrieval_token_budget)


async def _traversal(
    store: VectorStore, query_vector: np.ndarray, *, cfg: RaptorCfg, top_k: int
) -> list[SearchResult]:
    start = await _max_level(store, query_vector, cfg)
    if start == 0:
        leaves = await store.search_by_level(query_vector, top_k * 3, 0)
        return select_within_budget(_dedup(leaves), cfg.retrieval_token_budget)

    collected: list[SearchResult] = []
    frontier = await store.search_by_level(query_vector, cfg.traversal_top_k, start)
    for level in range(start, -1, -1):
        collected.extend(frontier)
        if level == 0:
            break
        # SearchResult does not carry child_ids; refetch the rows that do.
        frontier_rows = await store.get_by_ids([r.chunk_id for r in frontier])
        child_ids = [cid for row in frontier_rows for cid in row.child_ids]
        if not child_ids:
            # No stored children; fall back to a flat search at the next level down.
            frontier = await store.search_by_level(query_vector, cfg.traversal_top_k, level - 1)
            continue
        children = await store.get_by_ids(child_ids)
        frontier = _rank_children(children, query_vector, cfg.traversal_top_k)

    return select_within_budget(_dedup(collected), cfg.retrieval_token_budget)


def _rank_children(
    children: list[StoreRow], query_vector: np.ndarray, top_k: int
) -> list[SearchResult]:
    q = np.asarray(query_vector, dtype=np.float32)
    qn = q / (np.linalg.norm(q) + 1e-12)
    scored: list[SearchResult] = []
    for row in children:
        v = np.asarray(row.embedding, dtype=np.float32)
        cos = float(v @ qn / (np.linalg.norm(v) + 1e-12))
        scored.append(
            SearchResult(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                content=row.content,
                relevance_score=(cos + 1.0) / 2.0,
                chunk_index=row.chunk_index,
                level=row.level,
                filename=row.filename,
                page_number=row.page_number,
                section_name=row.section_name,
                embedding=row.embedding,
            )
        )
    scored.sort(key=lambda r: r.relevance_score, reverse=True)
    return scored[:top_k]
