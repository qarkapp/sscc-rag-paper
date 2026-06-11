"""RAPTOR tree construction.

Starting from leaf chunks (level 0), each level clusters the current nodes, summarizes
each cluster into a parent node at the next level, embeds the summaries, and stores
them with links to their children. Construction stops at ``max_levels`` or when a
level has too few nodes to cluster.
"""

from __future__ import annotations

import asyncio

import numpy as np

from sage.config.schema import RaptorCfg
from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.types import StoreRow
from sage.raptor.clustering import hierarchical_cluster
from sage.raptor.summarize import summarize_cluster

__all__ = ["build_tree"]


async def build_tree(
    leaves: list[StoreRow],
    *,
    document_id: str,
    embedder: Embedder,
    generator: Generator,
    store: VectorStore,
    cfg: RaptorCfg,
    seed: int,
    concurrency: int = 5,
    level_tag: str = "L",
) -> list[StoreRow]:
    """Build summary levels above ``leaves`` and persist them. Returns all summaries."""
    summaries: list[StoreRow] = []
    current = leaves
    semaphore = asyncio.Semaphore(concurrency)

    for level in range(1, cfg.max_levels + 1):
        if len(current) < cfg.min_nodes_for_level:
            break
        embeddings = np.vstack([r.embedding for r in current])
        clusters = hierarchical_cluster(embeddings, cfg=cfg, seed=seed)
        if len(clusters) <= 1:
            break  # nothing left to abstract

        async def _summarize(members: list[int], nodes: list[StoreRow], lvl: int) -> str:
            async with semaphore:
                return await summarize_cluster(
                    [nodes[i].content for i in members],
                    level=lvl,
                    generator=generator,
                )

        texts = await asyncio.gather(*(_summarize(m, current, level) for m in clusters))
        summary_embeddings = await embedder.embed_documents(texts)

        level_nodes: list[StoreRow] = []
        for idx, (members, text) in enumerate(zip(clusters, texts, strict=True)):
            level_nodes.append(
                StoreRow(
                    chunk_id=f"{document_id}:{level_tag}{level}:{idx}",
                    document_id=document_id,
                    chunk_index=idx,
                    content=text,
                    embedding=summary_embeddings[idx],
                    level=level,
                    child_ids=tuple(current[i].chunk_id for i in members),
                )
            )
        await store.upsert(level_nodes)
        summaries.extend(level_nodes)
        current = level_nodes

    return summaries
