"""Cross-document RAPTOR tier.

The original RAPTOR builds one tree per document. This tier adds a corpus-level
hierarchy on top: it re-clusters the per-document top-level summaries across the
whole collection and summarizes those clusters, so retrieval can surface themes that
span documents. Corpus nodes are stored under a sentinel document id.

Incremental caching: each corpus cluster is keyed by a hash of its member ids, so a
cluster whose membership is unchanged since the last build reuses its stored summary
instead of regenerating it. This makes re-indexing after adding a document cheap.
"""

from __future__ import annotations

import asyncio
import hashlib

import numpy as np

from sage.config.schema import RaptorCfg
from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.types import StoreRow
from sage.raptor.clustering import hierarchical_cluster
from sage.raptor.summarize import summarize_cluster

__all__ = ["CORPUS_DOCUMENT_ID", "build_cross_document_tier", "membership_hash"]

CORPUS_DOCUMENT_ID = "__corpus__"


def membership_hash(member_ids: list[str]) -> str:
    """Stable hash of a cluster's member ids (order-independent)."""
    digest = hashlib.sha256("\n".join(sorted(member_ids)).encode()).hexdigest()
    return digest[:16]


async def build_cross_document_tier(
    doc_top_summaries: list[StoreRow],
    *,
    embedder: Embedder,
    generator: Generator,
    store: VectorStore,
    cfg: RaptorCfg,
    seed: int,
    cached_summaries: dict[str, StoreRow] | None = None,
    concurrency: int = 5,
) -> list[StoreRow]:
    """Cluster per-document top summaries corpus-wide and summarize each cluster.

    ``cached_summaries`` maps ``membership_hash`` -> previously built corpus node;
    matching clusters reuse the cached summary instead of regenerating it.
    """
    distinct_docs = {r.document_id for r in doc_top_summaries}
    if len(distinct_docs) < 2 or len(doc_top_summaries) < cfg.min_nodes_for_level:
        return []  # nothing to bridge across

    cache = cached_summaries or {}
    embeddings = np.vstack([r.embedding for r in doc_top_summaries])
    clusters = hierarchical_cluster(embeddings, cfg=cfg, seed=seed)
    semaphore = asyncio.Semaphore(concurrency)

    async def _node(idx: int, members: list[int]) -> StoreRow:
        member_ids = [doc_top_summaries[i].chunk_id for i in members]
        h = membership_hash(member_ids)
        if h in cache:
            return cache[h]  # unchanged cluster -> reuse cached summary
        async with semaphore:
            text = await summarize_cluster(
                [doc_top_summaries[i].content for i in members],
                level=2,
                generator=generator,
            )
        embedding = (await embedder.embed_documents([text]))[0]
        return StoreRow(
            chunk_id=f"{CORPUS_DOCUMENT_ID}:{h}",
            document_id=CORPUS_DOCUMENT_ID,
            chunk_index=idx,
            content=text,
            embedding=embedding,
            level=1,
            child_ids=tuple(member_ids),
        )

    nodes = await asyncio.gather(*(_node(i, m) for i, m in enumerate(clusters)))
    corpus_nodes = list(nodes)
    await store.upsert(corpus_nodes)
    return corpus_nodes
