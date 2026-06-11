"""Query-time graph context: build the chunk graph and expand results via PPR.

Built lazily from the vector store's leaf rows the first time a graph-enabled
pipeline runs. Holds the typed chunk graph (and, optionally, GraphSAGE-refined
embeddings) and exposes Personalized PageRank expansion of a retrieved result set.
"""

from __future__ import annotations

import numpy as np

from sage.config.schema import GraphCfg
from sage.core.protocols import VectorStore
from sage.core.types import Chunk
from sage.graph.build import EDGE_TYPES, ChunkGraph, build_chunk_graph
from sage.graph.ppr import expand_by_ppr

__all__ = ["GraphContext"]


class GraphContext:
    """A built chunk graph plus PPR-based result expansion."""

    def __init__(self, graph: ChunkGraph, cfg: GraphCfg) -> None:
        self._graph = graph
        self._cfg = cfg
        # Structural edge types that are both configured and present in the graph.
        self._edge_types = [e for e in cfg.edges if e in EDGE_TYPES]

    @classmethod
    async def build(
        cls, store: VectorStore, cfg: GraphCfg, *, seed: int = 42
    ) -> GraphContext | None:
        """Construct the graph from the store's leaf rows (None if too few)."""
        rows = await store.all_leaf_rows()
        if len(rows) < 3:
            return None
        chunks = [
            Chunk(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                content=r.content,
                language=r.language,
            )
            for r in rows
        ]
        embeddings = np.vstack([r.embedding for r in rows])
        graph = build_chunk_graph(chunks, embeddings, semantic_threshold=cfg.semantic_threshold)
        return cls(graph, cfg)

    def expand(self, seed_ids: list[str], *, budget: int) -> list[str]:
        """Return up to ``budget`` structurally-related chunk ids via PPR."""
        if budget <= 0 or not self._edge_types:
            return []
        return expand_by_ppr(
            self._graph,
            seed_ids,
            self._edge_types,
            budget=budget,
            alpha=self._cfg.ppr_alpha,
            steps=self._cfg.ppr_steps,
        )
