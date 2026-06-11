"""Query-time graph context: build the chunk graph, refine, and expand.

Built lazily from the vector store's leaf rows the first time a graph-enabled
pipeline runs. Holds the typed chunk graph and (when GraphSAGE is enabled)
graph-refined embeddings, and exposes two operations on a retrieved result set:

* **expand** -- add structurally-related chunks via Personalized PageRank.
* **rescore** -- blend the bi-encoder score with a graph-refined cosine score
  (late fusion). With ``gnn_layers = 0`` refinement is the identity, so rescoring
  reduces exactly to the bi-encoder baseline.
"""

from __future__ import annotations

import asyncio

import numpy as np

from sage.config.schema import GraphCfg
from sage.core.protocols import VectorStore
from sage.core.types import Chunk, SearchResult
from sage.graph.build import EDGE_TYPES, ChunkGraph, build_chunk_graph
from sage.graph.gnn import late_fusion_score, refine_embeddings
from sage.graph.nli_edges import EntailmentEdge, NliClassifier, build_entailment_edges
from sage.graph.ppr import expand_by_ppr
from sage.graph.traversal import entailment_chains

__all__ = ["GraphContext", "get_or_build_graph"]

# Cache built graphs (incl. trained GraphSAGE) so ablations sharing a store + graph
# config reuse one build instead of re-training the GNN for every configuration.
_GRAPH_CACHE: dict[str, GraphContext | None] = {}
_GRAPH_LOCK = asyncio.Lock()


async def get_or_build_graph(
    store: VectorStore,
    cfg: GraphCfg,
    *,
    seed: int = 42,
    nli_classifier: NliClassifier | None = None,
) -> GraphContext | None:
    """Build the graph for (store, cfg) once and reuse it across pipelines."""
    key = f"{id(store)}:{cfg.model_dump_json()}"
    async with _GRAPH_LOCK:
        if key not in _GRAPH_CACHE:
            _GRAPH_CACHE[key] = await GraphContext.build(
                store, cfg, seed=seed, nli_classifier=nli_classifier
            )
        return _GRAPH_CACHE[key]


class GraphContext:
    """A built chunk graph with PPR expansion, GraphSAGE rescoring, and NLI chains."""

    def __init__(
        self,
        graph: ChunkGraph,
        cfg: GraphCfg,
        refined: dict[str, np.ndarray] | None = None,
        entailment: list[EntailmentEdge] | None = None,
    ) -> None:
        self._graph = graph
        self._cfg = cfg
        self._edge_types = [e for e in cfg.edges if e in EDGE_TYPES]
        self._base = {cid: graph.embeddings[i] for i, cid in enumerate(graph.chunk_ids)}
        self._refined = refined or {}
        self._entailment = entailment or []

    @classmethod
    async def build(
        cls,
        store: VectorStore,
        cfg: GraphCfg,
        *,
        seed: int = 42,
        nli_classifier: NliClassifier | None = None,
    ) -> GraphContext | None:
        """Construct the graph, refined embeddings, and (optional) entailment edges."""
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

        refined: dict[str, np.ndarray] | None = None
        edge_types = [e for e in cfg.edges if e in EDGE_TYPES]
        if cfg.gnn_layers > 0 and edge_types:
            refined_matrix = refine_embeddings(graph, cfg, edge_types, seed=seed)
            refined = {cid: refined_matrix[i] for i, cid in enumerate(graph.chunk_ids)}

        entailment: list[EntailmentEdge] | None = None
        if "nli" in cfg.edges and nli_classifier is not None:
            entailment = build_entailment_edges(
                chunks, embeddings, nli_classifier, cos_gate=cfg.nli_cos_gate
            )
        return cls(graph, cfg, refined, entailment)

    def entailment_expand(self, seed_ids: list[str], *, max_hops: int = 3) -> list[str]:
        """Return chunk ids reached along entailment chains from the seeds."""
        if not self._entailment:
            return []
        seeds = set(seed_ids)
        chains = entailment_chains(self._entailment, seed_ids, max_hops=max_hops)
        ordered: list[str] = []
        for path, _score in chains:
            for cid in path:
                if cid not in seeds and cid not in ordered:
                    ordered.append(cid)
        return ordered

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
            min_score=self._cfg.ppr_min_score,
        )

    def rescore(self, query_vector: np.ndarray, results: list[SearchResult]) -> list[SearchResult]:
        """Re-rank results by blending bi-encoder and graph-refined similarity."""
        if not self._refined:
            return results
        rescored: list[SearchResult] = []
        for r in results:
            refined = self._refined.get(r.chunk_id)
            base = self._base.get(r.chunk_id)
            if refined is None or base is None:
                rescored.append(r)
                continue
            if self._cfg.query_fusion == "late_fusion":
                score = late_fusion_score(query_vector, base, refined, self._cfg.late_fusion_beta)
            else:  # project_query / query_proj_head: score directly in the refined space
                score = _cos(query_vector, refined)
            rescored.append(r.with_score(score, r.score_source))
        rescored.sort(key=lambda x: x.relevance_score, reverse=True)
        return rescored


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)
