"""Retrieval pipeline: route -> retrieve -> fuse -> rerank -> correct -> expand.

The pipeline is assembled from configuration; each stage is guarded by its
component's ``enabled`` flag, and every stage records a structured entry in the
:class:`~sage.core.types.RetrievalTrace` for analysis and figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sage.config.schema import PipelineConfig
from sage.core.protocols import (
    Corrector,
    Embedder,
    Generator,
    Reranker,
    RetrievalStrategy,
    Router,
    VectorStore,
)
from sage.core.types import (
    RetrievalTrace,
    ScoreSource,
    SearchResult,
    Strategy,
    StrategyDecision,
)
from sage.graph.index import GraphContext
from sage.graph.nli_edges import NliClassifier
from sage.pipeline.expansion import expand_parent_chunks
from sage.pipeline.rerank import apply_reranker
from sage.prefetch.srp import SpeculativeRetrievalPrefetcher
from sage.raptor.retrieval import raptor_retrieve
from sage.strategies.fusion import merge_deduplicate

__all__ = ["RetrievalPipeline"]


@dataclass(slots=True)
class RetrievalPipeline:
    config: PipelineConfig
    embedder: Embedder
    store: VectorStore
    router: Router
    strategies: dict[Strategy, RetrievalStrategy]
    reranker: Reranker | None = None
    corrector: Corrector | None = None
    generator: Generator | None = None
    graph: GraphContext | None = None
    prefetcher: SpeculativeRetrievalPrefetcher | None = None
    nli_classifier: NliClassifier | None = None
    _graph_built: bool = False

    async def run(
        self, query: str, top_k: int | None = None
    ) -> tuple[list[SearchResult], RetrievalTrace]:
        cfg = self.config
        k = top_k or cfg.top_k
        trace = RetrievalTrace(query=query)

        query_vector = await self.embedder.embed_query(query)

        # Speculative prefetch: serve from the buffer on a near-duplicate query.
        if cfg.prefetch.enabled and self.prefetcher is not None:
            hit = await self.prefetcher.maybe_hit(query_vector)
            if hit is not None:
                trace.record("prefetch_hit", n=len(hit))
                return hit[:k], trace

        decision = await self._route(query, query_vector)
        trace.record("route", strategy=str(decision.strategy), entropy=decision.entropy)

        fetch = k * cfg.rerank.over_fetch if cfg.rerank.enabled else k
        strategy = self.strategies[_strategy_key(decision.strategy)]
        candidates = await strategy.retrieve(
            query,
            query_vector,
            store=self.store,
            embedder=self.embedder,
            generator=self.generator,
            top_k=fetch,
        )
        if cfg.raptor.enabled:
            raptor_hits = await raptor_retrieve(
                self.store, query_vector, cfg=cfg.raptor, top_k=fetch
            )
            candidates = merge_deduplicate([candidates, raptor_hits])
        trace.record("retrieve", n=len(candidates))

        if cfg.graph.enabled:
            candidates = await self._graph_stage(
                query_vector, candidates, k, decision.is_multi_hop, trace
            )

        if cfg.rerank.enabled and self.reranker is not None:
            candidates = await apply_reranker(self.reranker, query, candidates, k)
            trace.record("rerank", n=len(candidates))
        else:
            candidates = candidates[:k]

        if cfg.correction.enabled and self.corrector is not None and self.generator is not None:
            outcome = await self.corrector.correct(
                query,
                candidates,
                generator=self.generator,
                embedder=self.embedder,
                store=self.store,
            )
            candidates = outcome.results
            trace.record(
                "correct",
                confidence=str(outcome.confidence),
                rewritten=outcome.rewritten_query,
            )

        if cfg.expansion.enabled:
            candidates = await expand_parent_chunks(
                self.store, candidates, cfg.expansion.window_chars
            )
        return candidates[:k], trace

    async def _graph_stage(
        self,
        query_vector: np.ndarray,
        candidates: list[SearchResult],
        k: int,
        is_multi_hop: bool,
        trace: RetrievalTrace,
    ) -> list[SearchResult]:
        if not self._graph_built:
            self.graph = await GraphContext.build(
                self.store,
                self.config.graph,
                seed=self.config.seed,
                nli_classifier=self.nli_classifier,
            )
            self._graph_built = True
        if self.graph is None or not candidates:
            return candidates

        # Personalized PageRank expansion: add structurally-related chunks.
        seeds = [r.chunk_id for r in candidates]
        added_ids = self.graph.expand(seeds, budget=max(1, k))
        # Entailment-chain expansion for multi-hop queries.
        if is_multi_hop:
            added_ids = list(dict.fromkeys([*added_ids, *self.graph.entailment_expand(seeds)]))
        if added_ids:
            rows = await self.store.get_by_ids(added_ids)
            extra = [
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    content=row.content,
                    relevance_score=0.0,
                    chunk_index=row.chunk_index,
                    level=row.level,
                    score_source=ScoreSource.PPR,
                    filename=row.filename,
                    embedding=row.embedding,
                )
                for row in rows
            ]
            candidates = merge_deduplicate([candidates, extra])
            trace.record("graph_expand", added=len(extra))

        # GraphSAGE refinement: blend bi-encoder and refined-embedding similarity.
        rescored = self.graph.rescore(query_vector, candidates)
        if rescored is not candidates:
            trace.record("graph_rescore", n=len(rescored))
        return rescored

    async def _route(self, query: str, query_vector: np.ndarray) -> StrategyDecision:
        if not self.config.router.enabled:
            return StrategyDecision(strategy=Strategy.SEMANTIC, rationale="router disabled")
        return await self.router.route(query, query_vector, self.store)


def _strategy_key(strategy: Strategy) -> Strategy:
    # HyDE and DPHF share one dual-path strategy; fusion behaviour is set by config.
    return Strategy.HYDE if strategy is Strategy.DPHF else strategy
