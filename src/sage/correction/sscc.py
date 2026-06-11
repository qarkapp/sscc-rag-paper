"""Score-source calibrated correction (SSCC).

CRAG applies one relevance threshold regardless of where a result's score came from.
But bi-encoder cosine scores and cross-encoder rerank logits live on different
scales, so one threshold systematically over- or under-filters one of them. SSCC
keeps a separate, dev-calibrated threshold per score source. Sources that are not
calibrated (RRF, PPR) are not threshold-filtered.
"""

from __future__ import annotations

from sage.config.schema import CorrectionCfg
from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.registry import register
from sage.core.types import Confidence, CorrectionOutcome, ScoreSource, SearchResult

__all__ = ["SsccCorrector"]


@register("corrector", "sscc")
class SsccCorrector:
    """Implements :class:`sage.core.protocols.Corrector`."""

    def __init__(self, cfg: CorrectionCfg) -> None:
        self._cfg = cfg

    def threshold_for(self, source: ScoreSource) -> float:
        if source is ScoreSource.BI_ENCODER:
            return self._cfg.sscc_tau_bi
        if source is ScoreSource.CROSS_ENCODER:
            return self._cfg.sscc_tau_cross
        return float("-inf")  # RRF/PPR scales are not calibrated -> do not filter

    async def correct(
        self,
        query: str,
        results: list[SearchResult],
        *,
        generator: Generator,
        embedder: Embedder,
        store: VectorStore,
    ) -> CorrectionOutcome:
        kept = [r for r in results if r.relevance_score >= self.threshold_for(r.score_source)]

        if kept:
            confidence = (
                Confidence.HIGH if len(kept) >= max(1, len(results)) // 2 else Confidence.MEDIUM
            )
            return CorrectionOutcome(confidence, kept)

        if not self._cfg.enable_query_rewrite:
            return CorrectionOutcome(Confidence.LOW, results)

        rewritten = (
            await generator.complete(
                "You rewrite search queries for better retrieval. Return ONLY the query.",
                f"Original query: {query}",
            )
        ).strip()
        if not rewritten:
            return CorrectionOutcome(Confidence.NONE if not results else Confidence.LOW, results)
        retried = await store.search(await embedder.embed_query(rewritten), max(len(results), 5))
        confidence = Confidence.MEDIUM if retried else Confidence.NONE
        return CorrectionOutcome(confidence, retried or results, rewritten_query=rewritten)
