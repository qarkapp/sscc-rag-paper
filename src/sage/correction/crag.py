"""Corrective retrieval (CRAG): a single-threshold LLM relevance judge.

An LLM scores the relevance of the top results on a 1-100 scale. Scores at or above
the upper threshold yield high confidence; between the thresholds, medium; below the
lower threshold, the query is rewritten and retrieval is retried. This is the
baseline that score-source-calibrated correction (SSCC) improves on.
"""

from __future__ import annotations

import re

from sage.config.schema import CorrectionCfg
from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.registry import register
from sage.core.types import Confidence, CorrectionOutcome, SearchResult

__all__ = ["CragCorrector"]

_JUDGE_SYSTEM = "You evaluate search-result relevance. Return only a number from 1 to 100."
_REWRITE_SYSTEM = (
    "You rewrite search queries for better retrieval. Return ONLY the rewritten query."
)
_SNIPPET_CHARS = 500


@register("corrector", "crag")
class CragCorrector:
    """Implements :class:`sage.core.protocols.Corrector`."""

    def __init__(self, cfg: CorrectionCfg) -> None:
        self._cfg = cfg

    async def correct(
        self,
        query: str,
        results: list[SearchResult],
        *,
        generator: Generator,
        embedder: Embedder,
        store: VectorStore,
    ) -> CorrectionOutcome:
        if not results:
            return await self._rewrite(query, results, generator, embedder, store)

        score = await self._judge(query, results, generator)
        if score >= self._cfg.crag_upper:
            return CorrectionOutcome(Confidence.HIGH, results, raw_score=score)
        if score >= self._cfg.crag_lower:
            return CorrectionOutcome(Confidence.MEDIUM, results, raw_score=score)
        if self._cfg.enable_query_rewrite:
            return await self._rewrite(query, results, generator, embedder, store, score)
        return CorrectionOutcome(Confidence.LOW, results, raw_score=score)

    # -- internals ---------------------------------------------------------

    async def _judge(self, query: str, results: list[SearchResult], generator: Generator) -> float:
        snippets = "\n".join(f"- {r.content[:_SNIPPET_CHARS]}" for r in results[:3])
        prompt = (
            "Rate how relevant these search results are to the query, from 1 "
            "(irrelevant) to 100 (perfectly relevant). Return only the number.\n\n"
            f"Query: {query}\n{snippets}"
        )
        raw = await generator.complete(_JUDGE_SYSTEM, prompt, max_tokens=8)
        return _parse_score(raw)

    async def _rewrite(
        self,
        query: str,
        results: list[SearchResult],
        generator: Generator,
        embedder: Embedder,
        store: VectorStore,
        raw_score: float | None = None,
    ) -> CorrectionOutcome:
        rewritten = (await generator.complete(_REWRITE_SYSTEM, f"Original query: {query}")).strip()
        if not rewritten:
            return CorrectionOutcome(Confidence.LOW, results, raw_score=raw_score)
        new_vector = await embedder.embed_query(rewritten)
        retried = await store.search(new_vector, max(len(results), 5))
        if retried:
            return CorrectionOutcome(
                Confidence.MEDIUM, retried, rewritten_query=rewritten, raw_score=raw_score
            )
        confidence = Confidence.NONE if not results else Confidence.LOW
        return CorrectionOutcome(
            confidence, results, rewritten_query=rewritten, raw_score=raw_score
        )


def _parse_score(text: str) -> float:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return 100.0  # if the judge is unparseable, assume relevant (reference behaviour)
    return max(1.0, min(100.0, float(match.group())))
