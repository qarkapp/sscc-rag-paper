"""Answer-side evaluation: synthesise an answer from retrieved context and score it.

The retrieval harness (:mod:`sage.eval.ablate`) measures *retrieval* quality only.
Several contributions -- HyDE/DPHF, CRAG/SSCC correction, query rewriting -- act on
the answer rather than the ranking, so their effect is invisible to nDCG/Recall. This
module closes that gap: it runs the full pipeline, feeds the top-k passages to a
generator, and scores the generated answer against the gold answer(s) with EM and
token-F1. Generation calls go through the same cache as everything else, so repeated
ablations over a frozen retrieval set re-run at ~zero cost.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from sage.core.protocols import Generator
from sage.core.types import SearchResult
from sage.eval.dataset import QAExample, RetrievalDataset
from sage.eval.metrics import exact_match, token_f1
from sage.pipeline.retrieval import RetrievalPipeline

__all__ = ["AnswerScore", "answer_eval", "synthesize_answer"]

_SYSTEM = (
    "You answer questions strictly from the provided context. Give the shortest exact "
    "answer -- a name, entity, number, or short phrase. Do not explain. If the context "
    "does not contain the answer, reply with the single word: unknown."
)


def _prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"


async def synthesize_answer(
    question: str,
    results: list[SearchResult],
    generator: Generator,
    *,
    top_k: int = 5,
    max_context_chars: int = 1600,
) -> str:
    """Generate a short extractive answer from the top-k retrieved passages."""
    contexts = [r.content[:max_context_chars] for r in results[:top_k]]
    if not contexts:
        return ""
    return await generator.complete(_SYSTEM, _prompt(question, contexts), max_tokens=64)


@dataclass(slots=True)
class AnswerScore:
    """Per-query answer scores keyed by qid."""

    em: dict[str, float]
    f1: dict[str, float]

    @property
    def mean_em(self) -> float:
        return sum(self.em.values()) / len(self.em) if self.em else 0.0

    @property
    def mean_f1(self) -> float:
        return sum(self.f1.values()) / len(self.f1) if self.f1 else 0.0


def _best(example: QAExample, prediction: str) -> tuple[float, float]:
    golds = example.answers or ("",)
    em = max(exact_match(prediction, g) for g in golds)
    f1 = max(token_f1(prediction, g) for g in golds)
    return em, f1


async def answer_eval(
    pipeline: RetrievalPipeline,
    dataset: RetrievalDataset,
    generator: Generator,
    *,
    top_k: int = 5,
    concurrency: int = 4,
    query_timeout: float = 90.0,
) -> AnswerScore:
    """Run retrieval + answer synthesis over the dataset; return per-query EM/F1.

    A query that errors or times out scores zero rather than aborting the sweep, so a
    single slow generator call cannot sink an entire ablation row.
    """
    semaphore = asyncio.Semaphore(concurrency)
    em: dict[str, float] = {}
    f1: dict[str, float] = {}

    async def one(example: QAExample) -> None:
        async with semaphore:
            try:
                results, _ = await asyncio.wait_for(
                    pipeline.run(example.question, top_k=top_k), timeout=query_timeout
                )
                prediction = await asyncio.wait_for(
                    synthesize_answer(example.question, results, generator, top_k=top_k),
                    timeout=query_timeout,
                )
            except (TimeoutError, Exception):
                prediction = ""
            e, f = _best(example, prediction)
            em[example.qid] = e
            f1[example.qid] = f

    await asyncio.gather(*(one(ex) for ex in dataset.examples))
    return AnswerScore(em=em, f1=f1)
