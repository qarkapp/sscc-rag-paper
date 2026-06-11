"""Run a retrieval pipeline over a dataset and compute metrics."""

from __future__ import annotations

from dataclasses import dataclass

from sage.eval.dataset import RetrievalDataset
from sage.eval.metrics import DEFAULT_MEASURES, retrieval_metrics
from sage.pipeline.retrieval import RetrievalPipeline

__all__ = ["EvalResult", "evaluate_retrieval"]


@dataclass(slots=True)
class EvalResult:
    dataset: str
    metrics: dict[str, float]
    run: dict[str, dict[str, float]]
    n_queries: int


async def evaluate_retrieval(
    pipeline: RetrievalPipeline,
    dataset: RetrievalDataset,
    *,
    top_k: int = 10,
    measures: tuple[str, ...] = DEFAULT_MEASURES,
) -> EvalResult:
    """Evaluate a pipeline's retrieval quality on a dataset.

    The corpus must already be indexed into the pipeline's store. Returns aggregate
    metrics and the full run (for bootstrap confidence intervals downstream).
    """
    run: dict[str, dict[str, float]] = {}
    for example in dataset.examples:
        results, _ = await pipeline.run(example.question, top_k=top_k)
        run[example.qid] = {r.chunk_id: float(r.relevance_score) for r in results}

    # Restrict qrels to evaluated queries so metrics are computed over the same set.
    qrels = {q: dataset.qrels[q] for q in run if q in dataset.qrels}
    metrics = retrieval_metrics(qrels, run, measures)
    return EvalResult(dataset=dataset.name, metrics=metrics, run=run, n_queries=len(run))
