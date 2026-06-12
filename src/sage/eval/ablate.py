"""Run an ablation matrix over a dataset and compare with statistical rigor.

Each ablation is a named transform of the full configuration (see
:mod:`sage.config.presets`). Given a pre-indexed store, every ablation runs over the
same queries; results are compared to a reference configuration with per-query
bootstrap confidence intervals, a paired bootstrap test, and multiple-comparison
correction. The store must already be indexed for the base configuration; ablations
that only change query-time behaviour reuse it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from sage.config.presets import apply
from sage.config.schema import PipelineConfig
from sage.core.protocols import Embedder, Generator, Reranker, VectorStore
from sage.eval.dataset import RetrievalDataset
from sage.eval.metrics import retrieval_metrics, retrieval_metrics_per_query
from sage.eval.stats import holm_bonferroni, paired_bootstrap_test, paired_diff_ci
from sage.pipeline.assembly import build_retrieval_pipeline
from sage.pipeline.retrieval import RetrievalPipeline

__all__ = [
    "AblationOutcome",
    "Comparison",
    "compare_to_reference",
    "run_ablations",
    "run_dataset",
]


async def run_dataset(
    pipeline: RetrievalPipeline,
    dataset: RetrievalDataset,
    *,
    top_k: int,
    semaphore: asyncio.Semaphore,
    query_timeout: float = 90.0,
) -> list[tuple[str, dict[str, float]]]:
    """Run a pipeline over every question concurrently; return (qid, run) pairs.

    Each query is bounded by ``query_timeout`` and failures degrade to an empty
    result, so one stuck backend call cannot freeze the whole batch.
    """

    async def one(example: object) -> tuple[str, dict[str, float]]:
        qid: str = example.qid  # type: ignore[attr-defined]
        async with semaphore:
            try:
                results, _ = await asyncio.wait_for(
                    pipeline.run(example.question, top_k=top_k),  # type: ignore[attr-defined]
                    timeout=query_timeout,
                )
                return qid, {r.chunk_id: float(r.relevance_score) for r in results}
            except (TimeoutError, Exception):
                return qid, {}

    return await asyncio.gather(*(one(ex) for ex in dataset.examples))


@dataclass(slots=True)
class AblationOutcome:
    name: str
    metrics: dict[str, float]
    per_query: dict[str, float]  # primary-metric score per query id


@dataclass(slots=True)
class Comparison:
    name: str
    metrics: dict[str, float]
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    significant: bool = field(default=False)


async def run_ablations(
    names: Sequence[str],
    base: PipelineConfig,
    dataset: RetrievalDataset,
    *,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
    top_k: int = 10,
    primary_metric: str = "nDCG@10",
    measures: Sequence[str] | None = None,
    concurrency: int = 8,
    router_override: object | None = None,
) -> list[AblationOutcome]:
    """Run each named ablation over the dataset and collect metrics + per-query scores.

    ``router_override`` (a fitted :class:`~sage.core.protocols.Router`) replaces the
    config-resolved router on every pipeline, so component ablations hold the routing
    decision fixed -- isolating the ablated component from routing noise. Ablations that
    explicitly change the router (the routing triad) should not pass it.
    """
    outcomes: list[AblationOutcome] = []
    semaphore = asyncio.Semaphore(concurrency)
    for name in names:
        cfg = apply(name, base)
        pipeline = build_retrieval_pipeline(
            cfg, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        if router_override is not None:
            pipeline.router = router_override  # type: ignore[assignment]
        run = dict(await run_dataset(pipeline, dataset, top_k=top_k, semaphore=semaphore))
        qrels = {q: dataset.qrels[q] for q in run if q in dataset.qrels}
        outcomes.append(
            AblationOutcome(
                name=name,
                metrics=retrieval_metrics(qrels, run, measures or (primary_metric,)),
                per_query=retrieval_metrics_per_query(qrels, run, primary_metric),
            )
        )
    return outcomes


def compare_to_reference(
    outcomes: Sequence[AblationOutcome],
    reference: str,
    *,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[Comparison]:
    """Compare each ablation to the reference: paired CI, p-value, Holm correction."""
    ref = next(o for o in outcomes if o.name == reference)
    others = [o for o in outcomes if o.name != reference]
    qids = sorted(ref.per_query)

    comparisons: list[Comparison] = []
    pvalues: list[float] = []
    for o in others:
        a = [o.per_query.get(q, 0.0) for q in qids]
        b = [ref.per_query.get(q, 0.0) for q in qids]
        delta, lo, hi = paired_diff_ci(a, b, seed=seed)
        p = paired_bootstrap_test(a, b, seed=seed)
        pvalues.append(p)
        comparisons.append(
            Comparison(o.name, o.metrics, delta=delta, ci_low=lo, ci_high=hi, p_value=p)
        )

    rejects = holm_bonferroni(pvalues, alpha=alpha)
    for comp, reject in zip(comparisons, rejects, strict=True):
        # Significant only if corrected test rejects AND the CI excludes zero.
        comp.significant = bool(reject and (comp.ci_low > 0 or comp.ci_high < 0))
    return comparisons
