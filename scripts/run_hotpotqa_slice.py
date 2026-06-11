"""Real vertical slice: the ablation matrix on a HotpotQA sample.

Pools the context paragraphs of a HotpotQA-distractor sample into one corpus (qrels
= the gold supporting-fact paragraphs per question), indexes with bge-m3 via oMLX,
and runs the query-time ablation matrix end-to-end (EGR routing, HyDE/step-back,
RRF, SSCC, cross-encoder rerank, graph expansion) with bootstrap CIs and paired
tests, under several OpenRouter generators (a robustness sweep). Embeddings and
reranking stay on local oMLX; queries run concurrently and everything is cached.

    uv run python scripts/run_hotpotqa_slice.py
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.schema import PipelineConfig
from sage.core.types import StrategyDecision
from sage.eval.ablate import AblationOutcome, compare_to_reference, run_ablations, run_dataset
from sage.eval.dataset import QAExample, RetrievalDataset, index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import bootstrap_ci
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore

_SEM = asyncio.Semaphore(8)

N_QUESTIONS = 120
TOP_K = 10
PRIMARY = "Success@10"

# Generator-robustness sweep: the ablation matrix is run under each generator.
# Embeddings and reranking stay on local oMLX; only HyDE/step-back/CRAG generation
# varies. These are distinct from the HetDocQA question generator (DeepSeek).
GENERATORS = [
    ("qwen3.6-35b", "qwen/qwen3.6-35b-a3b"),
    ("grok-4.20", "x-ai/grok-4.20"),
    ("gemini-2.5-flash", "google/gemini-2.5-flash"),
]


def load_slice() -> RetrievalDataset:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    corpus: dict[str, str] = {}
    examples: list[QAExample] = []
    qrels: dict[str, dict[str, int]] = {}
    for row in ds.select(range(N_QUESTIONS)):
        titles = row["context"]["title"]
        sentences = row["context"]["sentences"]
        for title, sents in zip(titles, sentences, strict=False):
            corpus.setdefault(title, (title + ". " + " ".join(sents)).strip())
        gold = dict.fromkeys(row["supporting_facts"]["title"], 1)
        qid = str(row["id"])
        qrels[qid] = gold
        examples.append(QAExample(qid=qid, question=row["question"], answers=(row["answer"],)))
    return RetrievalDataset("hotpotqa-slice", examples, corpus, qrels)


class _ForcedRouter:
    """Router that always returns a fixed strategy (for the oracle upper bound)."""

    def __init__(self, strategy) -> None:  # type: ignore[no-untyped-def]
        self._strategy = strategy

    async def route(self, query, query_vector, store):  # type: ignore[no-untyped-def]
        return StrategyDecision(strategy=self._strategy)


async def _per_query(pipeline, dataset, metric: str) -> dict[str, float]:  # type: ignore[no-untyped-def]
    run = dict(await run_dataset(pipeline, dataset, top_k=TOP_K, semaphore=_SEM))
    qrels = {q: dataset.qrels[q] for q in run if q in dataset.qrels}
    return retrieval_metrics_per_query(qrels, run, metric)


async def main() -> None:
    from sage.core.types import Strategy

    t0 = time.time()
    dataset = load_slice()
    print(f"loaded {len(dataset.examples)} questions over {len(dataset.corpus)} passages")

    cache = CallCache(".cache/hotpot", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)

    dim = await embedder.probe()
    store = LanceDBStore(".cache/hotpot/db", dim=dim)
    await index_passages(store, embedder, dataset.corpus)
    print(f"indexed {len(dataset.corpus)} passages ({time.time() - t0:.0f}s)")

    base = full(PipelineConfig())
    base.raptor.enabled = False  # passages are single paragraphs; RAPTOR does not apply
    ablations = [
        "full",
        "router_keyword",
        "wo_dphf",
        "wo_sscc",
        "wo_rerank",
        "wo_graph",
        "semantic_only",
    ]

    for label, model in GENERATORS:
        generator = ChatGenerator(
            BackendConfig(provider="openrouter", model=model, timeout=120), cache
        )
        outcomes = await run_ablations(
            ablations,
            base,
            dataset,
            embedder=embedder,
            store=store,
            generator=generator,
            reranker=reranker,
            top_k=TOP_K,
            primary_metric=PRIMARY,
            measures=("nDCG@10", "Success@10", "R@10", "RR@10"),
        )

        # Oracle upper bound = per-query max over forced strategies.
        forced: dict[str, dict[str, float]] = {}
        for sname, strat in [
            ("semantic", Strategy.SEMANTIC),
            ("dphf", Strategy.DPHF),
            ("stepback", Strategy.STEP_BACK),
        ]:
            pipe = build_retrieval_pipeline(
                base, embedder=embedder, store=store, generator=generator, reranker=reranker
            )
            pipe.router = _ForcedRouter(strat)
            forced[sname] = await _per_query(pipe, dataset, PRIMARY)
        qids = sorted(forced["semantic"])
        oracle_pq = {
            q: max(forced["semantic"][q], forced["dphf"][q], forced["stepback"][q]) for q in qids
        }
        outcomes.append(
            AblationOutcome(
                "router_oracle", {PRIMARY: float(np.mean(list(oracle_pq.values())))}, oracle_pq
            )
        )

        print(f"\n========== generator: {label} ({model}) ==========")
        print(f"{'config':16} {PRIMARY:>10} {'nDCG@10':>9} {'95% CI':>18}")
        for o in outcomes:
            mean, lo, hi = bootstrap_ci(list(o.per_query.values()), seed=0)
            ndcg = o.metrics.get("nDCG@10", float("nan"))
            print(
                f"{o.name:16} {o.metrics.get(PRIMARY, mean):>10.4f} {ndcg:>9.4f}   [{lo:.3f}, {hi:.3f}]"
            )
        print("-- paired vs semantic_only (Holm-corrected) --")
        for c in compare_to_reference(outcomes, "semantic_only"):
            star = "*" if c.significant else " "
            print(
                f"{c.name:16} d={c.delta:+.4f} [{c.ci_low:+.3f}, {c.ci_high:+.3f}] p={c.p_value:.4f} {star}"
            )
        print(f"(elapsed {time.time() - t0:.0f}s)")

    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
