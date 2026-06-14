"""Retrieval-only ablation matrix (no answer generation).

A fast variant of ``run_benchmark.py`` for the reranker robustness check (review
issue M1): it runs the same component ablations but scores only retrieval
(nDCG@10, Success@10, R@10), skipping the gpt-4.1-mini answer generation that
dominates wall-clock. HyDE hypotheses and CRAG rewrites are query-keyed, so under
a swapped reranker they are cache hits and only the rerank step recomputes; a full
four-benchmark sweep finishes in minutes rather than hours.

The reranker is selected by $SAGE_RERANK_MODEL (default jina-reranker-v3-mlx); the
per-query nDCG dump lands at results/{name}_{split}_retonly{SAGE_RESULT_SUFFIX}.json.

    SAGE_RERANK_MODEL=BAAI-bge-reranker-v2-m3-mlx-fp16 SAGE_RESULT_SUFFIX=_bge \\
        uv run python scripts/run_retrieval_only.py hetdocqa dev
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.schema import PipelineConfig
from sage.core.types import Strategy, StrategyDecision
from sage.eval.ablate import AblationOutcome, run_ablations, run_dataset
from sage.eval.benchmarks import load_musique, load_qasper
from sage.eval.dataset import RetrievalDataset, build_raptor_index, index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import bootstrap_ci, paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.pipeline import build_retrieval_pipeline
from sage.routing.compositional import ROUTED_STRATEGIES, cross_fit_decisions
from sage.routing.oracle import OracleRouter
from sage.store import LanceDBStore


class _ForcedRouter:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    async def route(self, query: str, query_vector: object, store: object) -> StrategyDecision:
        return StrategyDecision(strategy=self._strategy)


CONCURRENCY = 8
TOP_K = 10
ABLATION_SPLIT = "dev"
SAGE_GENERATOR = "openai/gpt-4.1-mini"
FORCED = [("semantic", Strategy.SEMANTIC), ("dphf", Strategy.DPHF), ("stepback", Strategy.STEP_BACK)]
_LABELS = ["semantic", "dphf", "stepback"]

# Same component ablations as run_benchmark.py (retrieval-affecting ones).
ABLATIONS = [
    "full", "wo_dphf", "wo_hyde", "wo_sscc", "wo_crag", "wo_rerank",
    "wo_graph", "wo_raptor", "wo_cross_doc", "semantic_only",
]


def _load(name: str) -> tuple[RetrievalDataset, bool]:
    if name == "musique":
        return load_musique(max_queries=200), False
    if name == "qasper":
        return load_qasper(max_papers=80), True
    if name == "hetdocqa":
        ds = build_hetdocqa_dataset(
            "data/hetdocqa/hetdocqa.jsonl",
            "data/hetdocqa/corpus_manifest.json",
            cache_dir=".cache/hetdoc/docs",
        )
        return ds, True
    raise SystemExit(f"unknown benchmark {name!r}; choose musique|qasper|hetdocqa")


def _subset_split(dataset: RetrievalDataset, split: str) -> RetrievalDataset:
    examples = [ex for ex in dataset.examples if ex.metadata.get("split") == split]
    if not examples:
        return dataset
    qids = {ex.qid for ex in examples}
    qrels = {q: r for q, r in dataset.qrels.items() if q in qids}
    return RetrievalDataset(
        name=f"{dataset.name}-{split}", examples=examples, corpus=dataset.corpus, qrels=qrels
    )


def _mean(d: dict[str, float], qids: list[str]) -> float:
    return float(np.mean([d.get(q, 0.0) for q in qids])) if qids else 0.0


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "hetdocqa"
    split = sys.argv[2] if len(sys.argv) > 2 else ABLATION_SPLIT
    t0 = time.time()
    dataset, raptor_on = _load(name)
    print(f"{name}: {len(dataset.examples)} questions over {len(dataset.corpus)} passages")

    cache = CallCache(f".cache/{name}", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    rerank_model = os.environ.get("SAGE_RERANK_MODEL", "jina-reranker-v3-mlx")
    reranker = RerankClient(BackendConfig(provider="omlx", model=rerank_model), cache)
    print(f"reranker: {rerank_model}", flush=True)
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model=SAGE_GENERATOR, timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(f".cache/{name}/db", dim=dim)
    base = full(PipelineConfig())
    base.raptor.enabled = raptor_on
    rcfg = base.router

    await index_passages(store, embedder, dataset.corpus)
    if raptor_on:
        await build_raptor_index(store, embedder, generator, base.raptor, seed=base.seed)
    dataset = _subset_split(dataset, split)
    qids = [ex.qid for ex in dataset.examples]
    qtext = {ex.qid: ex.question for ex in dataset.examples}
    print(f"indexed ({time.time() - t0:.0f}s); evaluating {len(qids)} queries (split={split})",
          flush=True)

    # ---- forced-strategy retrieval runs: per-query nDCG (for the router fit) --------
    per_ndcg: dict[str, dict[str, float]] = {}
    for label, strat in FORCED:
        pipe = build_retrieval_pipeline(
            base, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        pipe.router = _ForcedRouter(strat)  # type: ignore[assignment]
        run = dict(await run_dataset(
            pipe, dataset, top_k=TOP_K, semaphore=asyncio.Semaphore(CONCURRENCY)
        ))
        qrels = {q: dataset.qrels[q] for q in run if q in dataset.qrels}
        per_ndcg[label] = retrieval_metrics_per_query(qrels, run, "nDCG@10")
        print(f"  forced[{label}] nDCG@10={_mean(per_ndcg[label], qids):.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- compositional router (cross-fitted on retrieval reward only) --------------
    embeddings = np.vstack([np.asarray(await embedder.embed_query(qtext[q]), dtype=np.float64)
                            for q in qids])
    distances: list[np.ndarray] = []
    for i in range(len(qids)):
        hits = await store.search(embeddings[i], rcfg.egr_k)
        distances.append(np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits]))
    rewards = {s: np.array([per_ndcg[_LABELS[i]].get(q, 0.0) for q in qids])
               for i, s in enumerate(ROUTED_STRATEGIES)}
    comp_dec = cross_fit_decisions(
        rcfg, qids, [qtext[q] for q in qids], distances, embeddings, rewards
    )
    comp_router = OracleRouter(rcfg)
    comp_router.set_decisions({qtext[q]: comp_dec[q] for q in qids})

    # ---- component ablations (router fixed = compositional), retrieval metrics -----
    rows: list[AblationOutcome] = []
    for abl in ABLATIONS:
        (o,) = await run_ablations(
            [abl], base, dataset,
            embedder=embedder, store=store, generator=generator, reranker=reranker,
            top_k=TOP_K, primary_metric="nDCG@10",
            measures=("nDCG@10", "R@10", "Success@10", "RR@10"),
            concurrency=CONCURRENCY, router_override=comp_router,
        )
        rows.append(o)
        print(f"  [{abl}] nDCG@10={o.metrics.get('nDCG@10', 0):.4f} "
              f"Su@10={o.metrics.get('Success@10', 0):.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- print paired tests vs full (retrieval nDCG) -------------------------------
    full_nd = next(o for o in rows if o.name == "full").per_query
    print(f"\n===== {name}/{split} retrieval ablation (reranker={rerank_model}) =====")
    print(f"{'config':16} {'nDCG@10':>9} {'Su@10':>7} {'p(nDCG vs full)':>16}")
    for o in rows:
        nd = o.metrics.get("nDCG@10", 0.0)
        if o.name == "full":
            print(f"{o.name:16} {nd:>9.4f} {o.metrics.get('Success@10', 0):>7.3f} {'--':>16}")
            continue
        a = [o.per_query.get(q, 0.0) for q in qids]
        b = [full_nd.get(q, 0.0) for q in qids]
        p = paired_bootstrap_test(a, b, seed=0)
        print(f"{o.name:16} {nd:>9.4f} {o.metrics.get('Success@10', 0):>7.3f} {p:>16.3f}")

    # ---- persist per-query nDCG for the robustness analyzer -------------------------
    Path("results").mkdir(exist_ok=True)
    dump = {
        "name": name, "split": split, "reranker": rerank_model, "qids": qids,
        "configs": {o.name: {"metrics": o.metrics, "ndcg": o.per_query} for o in rows},
    }
    suffix = os.environ.get("SAGE_RESULT_SUFFIX", "")
    Path(f"results/{name}_{split}_retonly{suffix}.json").write_text(json.dumps(dump))
    print(f"\ntotal {time.time() - t0:.0f}s -> results/{name}_{split}_retonly{suffix}.json")


if __name__ == "__main__":
    asyncio.run(main())
