"""Ablation matrix on a benchmark, with retrieval AND answer metrics.

Runs the contributions on datasets where they should bite and reports answer-side
EM/F1 (not just nDCG), so the answer-acting contributions (HyDE/DPHF, CRAG/SSCC) are
visible. Two tables are produced:

* **routing triad** -- keyword / EGR / compositional / oracle routers, plus the three
  fixed strategies, all scored from one set of forced-strategy runs (no re-retrieval).
  This isolates the router. EGR's entropy signal is degenerate on normalized
  embeddings (it collapses to a constant strategy); the compositional router routes on
  query intent + anchoring scale and is cross-fitted for leakage-free decisions.
* **component ablation** -- full + each -component, with the router held fixed at the
  (fitted) compositional router so component effects are not confounded by routing.

    uv run python scripts/run_benchmark.py musique
    uv run python scripts/run_benchmark.py qasper
    uv run python scripts/run_benchmark.py hetdocqa
"""

from __future__ import annotations

import asyncio
import sys
import time

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.presets import apply
from sage.config.schema import PipelineConfig
from sage.core.types import Strategy, StrategyDecision
from sage.eval.ablate import AblationOutcome, run_ablations, run_dataset
from sage.eval.answer import AnswerScore, answer_eval
from sage.eval.benchmarks import load_musique, load_qasper
from sage.eval.dataset import RetrievalDataset, build_raptor_index, index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import bootstrap_ci, paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.pipeline import build_retrieval_pipeline
from sage.routing.compositional import ROUTED_STRATEGIES, cross_fit_decisions
from sage.routing.egr import routing_entropy
from sage.routing.keyword import KeywordRouter
from sage.routing.oracle import OracleRouter
from sage.store import LanceDBStore


class _ForcedRouter:
    """Router that always returns a fixed strategy (for forced-strategy runs)."""

    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    async def route(self, query: str, query_vector: object, store: object) -> StrategyDecision:
        return StrategyDecision(strategy=self._strategy)


CONCURRENCY = 8
TOP_K = 10
ANSWER_K = 5
# Ablations/sweeps run on the dev split; the test split is reserved for the final
# frozen run (no test-set tuning). Other benchmarks have no split metadata -> all.
ABLATION_SPLIT = "dev"
SAGE_GENERATOR = "openai/gpt-4.1-mini"

# Forced first-stage strategies the routers choose among (DPHF == dual-path HyDE).
FORCED = [("semantic", Strategy.SEMANTIC), ("dphf", Strategy.DPHF), ("stepback", Strategy.STEP_BACK)]
_LABELS = ["semantic", "dphf", "stepback"]

# Component ablations (each contribution isolated). Routing is evaluated separately in
# the triad, so no router_* rows here; the router is held fixed at compositional.
ABLATIONS = [
    "full",
    "wo_dphf",        # DPHF -> single-path HyDE
    "wo_hyde",        # HyDE -> query-only dense
    "wo_sscc",        # SSCC -> single-threshold CRAG
    "wo_crag",        # correction off entirely
    "wo_rerank",      # cross-encoder off
    "wo_graph",       # GAHR off
    "wo_raptor",      # hierarchy off
    "wo_cross_doc",   # cross-document tier off
    "semantic_only",  # dense, no composition
]


def _load(name: str) -> tuple[RetrievalDataset, bool]:
    if name == "musique":
        return load_musique(max_queries=200), False   # short passages: RAPTOR N/A
    if name == "qasper":
        return load_qasper(max_papers=80), True        # full papers: RAPTOR applies
    if name == "hetdocqa":
        ds = build_hetdocqa_dataset(
            "data/hetdocqa/hetdocqa.jsonl",
            "data/hetdocqa/corpus_manifest.json",
            cache_dir=".cache/hetdoc/docs",
        )
        return ds, True                                # heterogeneous, long docs
    raise SystemExit(f"unknown benchmark {name!r}; choose musique|qasper|hetdocqa")


def _subset_split(dataset: RetrievalDataset, split: str) -> RetrievalDataset:
    """Keep only queries in ``split`` (the full corpus stays, as distractors)."""
    examples = [ex for ex in dataset.examples if ex.metadata.get("split") == split]
    if not examples:
        return dataset  # no split metadata (musique/qasper): use everything
    qids = {ex.qid for ex in examples}
    qrels = {q: r for q, r in dataset.qrels.items() if q in qids}
    return RetrievalDataset(
        name=f"{dataset.name}-{split}", examples=examples, corpus=dataset.corpus, qrels=qrels
    )


def _kw_label(strategy: Strategy) -> str:
    if strategy is Strategy.SEMANTIC:
        return "semantic"
    if strategy is Strategy.STEP_BACK:
        return "stepback"
    return "dphf"  # HYDE / DPHF dual-path


def _mean(d: dict[str, float], qids: list[str]) -> float:
    return float(np.mean([d.get(q, 0.0) for q in qids])) if qids else 0.0


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "musique"
    split = sys.argv[2] if len(sys.argv) > 2 else ABLATION_SPLIT
    t0 = time.time()
    dataset, raptor_on = _load(name)
    print(f"{name}: {len(dataset.examples)} questions over {len(dataset.corpus)} passages")

    cache = CallCache(f".cache/{name}", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
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
        n = await build_raptor_index(store, embedder, generator, base.raptor, seed=base.seed)
        print(f"raptor index: +{n} summary nodes ({time.time() - t0:.0f}s)", flush=True)
    # Evaluate on one split (full corpus kept as distractors). Dev for development;
    # the frozen test split is run exactly once for the final numbers.
    dataset = _subset_split(dataset, split)
    print(f"indexed ({time.time() - t0:.0f}s); evaluating {len(dataset.examples)} "
          f"queries (split={split if dataset.name.endswith(split) else 'all'})")

    qids = [ex.qid for ex in dataset.examples]
    qtext = {ex.qid: ex.question for ex in dataset.examples}

    # ---- forced-strategy runs: per-query nDCG + F1 (+ router features) -------------
    per_ndcg: dict[str, dict[str, float]] = {}
    per_f1: dict[str, dict[str, float]] = {}
    per_em: dict[str, dict[str, float]] = {}
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
        ans = await answer_eval(pipe, dataset, generator, top_k=ANSWER_K, concurrency=CONCURRENCY)
        per_f1[label] = ans.f1
        per_em[label] = ans.em
        print(f"  forced[{label}] nDCG@10={_mean(per_ndcg[label], qids):.4f} "
              f"F1={ans.mean_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)

    embeddings = np.vstack([np.asarray(await embedder.embed_query(qtext[q]), dtype=np.float64)
                            for q in qids])
    distances: list[np.ndarray] = []
    for i in range(len(qids)):
        hits = await store.search(embeddings[i], rcfg.egr_k)
        distances.append(np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits]))

    # ---- compositional router: cross-fitted, leakage-free decisions ----------------
    rewards = {
        s: np.array([per_ndcg[_LABELS[i]].get(q, 0.0) for q in qids])
        for i, s in enumerate(ROUTED_STRATEGIES)
    }
    comp_dec = cross_fit_decisions(
        rcfg, qids, [qtext[q] for q in qids], distances, embeddings, rewards
    )
    strat_to_label = dict(zip(ROUTED_STRATEGIES, _LABELS, strict=True))
    comp_router = OracleRouter(rcfg)
    comp_router.set_decisions({qtext[q]: comp_dec[q] for q in qids})

    # ---- component ablations, router held fixed at compositional -------------------
    rows: list[tuple[str, AblationOutcome, AnswerScore]] = []
    for abl in ABLATIONS:
        (o,) = await run_ablations(
            [abl], base, dataset,
            embedder=embedder, store=store, generator=generator, reranker=reranker,
            top_k=TOP_K, primary_metric="nDCG@10",
            measures=("nDCG@10", "R@10", "Success@10", "RR@10"),
            concurrency=CONCURRENCY, router_override=comp_router,
        )
        cfg = apply(abl, base)
        pipe = build_retrieval_pipeline(
            cfg, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        pipe.router = comp_router  # type: ignore[assignment]
        ans = await answer_eval(pipe, dataset, generator, top_k=ANSWER_K, concurrency=CONCURRENCY)
        rows.append((abl, o, ans))
        print(f"  [{abl}] nDCG@10={o.metrics.get('nDCG@10', 0):.4f} "
              f"EM={ans.mean_em:.4f} F1={ans.mean_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- routing triad (scored from the forced runs; no re-retrieval) --------------
    kw_router = KeywordRouter(rcfg)
    decisions: dict[str, dict[str, str]] = {
        "always-semantic": dict.fromkeys(qids, "semantic"),
        "always-dphf": dict.fromkeys(qids, "dphf"),
        "always-stepback": dict.fromkeys(qids, "stepback"),
        "keyword": {q: _kw_label((await kw_router.route(qtext[q], embeddings[i], store)).strategy)
                    for i, q in enumerate(qids)},
        "egr": {q: ("semantic" if (h := routing_entropy(distances[i], rcfg.egr_temperature))
                    < rcfg.egr_tau_low else "dphf" if h < rcfg.egr_tau_high else "stepback")
                for i, q in enumerate(qids)},
        "compositional": {q: strat_to_label[comp_dec[q]] for q in qids},
        "oracle": {q: max(_LABELS, key=lambda s: per_ndcg[s].get(q, 0.0)) for q in qids},
    }
    best_fixed = max(["always-semantic", "always-dphf", "always-stepback"],
                     key=lambda r: _mean({q: per_ndcg[decisions[r][q]][q] for q in qids}, qids))
    bf_f1 = [per_f1[decisions[best_fixed][q]][q] for q in qids]

    print(f"\n===== {name}: routing triad ({time.time() - t0:.0f}s) =====  best-fixed={best_fixed}")
    print(f"{'router':16} {'nDCG@10':>9} {'EM':>7} {'F1':>7} {'F1 95% CI':>16} {'p(F1 vs bf)':>11}")
    for rname, dec in decisions.items():
        nd = _mean({q: per_ndcg[dec[q]][q] for q in qids}, qids)
        em = _mean({q: per_em[dec[q]][q] for q in qids}, qids)
        f1_vals = [per_f1[dec[q]][q] for q in qids]
        _, lo, hi = bootstrap_ci(f1_vals, seed=0)
        p = paired_bootstrap_test(f1_vals, bf_f1, seed=0)
        print(f"{rname:16} {nd:>9.4f} {em:>7.4f} {float(np.mean(f1_vals)):>7.4f} "
              f"  [{lo:.3f}, {hi:.3f}] {p:>11.3f}")

    # ---- component ablation table (router fixed = compositional) --------------------
    full_f1 = next(a for n, _, a in rows if n == "full").f1
    print(f"\n===== {name}: component ablation ({time.time() - t0:.0f}s) =====")
    print(f"{'config':16} {'nDCG@10':>9} {'Su@10':>7} {'EM':>7} {'F1':>7} {'F1 95% CI':>16} {'p(F1)':>7}")
    for n, o, a in rows:
        f1_vals = [a.f1[q] for q in qids if q in a.f1]
        _, lo, hi = bootstrap_ci(f1_vals, seed=0)
        p = (paired_bootstrap_test([a.f1.get(q, 0.0) for q in qids],
                                   [full_f1[q] for q in qids], seed=0)
             if n != "full" else float("nan"))
        print(f"{n:16} {o.metrics.get('nDCG@10', 0):>9.4f} "
              f"{o.metrics.get('Success@10', 0):>7.3f} {a.mean_em:>7.4f} {a.mean_f1:>7.4f} "
              f"  [{lo:.3f}, {hi:.3f}] {p:>7.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
