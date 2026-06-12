"""Diagnose whether the EGR entropy signal can actually route.

Evidence-first check before redesigning the router. For each query we compute the
EGR routing entropy and the kNN distance moments, then derive a per-query *oracle
label* = the forced strategy (semantic / dphf / step_back) that maximizes retrieval
nDCG@10 (cache hits from the benchmark run -- no generation, no live calls). We then
ask three questions:

  1. Is the entropy signal degenerate?  (spread vs the log K ceiling)
  2. Does entropy separate the oracle classes?  (mean entropy per class; if the
     class-conditional means overlap, a threshold on entropy cannot route)
  3. How do EGR and the keyword heuristic actually agree with the oracle?

    uv run python scripts/diag_router.py musique
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
from sage.config.schema import PipelineConfig
from sage.core.types import Strategy, StrategyDecision
from sage.eval.benchmarks import load_musique, load_qasper
from sage.eval.dataset import index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import paired_bootstrap_test
from sage.pipeline import build_retrieval_pipeline
from sage.routing.egr import routing_entropy
from sage.routing.keyword import KeywordRouter
from sage.store import LanceDBStore

CONCURRENCY = 4
TOP_K = 10
FORCED = [("semantic", Strategy.SEMANTIC), ("dphf", Strategy.DPHF), ("stepback", Strategy.STEP_BACK)]
# EGR maps onto these three; HYDE is the keyword router's dual-path label == DPHF.
_LABELS = ["semantic", "dphf", "stepback"]


class _ForcedRouter:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy

    async def route(self, query: str, query_vector: object, store: object) -> StrategyDecision:
        return StrategyDecision(strategy=self._strategy)


def _egr_label(entropy: float, tau_low: float, tau_high: float) -> str:
    if entropy < tau_low:
        return "semantic"
    if entropy < tau_high:
        return "dphf"
    return "stepback"


def _kw_label(strategy: Strategy) -> str:
    if strategy is Strategy.SEMANTIC:
        return "semantic"
    if strategy is Strategy.STEP_BACK:
        return "stepback"
    return "dphf"  # HYDE / DPHF dual-path


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "musique"
    t0 = time.time()
    if name == "musique":
        dataset, raptor_on = load_musique(max_queries=200), False
    elif name == "qasper":
        dataset, raptor_on = load_qasper(max_papers=80), True
    else:
        raise SystemExit("choose musique|qasper")
    print(f"{name}: {len(dataset.examples)} queries / {len(dataset.corpus)} passages")

    cache = CallCache(f".cache/{name}", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    # Cached from the benchmark run; HyDE/step-back generations replay as cache hits.
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model="openai/gpt-4.1-mini", timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(f".cache/{name}/db", dim=dim)
    await index_passages(store, embedder, dataset.corpus)

    base = full(PipelineConfig())
    base.raptor.enabled = raptor_on
    rcfg = base.router
    k, temp = rcfg.egr_k, rcfg.egr_temperature
    print(f"indexed ({time.time() - t0:.0f}s); egr_k={k} T={temp} "
          f"tau=({rcfg.egr_tau_low},{rcfg.egr_tau_high}) logK={np.log(k):.3f}")

    # 1. entropy + distance features per query (cached embeds + kNN).
    qids = [ex.qid for ex in dataset.examples]
    qtext = {ex.qid: ex.question for ex in dataset.examples}
    entropy: dict[str, float] = {}
    dist_by_q: dict[str, np.ndarray] = {}
    emb_by_q: dict[str, np.ndarray] = {}
    for ex in dataset.examples:
        qvec = await embedder.embed_query(ex.question)
        hits = await store.search(qvec, k)
        d = np.array([1.0 / max(h.relevance_score, 1e-9) - 1.0 for h in hits])
        entropy[ex.qid] = routing_entropy(d, temp)
        dist_by_q[ex.qid] = d
        emb_by_q[ex.qid] = np.asarray(qvec, dtype=np.float64)

    # 2. oracle label = argmax forced-strategy nDCG@10 per query (cache hits).
    per_strat: dict[str, dict[str, float]] = {}
    for sname, strat in FORCED:
        pipe = build_retrieval_pipeline(
            base, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        pipe.router = _ForcedRouter(strat)  # type: ignore[assignment]
        from sage.eval.ablate import run_dataset

        run = dict(await run_dataset(
            pipe, dataset, top_k=TOP_K, semaphore=asyncio.Semaphore(CONCURRENCY)
        ))
        qrels = {q: dataset.qrels[q] for q in run if q in dataset.qrels}
        per_strat[sname] = retrieval_metrics_per_query(qrels, run, "nDCG@10")
    oracle: dict[str, str] = {}
    margins: list[float] = []
    for q in qids:
        scores = {s: per_strat[s].get(q, 0.0) for s in _LABELS}
        best = max(_LABELS, key=lambda s: scores[s])
        oracle[q] = best
        ranked = sorted(scores.values(), reverse=True)
        margins.append(ranked[0] - ranked[1])

    # 3. EGR + keyword decisions.
    egr = {q: _egr_label(entropy[q], rcfg.egr_tau_low, rcfg.egr_tau_high) for q in qids}
    kw_router = KeywordRouter(rcfg)
    kw = {}
    for q in qids:
        dec = await kw_router.route(qtext[q], np.zeros(dim), store)
        kw[q] = _kw_label(dec.strategy)

    # ---- report -------------------------------------------------------------
    ent = np.array([entropy[q] for q in qids])
    print("\n== 1. entropy distribution (is the signal degenerate?) ==")
    print(f"  min={ent.min():.3f}  p25={_pct(ent,25):.3f}  median={np.median(ent):.3f}  "
          f"p75={_pct(ent,75):.3f}  max={ent.max():.3f}  std={ent.std():.3f}")
    print(f"  log K ceiling = {np.log(k):.3f};  mean/ceiling = {ent.mean()/np.log(k):.3f}  "
          f"(1.0 => uniform kNN => no signal)")

    print("\n== 2. does entropy separate the oracle classes? ==")
    counts = {s: sum(1 for q in qids if oracle[q] == s) for s in _LABELS}
    print(f"  oracle label counts: {counts}")
    for s in _LABELS:
        cls = np.array([entropy[q] for q in qids if oracle[q] == s])
        if cls.size:
            print(f"    {s:9} n={cls.size:3}  entropy mean={cls.mean():.3f} std={cls.std():.3f}")
    # one-way separation: between-class variance / within-class variance (F-like ratio)
    grand = ent.mean()
    between = sum(
        len([q for q in qids if oracle[q] == s]) *
        (np.mean([entropy[q] for q in qids if oracle[q] == s]) - grand) ** 2
        for s in _LABELS if any(oracle[q] == s for q in qids)
    )
    within = sum(
        sum((entropy[q] - np.mean([entropy[r] for r in qids if oracle[r] == oracle[q]])) ** 2
            for q in qids if oracle[q] == s)
        for s in _LABELS if any(oracle[q] == s for q in qids)
    )
    print(f"  between/within dispersion ratio = {between / max(within, 1e-9):.4f}  "
          f"(~0 => entropy carries no class signal)")

    print("\n== 3. agreement with oracle (3-class) ==")
    egr_acc = sum(1 for q in qids if egr[q] == oracle[q]) / len(qids)
    kw_acc = sum(1 for q in qids if kw[q] == oracle[q]) / len(qids)
    maj = max(counts, key=lambda s: counts[s])
    maj_acc = counts[maj] / len(qids)
    print(f"  EGR     vs oracle: {egr_acc:.3f}")
    print(f"  keyword vs oracle: {kw_acc:.3f}")
    print(f"  always-'{maj}' baseline: {maj_acc:.3f}  (router must beat this to be useful)")
    print(f"  EGR label distribution:     "
          f"{({s: sum(1 for q in qids if egr[q]==s) for s in _LABELS})}")
    print(f"  keyword label distribution: "
          f"{({s: sum(1 for q in qids if kw[q]==s) for s in _LABELS})}")

    print("\n== 4. how much is even on the table? (mean nDCG@10) ==")
    for s in _LABELS:
        m = np.mean([per_strat[s].get(q, 0.0) for q in qids])
        print(f"  always-{s:9} {m:.4f}")
    omean = np.mean([per_strat[oracle[q]].get(q, 0.0) for q in qids])
    print(f"  per-query oracle {omean:.4f}   (mean win margin over 2nd-best = {np.mean(margins):.4f})")

    # 5. compositional router: cross-fitted, leakage-free, reward-regression decisions.
    from sage.routing.compositional import ROUTED_STRATEGIES, cross_fit_decisions

    embeddings = np.vstack([emb_by_q[q] for q in qids])
    distances = [dist_by_q[q] for q in qids]
    rewards = {
        s: np.array([per_strat[_LABELS[i]].get(q, 0.0) for q in qids])
        for i, s in enumerate(ROUTED_STRATEGIES)
    }
    comp = cross_fit_decisions(rcfg, qids, [qtext[q] for q in qids], distances, embeddings, rewards)
    strat_to_label = dict(zip(ROUTED_STRATEGIES, _LABELS, strict=True))
    comp_label = {q: strat_to_label[comp[q]] for q in qids}
    comp_ndcg = np.array([per_strat[comp_label[q]].get(q, 0.0) for q in qids])
    comp_acc = sum(1 for q in qids if comp_label[q] == oracle[q]) / len(qids)
    best_fixed = max(_LABELS, key=lambda s: np.mean([per_strat[s].get(q, 0.0) for q in qids]))
    bf = np.array([per_strat[best_fixed].get(q, 0.0) for q in qids])
    p = paired_bootstrap_test(comp_ndcg.tolist(), bf.tolist(), seed=0)

    print("\n== 5. compositional router (5-fold cross-fit, reward regression) ==")
    print(f"  label distribution: "
          f"{({s: sum(1 for q in qids if comp_label[q]==s) for s in _LABELS})}")
    print(f"  vs-oracle agreement: {comp_acc:.3f}")
    print(f"  routed nDCG@10 = {comp_ndcg.mean():.4f}   "
          f"(best-fixed always-{best_fixed} = {bf.mean():.4f}, oracle = {omean:.4f})")
    print(f"  paired-bootstrap p(routed vs best-fixed) = {p:.3f}")
    captured = (comp_ndcg.mean() - bf.mean()) / max(omean - bf.mean(), 1e-9)
    print(f"  oracle-gap captured = {captured:+.1%}  (0% = best-fixed, 100% = oracle)")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
