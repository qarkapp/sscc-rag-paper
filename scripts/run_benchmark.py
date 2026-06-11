"""Ablation matrix on an existing benchmark, with retrieval AND answer metrics.

Unlike the HotpotQA slice (retrieval-only on a homogeneous corpus), this runs the
contributions on datasets where they should bite, and -- crucially -- reports
answer-side EM/F1 so the answer-acting contributions (HyDE/DPHF, CRAG/SSCC, query
rewrite) are visible, not just nDCG. The ablation set isolates each contribution,
including the routing triad (keyword / EGR / oracle) for the smart router.

    uv run python scripts/run_benchmark.py musique
    uv run python scripts/run_benchmark.py qasper
"""

from __future__ import annotations

import asyncio
import sys
import time

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.presets import apply
from sage.config.schema import PipelineConfig
from sage.eval.ablate import run_ablations
from sage.eval.answer import answer_eval
from sage.eval.benchmarks import load_musique, load_qasper
from sage.eval.dataset import RetrievalDataset, index_passages
from sage.eval.stats import bootstrap_ci, paired_bootstrap_test
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore

CONCURRENCY = 4
TOP_K = 10
ANSWER_K = 5

# Controlled generator for HyDE/step-back/CRAG and answer synthesis. A non-reasoning
# GPT-4-family model is used deliberately: it follows instructions, returns visible
# content (reasoning models can spend their whole budget on hidden tokens and return
# an empty string), and is reproducible. It is distinct from the HetDocQA question
# generator (DeepSeek) and any faithfulness judge, preserving the no-self-preference
# firewall.
SAGE_GENERATOR = "openai/gpt-4.1-mini"

# Each contribution isolated as its own row, plus the routing triad. RAPTOR/cross-doc
# rows are meaningful on long-doc corpora (qasper); on musique they are inert but kept
# for a uniform table.
ABLATIONS = [
    "full",
    "router_keyword",   # EGR -> keyword heuristic (lower bound)
    "router_oracle",    # routing upper bound
    "wo_dphf",          # DPHF -> single-path HyDE
    "wo_hyde",          # HyDE -> query-only dense
    "wo_sscc",          # SSCC -> single-threshold CRAG
    "wo_crag",          # correction off entirely
    "wo_rerank",        # cross-encoder off
    "wo_graph",         # GAHR off
    "wo_raptor",        # hierarchy off
    "wo_cross_doc",     # cross-document tier off
    "semantic_only",    # dense, no composition
]


def _load(name: str) -> tuple[RetrievalDataset, bool]:
    if name == "musique":
        return load_musique(max_queries=200), False  # short passages: RAPTOR N/A
    if name == "qasper":
        return load_qasper(max_papers=80), True       # full papers: RAPTOR applies
    raise SystemExit(f"unknown benchmark {name!r}; choose musique|qasper")


async def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "musique"
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
    await index_passages(store, embedder, dataset.corpus)
    print(f"indexed ({time.time() - t0:.0f}s)")

    base = full(PipelineConfig())
    base.raptor.enabled = raptor_on

    rows = []
    for abl in ABLATIONS:
        (o,) = await run_ablations(
            [abl], base, dataset,
            embedder=embedder, store=store, generator=generator, reranker=reranker,
            top_k=TOP_K, primary_metric="nDCG@10",
            measures=("nDCG@10", "R@10", "Success@10", "RR@10"), concurrency=CONCURRENCY,
        )
        cfg = apply(abl, base)
        pipe = build_retrieval_pipeline(
            cfg, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        ans = await answer_eval(pipe, dataset, generator, top_k=ANSWER_K, concurrency=CONCURRENCY)
        rows.append((abl, o, ans))
        print(f"  [{abl}] nDCG@10={o.metrics.get('nDCG@10', 0):.4f} "
              f"EM={ans.mean_em:.4f} F1={ans.mean_f1:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # Significance of answer-F1 vs the full system, per ablation (paired bootstrap).
    full_f1 = next(a for n, _, a in rows if n == "full").f1
    qids = sorted(full_f1)
    print(f"\n===== {name}: ablation matrix ({time.time() - t0:.0f}s) =====")
    print(f"{'config':16} {'nDCG@10':>9} {'Su@10':>7} {'EM':>7} {'F1':>7} {'F1 95% CI':>16} {'p(F1)':>7}")
    for n, o, a in rows:
        f1_vals = [a.f1[q] for q in qids if q in a.f1]
        _, lo, hi = bootstrap_ci(f1_vals, seed=0)
        p = paired_bootstrap_test(
            [a.f1.get(q, 0.0) for q in qids], [full_f1[q] for q in qids], seed=0
        ) if n != "full" else float("nan")
        print(f"{n:16} {o.metrics.get('nDCG@10', 0):>9.4f} "
              f"{o.metrics.get('Success@10', 0):>7.3f} {a.mean_em:>7.4f} {a.mean_f1:>7.4f} "
              f"  [{lo:.3f}, {hi:.3f}] {p:>7.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
