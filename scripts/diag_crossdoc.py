"""Type-stratified ablation for the cross-document tier (and RAPTOR).

The aggregate dev ablation mixes question types; the cross-document tier is designed to
help ``cross_document`` questions specifically, so its benefit can be washed out. This
runs full / wo_cross_doc / wo_raptor on the ``cross_document``-type dev subset only,
holding everything else fixed, and reports retrieval + answer metrics with a paired
bootstrap vs full. Reuses the cached store and generations from the dev run.

    uv run python scripts/diag_crossdoc.py
"""

from __future__ import annotations

import asyncio
import time

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.presets import apply
from sage.config.schema import PipelineConfig
from sage.eval.ablate import run_dataset
from sage.eval.answer import answer_eval
from sage.eval.dataset import RetrievalDataset, index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import bootstrap_ci, paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore

CONCURRENCY = 8
TOP_K = 10
ANSWER_K = 5
CONFIGS = ["full", "wo_cross_doc", "wo_raptor"]


async def main() -> None:
    t0 = time.time()
    ds = build_hetdocqa_dataset(
        "data/hetdocqa/hetdocqa.jsonl", "data/hetdocqa/corpus_manifest.json",
        cache_dir=".cache/hetdoc/docs",
    )
    cache = CallCache(".cache/hetdocqa", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model="openai/gpt-4.1-mini", timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hetdocqa/db", dim=dim)
    await index_passages(store, embedder, ds.corpus)  # cached; RAPTOR already built

    # cross_document-type dev subset (full corpus kept as distractors).
    ex = [e for e in ds.examples
          if e.metadata.get("split") == "dev" and e.metadata.get("type") == "cross_document"]
    qids = [e.qid for e in ex]
    sub = RetrievalDataset(
        name="hetdocqa-xdoc-dev", examples=ex, corpus=ds.corpus,
        qrels={q: ds.qrels[q] for q in qids if q in ds.qrels},
    )
    print(f"cross_document dev subset: {len(ex)} queries (corpus {len(ds.corpus)} chunks)")

    base = full(PipelineConfig())
    base.raptor.enabled = True
    rows = []
    for name in CONFIGS:
        cfg = apply(name, base)
        pipe = build_retrieval_pipeline(
            cfg, embedder=embedder, store=store, generator=generator, reranker=reranker
        )
        run = dict(await run_dataset(
            pipe, sub, top_k=TOP_K, semaphore=asyncio.Semaphore(CONCURRENCY)
        ))
        qrels = {q: sub.qrels[q] for q in run if q in sub.qrels}
        ndcg = retrieval_metrics_per_query(qrels, run, "nDCG@10")
        rec = retrieval_metrics_per_query(qrels, run, "R@10")
        ans = await answer_eval(pipe, sub, generator, top_k=ANSWER_K, concurrency=CONCURRENCY)
        rows.append((name, ndcg, rec, ans))
        print(f"  [{name}] done ({time.time() - t0:.0f}s)", flush=True)

    full_f1 = {q: rows[0][3].f1.get(q, 0.0) for q in qids}
    print("\n===== cross_document-type dev: cross-doc / RAPTOR isolation =====")
    print(f"{'config':14} {'nDCG@10':>9} {'R@10':>7} {'EM':>7} {'F1':>7} {'F1 95% CI':>16} {'p(F1)':>7}")
    for name, ndcg, rec, ans in rows:
        mean = lambda d: sum(d.get(q, 0.0) for q in qids) / max(1, len(qids))  # noqa: E731
        f1v = [ans.f1.get(q, 0.0) for q in qids]
        _, lo, hi = bootstrap_ci(f1v, seed=0)
        p = (paired_bootstrap_test(f1v, [full_f1[q] for q in qids], seed=0)
             if name != "full" else float("nan"))
        print(f"{name:14} {mean(ndcg):>9.4f} {mean(rec):>7.4f} {ans.mean_em:>7.4f} "
              f"{ans.mean_f1:>7.4f}   [{lo:.3f}, {hi:.3f}] {p:>7.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
