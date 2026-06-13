"""Fast retrieval-only sweep of the MMR relevance/coverage weight on HetDocQA dev.

Reuses the cached candidate pools and cached facet decompositions, so it runs in ~1 min
with no generation. For each ``relevance_weight`` it re-orders the pool by blended
relevance + facet coverage and reports nDCG@10 / Recall@10 vs the relevance baseline,
overall and on the multi-evidence subset. Only if some weight improves recall is the
answer-side F1 worth measuring.

    uv run python scripts/diag_facet_sweep.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.eval.dataset import index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.store import LanceDBStore
from sage.strategies.facet_coverage import coverage_select, decompose_query, facet_relevance

EVAL_K = 10
WEIGHTS = [0.0, 0.5, 0.7, 0.85, 0.95]


async def main() -> None:
    t0 = time.time()
    ds = build_hetdocqa_dataset(
        "data/hetdocqa/hetdocqa.jsonl", "data/hetdocqa/corpus_manifest.json",
        cache_dir=".cache/hetdoc/docs",
    )
    pools = json.loads(Path(".cache/hetdocqa/mcf_pools.json").read_text())["dev"]
    cache = CallCache(".cache/hetdocqa", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model="openai/gpt-4.1-mini", timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hetdocqa/db", dim=dim)
    await index_passages(store, embedder, ds.corpus)

    ex = {e.qid: e for e in ds.examples}
    multi = {q for q, rel in ds.qrels.items() if len(rel) >= 2}
    all_ids = sorted({cid for rows in pools.values() for (cid, *_rest) in rows})
    rows_by_id = {r.chunk_id: r for r in await store.get_by_ids(all_ids)}

    # Precompute per-query: ids, base scores, facet-relevance matrix (facets cached).
    per_query = {}
    for qid, rows in pools.items():
        ids = [cid for (cid, *_rest) in rows if cid in rows_by_id]
        scores = [float(sc) for (cid, sc, *_rest) in rows if cid in rows_by_id]
        facets = await decompose_query(ex[qid].question, generator)
        femb = np.asarray(await embedder.embed_documents(facets), dtype=np.float64)
        cemb = np.vstack([rows_by_id[c].embedding for c in ids])
        per_query[qid] = (ids, scores, facet_relevance(cemb, femb))
    qrels = {q: ds.qrels[q] for q in pools if q in ds.qrels}
    run_base = {q: {c: s for c, s in zip(ids, sc, strict=True)}
                for q, (ids, sc, _r) in per_query.items()}
    print(f"dev {len(pools)} queries, multi-evidence {len(multi & set(pools))} "
          f"({time.time() - t0:.0f}s)")

    def _eval(run, qids, measure):  # type: ignore[no-untyped-def]
        m = retrieval_metrics_per_query({q: qrels[q] for q in qids}, run, measure)
        return [m[q] for q in qids]

    allq = sorted(qrels)
    mq = [q for q in allq if q in multi]
    for measure in (f"nDCG@{EVAL_K}", f"R@{EVAL_K}"):
        base_all = _eval(run_base, allq, measure)
        base_multi = _eval(run_base, mq, measure)
        print(f"\n== {measure} ==  baseline all={np.mean(base_all):.4f} "
              f"multi={np.mean(base_multi):.4f}")
        print(f"{'lambda':>7} {'all':>9} {'p_all':>7} {'multi':>9} {'p_multi':>8}")
        for w in WEIGHTS:
            run = {}
            for q, (ids, sc, rel) in per_query.items():
                order = coverage_select(ids, rel, sc, k=EVAL_K, relevance_weight=w)
                run[q] = {c: float(len(order) - i) for i, c in enumerate(order)}
            a = _eval(run, allq, measure)
            mm = _eval(run, mq, measure)
            pa = paired_bootstrap_test(a, base_all, seed=0)
            pm = paired_bootstrap_test(mm, base_multi, seed=0)
            print(f"{w:>7.2f} {np.mean(a):>9.4f} {pa:>7.3f} {np.mean(mm):>9.4f} {pm:>8.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
