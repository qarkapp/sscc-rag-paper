"""Prototype: does facet-coverage selection beat top-k relevance on HetDocQA?

Reuses the cached dev candidate pools (scripts/diag_modality_fusion.py wrote them),
decomposes each query into facets, and re-orders the pool by greedy facet coverage. It
compares retrieval (nDCG@10 / Recall@10) AND answer F1 against the relevance baseline,
overall and on the multi-evidence subset (>=2 gold spans) where coverage should matter
most. Leakage-free: no fitting, decomposition is query-only.

    uv run python scripts/diag_facet_coverage.py
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
from sage.core.types import SearchResult
from sage.eval.answer import synthesize_answer
from sage.eval.dataset import index_passages
from sage.eval.metrics import retrieval_metrics_per_query, token_f1
from sage.eval.stats import paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.store import LanceDBStore
from sage.strategies.facet_coverage import coverage_select, decompose_query, facet_relevance

EVAL_K = 10
ANSWER_K = 5
CONCURRENCY = 8


def _f1(answers: tuple[str, ...], prediction: str) -> float:
    return max((token_f1(prediction, g) for g in answers), default=0.0)


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
    multi = {q for q, rel in ds.qrels.items() if len(rel) >= 2}  # multi-evidence queries
    # Fetch embeddings + content for every pooled chunk once.
    all_ids = sorted({cid for rows in pools.values() for (cid, *_rest) in rows})
    store_rows = {r.chunk_id: r for r in await store.get_by_ids(all_ids)}
    print(f"dev {len(pools)} queries, {len(all_ids)} pooled chunks ({time.time() - t0:.0f}s)")

    sem = asyncio.Semaphore(CONCURRENCY)
    run_base: dict[str, dict[str, float]] = {}
    run_fcs: dict[str, dict[str, float]] = {}
    f1_base: dict[str, float] = {}
    f1_fcs: dict[str, float] = {}
    n_facets: list[int] = []

    async def one(qid: str, rows: list) -> None:  # type: ignore[type-arg]
        async with sem:
            ids = [cid for (cid, *_rest) in rows if cid in store_rows]
            scores = {cid: float(sc) for (cid, sc, *_rest) in rows}
            facets = await decompose_query(ex[qid].question, generator)
            n_facets.append(len(facets))
            facet_emb = np.asarray(await embedder.embed_documents(facets), dtype=np.float64)
            chunk_emb = np.vstack([store_rows[c].embedding for c in ids])
            rel = facet_relevance(chunk_emb, facet_emb)
            order = coverage_select(ids, rel, [scores[c] for c in ids], k=EVAL_K)

            run_base[qid] = scores
            run_fcs[qid] = {cid: float(len(order) - i) for i, cid in enumerate(order)}
            base_top = [store_rows[c] for c in sorted(ids, key=lambda c: scores[c], reverse=True)]
            fcs_top = [store_rows[c] for c in order]

            def _sr(rows_):  # type: ignore[no-untyped-def]
                return [SearchResult(chunk_id=r.chunk_id, document_id=r.document_id,
                                     content=r.content, relevance_score=1.0,
                                     chunk_index=r.chunk_index, level=r.level) for r in rows_]

            ab = await synthesize_answer(ex[qid].question, _sr(base_top), generator, top_k=ANSWER_K)
            af = await synthesize_answer(ex[qid].question, _sr(fcs_top), generator, top_k=ANSWER_K)
            f1_base[qid] = _f1(ex[qid].answers, ab)
            f1_fcs[qid] = _f1(ex[qid].answers, af)

    await asyncio.gather(*(one(q, rows) for q, rows in pools.items()))
    qrels = {q: ds.qrels[q] for q in pools if q in ds.qrels}
    print(f"avg facets/query: {np.mean(n_facets):.2f}; multi-evidence dev queries: {len(multi & set(pools))}")

    def _report(name: str, qids: list[str]) -> None:
        print(f"\n-- {name} ({len(qids)} queries) --")
        print(f"{'metric':10} {'baseline':>10} {'facet-cov':>11} {'delta':>9} {'p':>8}")
        for measure in (f"nDCG@{EVAL_K}", f"R@{EVAL_K}"):
            b = retrieval_metrics_per_query({q: qrels[q] for q in qids}, run_base, measure)
            c = retrieval_metrics_per_query({q: qrels[q] for q in qids}, run_fcs, measure)
            mb, mc = np.mean([b[q] for q in qids]), np.mean([c[q] for q in qids])
            p = paired_bootstrap_test([c[q] for q in qids], [b[q] for q in qids], seed=0)
            print(f"{measure:10} {mb:>10.4f} {mc:>11.4f} {mc - mb:>+9.4f} {p:>8.3f}")
        mb = np.mean([f1_base[q] for q in qids])
        mc = np.mean([f1_fcs[q] for q in qids])
        p = paired_bootstrap_test([f1_fcs[q] for q in qids], [f1_base[q] for q in qids], seed=0)
        print(f"{'F1':10} {mb:>10.4f} {mc:>11.4f} {mc - mb:>+9.4f} {p:>8.3f}")

    allq = sorted(qrels)
    _report("all dev", allq)
    _report("multi-evidence subset", [q for q in allq if q in multi])
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
