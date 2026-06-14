"""Prototype: does modality-aware HyDE retrieve evidence generic HyDE misses?

Both arms search with the query and a prose hypothetical, union, and rerank with the
same cross-encoder. The treatment arm additionally searches with a *code* and a *table*
hypothetical. The only difference is the modality-typed hypotheticals, so any gain is
attributable to them. Reports nDCG@10 / Recall@10 / answer-F1 on HetDocQA dev, overall
and on code+table-evidence questions where the effect should concentrate.

    uv run python scripts/diag_modality_hyde.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.core.types import SearchResult
from sage.eval.answer import synthesize_answer
from sage.eval.dataset import index_passages
from sage.eval.metrics import retrieval_metrics_per_query, token_f1
from sage.eval.stats import paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.store import LanceDBStore
from sage.strategies.modality_hyde import modality_hypotheticals

FETCH = 20       # per-path retrieval depth
RERANK_N = 20    # rerank the union, keep this many
EVAL_K = 10
ANSWER_K = 5
CONCURRENCY = 6


def _f1(answers: tuple[str, ...], pred: str) -> float:
    return max((token_f1(pred, g) for g in answers), default=0.0)


async def main() -> None:
    t0 = time.time()
    ds = build_hetdocqa_dataset(
        "data/hetdocqa/hetdocqa.jsonl", "data/hetdocqa/corpus_manifest.json",
        cache_dir=".cache/hetdoc/docs",
    )
    by_doc = {d["doc_id"]: d["modality"]
              for d in json.loads(Path("data/hetdocqa/corpus_manifest.json").read_text())}
    cache = CallCache(".cache/hetdocqa", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model="openai/gpt-4.1-mini", timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hetdocqa/db", dim=dim)
    await index_passages(store, embedder, ds.corpus)

    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    ex = {e.qid: e for e in ds.examples}
    dev = [e.qid for e in ds.examples if e.metadata.get("split") == split]
    # Questions whose gold evidence includes a code or table chunk (modality from manifest).
    def _has_ct(qid: str) -> bool:
        for cid in ds.qrels.get(qid, {}):
            if by_doc.get(cid.rsplit(":", 1)[0]) in {"code", "table"}:
                return True
        return False
    ct = {q for q in dev if _has_ct(q)}
    print(f"split={split}: {len(dev)} queries; code/table-evidence {len(ct)} ({time.time() - t0:.0f}s)")

    sem = asyncio.Semaphore(CONCURRENCY)
    run_base: dict[str, dict[str, float]] = {}
    run_ma: dict[str, dict[str, float]] = {}
    f1_base: dict[str, float] = {}
    f1_ma: dict[str, float] = {}

    async def _search(vec) -> list[SearchResult]:  # type: ignore[no-untyped-def]
        return await store.search(vec, FETCH)

    async def _reranked(qid: str, pool: dict[str, SearchResult]) -> dict[str, float]:
        ids = list(pool)
        ranked = await reranker.rerank(ex[qid].question, [pool[i].content for i in ids], RERANK_N)
        return {ids[idx]: float(score) for idx, score in ranked}

    async def one(qid: str) -> None:
        async with sem:
            q = ex[qid].question
            qvec = await embedder.embed_query(q)
            hypos = await modality_hypotheticals(q, generator)  # prose, code, table
            paths = {"query": qvec}
            for mod, text in hypos.items():
                paths[mod] = await embedder.embed_query(text)

            async def pool_for(keys: list[str]) -> dict[str, SearchResult]:
                pool: dict[str, SearchResult] = {}
                for k in keys:
                    if k not in paths:
                        continue
                    for r in await _search(paths[k]):
                        pool.setdefault(r.chunk_id, r)
                return pool

            base_pool = await pool_for(["query", "prose"])
            ma_pool = await pool_for(["query", "prose", "code", "table"])
            run_base[qid] = await _reranked(qid, base_pool)
            run_ma[qid] = await _reranked(qid, ma_pool)

            def _top(run: dict[str, float], pool: dict[str, SearchResult]) -> list[SearchResult]:
                top = sorted(run, key=lambda c: run[c], reverse=True)[:ANSWER_K]
                return [pool[c] for c in top]

            ab = await synthesize_answer(q, _top(run_base[qid], base_pool), generator, top_k=ANSWER_K)
            af = await synthesize_answer(q, _top(run_ma[qid], ma_pool), generator, top_k=ANSWER_K)
            f1_base[qid] = _f1(ex[qid].answers, ab)
            f1_ma[qid] = _f1(ex[qid].answers, af)

    await asyncio.gather(*(one(q) for q in dev))
    qrels = {q: ds.qrels[q] for q in dev if q in ds.qrels}

    def _report(name: str, qids: list[str]) -> None:
        print(f"\n-- {name} ({len(qids)} queries) --")
        print(f"{'metric':10} {'generic':>10} {'modality':>10} {'delta':>9} {'p':>8}")
        for measure in (f"nDCG@{EVAL_K}", f"R@{EVAL_K}"):
            b = retrieval_metrics_per_query({q: qrels[q] for q in qids}, run_base, measure)
            c = retrieval_metrics_per_query({q: qrels[q] for q in qids}, run_ma, measure)
            mb, mc = np.mean([b[q] for q in qids]), np.mean([c[q] for q in qids])
            p = paired_bootstrap_test([c[q] for q in qids], [b[q] for q in qids], seed=0)
            print(f"{measure:10} {mb:>10.4f} {mc:>10.4f} {mc - mb:>+9.4f} {p:>8.3f}")
        mb = np.mean([f1_base[q] for q in qids])
        mc = np.mean([f1_ma[q] for q in qids])
        p = paired_bootstrap_test([f1_ma[q] for q in qids], [f1_base[q] for q in qids], seed=0)
        print(f"{'F1':10} {mb:>10.4f} {mc:>10.4f} {mc - mb:>+9.4f} {p:>8.3f}")

    _report("all dev", sorted(qrels))
    _report("code/table-evidence", [q for q in sorted(qrels) if q in ct])
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
