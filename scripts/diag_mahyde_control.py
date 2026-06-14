"""Decisive control: is modality-aware HyDE's gain MODALITY or just more hypotheticals?

Three arms, identical except the hypothetical set, each unioned with the query path and
reranked by the same cross-encoder:
  * generic     -- query + 1 prose hypothetical
  * multi_prose -- query + 3 diverse prose hypotheticals  (same count as modality)
  * modality    -- query + prose + code + table hypotheticals

If modality > multi_prose on the code/table-evidence questions, the gain is the modality
typing, not the ensemble. Runs on the frozen-test code/table subgroup by default.

    uv run python scripts/diag_mahyde_control.py            # code/table test subgroup
    uv run python scripts/diag_mahyde_control.py all test   # all test queries
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
from sage.strategies.hyde import HYDE_SYSTEM
from sage.strategies.modality_hyde import modality_hypotheticals, prose_hypotheticals

FETCH, RERANK_N, EVAL_K, ANSWER_K, CONCURRENCY = 20, 20, 10, 5, 6


def _f1(answers: tuple[str, ...], pred: str) -> float:
    return max((token_f1(pred, g) for g in answers), default=0.0)


async def main() -> None:
    t0 = time.time()
    subset = sys.argv[1] if len(sys.argv) > 1 else "ct"   # "ct" | "all"
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
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

    ex = {e.qid: e for e in ds.examples}

    def _has_ct(qid: str) -> bool:
        return any(by_doc.get(c.rsplit(":", 1)[0]) in {"code", "table"}
                   for c in ds.qrels.get(qid, {}))

    qids = [e.qid for e in ds.examples if e.metadata.get("split") == split
            and (subset == "all" or _has_ct(e.qid))]
    print(f"split={split} subset={subset}: {len(qids)} queries ({time.time() - t0:.0f}s)")

    arms = ("generic", "multi_prose", "modality")
    runs: dict[str, dict[str, dict[str, float]]] = {a: {} for a in arms}
    f1: dict[str, dict[str, float]] = {a: {} for a in arms}
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _rr(qid: str, pool: dict[str, SearchResult]) -> dict[str, float]:
        ids = list(pool)
        ranked = await reranker.rerank(ex[qid].question, [pool[i].content for i in ids], RERANK_N)
        return {ids[i]: float(s) for i, s in ranked}

    async def one(qid: str) -> None:
        async with sem:
            q = ex[qid].question
            qvec = await embedder.embed_query(q)
            generic = [await generator.complete(HYDE_SYSTEM, f"Question: {q}", max_tokens=256)]
            mprose = await prose_hypotheticals(q, generator, n=3)
            modal = list((await modality_hypotheticals(q, generator)).values())
            texts = {"generic": generic, "multi_prose": mprose, "modality": modal}

            async def pool_for(hypos: list[str]) -> dict[str, SearchResult]:
                pool: dict[str, SearchResult] = {}
                for r in await store.search(qvec, FETCH):
                    pool.setdefault(r.chunk_id, r)
                for h in hypos:
                    for r in await store.search(await embedder.embed_query(h), FETCH):
                        pool.setdefault(r.chunk_id, r)
                return pool

            for arm in arms:
                pool = await pool_for(texts[arm])
                run = await _rr(qid, pool)
                runs[arm][qid] = run
                top = [pool[c] for c in sorted(run, key=lambda c: run[c], reverse=True)[:ANSWER_K]]
                sr = [SearchResult(chunk_id=r.chunk_id, document_id=r.document_id,
                                   content=r.content, relevance_score=1.0,
                                   chunk_index=r.chunk_index, level=r.level) for r in top]
                f1[arm][qid] = _f1(ex[qid].answers, await synthesize_answer(q, sr, generator, top_k=ANSWER_K))

    await asyncio.gather(*(one(q) for q in qids))
    qrels = {q: ds.qrels[q] for q in qids if q in ds.qrels}

    print(f"\n{'arm':12} {'nDCG@10':>9} {'R@10':>8} {'F1':>8}")
    means = {}
    for arm in arms:
        nd = np.mean(list(retrieval_metrics_per_query(qrels, runs[arm], "nDCG@10").values()))
        rc = np.mean(list(retrieval_metrics_per_query(qrels, runs[arm], "R@10").values()))
        ff = np.mean([f1[arm][q] for q in qids])
        means[arm] = (nd, rc, ff)
        print(f"{arm:12} {nd:>9.4f} {rc:>8.4f} {ff:>8.4f}")

    print("\npaired bootstrap (the decisive test = modality vs multi_prose):")
    for a, b in [("modality", "generic"), ("modality", "multi_prose"), ("multi_prose", "generic")]:
        for metric, run in [("nDCG@10", runs), ("F1", None)]:
            if metric == "F1":
                xa, xb = [f1[a][q] for q in qids], [f1[b][q] for q in qids]
            else:
                ma = retrieval_metrics_per_query(qrels, runs[a], metric)
                mb = retrieval_metrics_per_query(qrels, runs[b], metric)
                xa, xb = [ma[q] for q in qids], [mb[q] for q in qids]
            p = paired_bootstrap_test(xa, xb, seed=0)
            print(f"  {a:11} vs {b:11} {metric:8} delta={np.mean(xa) - np.mean(xb):+.4f} p={p:.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
