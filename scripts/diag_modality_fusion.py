"""Prototype: does modality-calibrated scoring improve ranking on HetDocQA?

Fit per-(modality, score_source) isotonic calibration on the calibration split (gold
labels), then re-rank the dev candidate pool by calibrated P(relevant) and compare
nDCG@10 / Recall@10 to the uncalibrated baseline. Retrieval-only (no generation), so
it is cheap; if the ranking lift is significant we wire it into the pipeline and run the
answer-side + frozen-test evaluation. Calibration and dev splits are disjoint by
collection, so this is leakage-free.

    uv run python scripts/diag_modality_fusion.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.schema import PipelineConfig
from sage.eval.dataset import index_passages
from sage.eval.metrics import retrieval_metrics_per_query
from sage.eval.stats import paired_bootstrap_test
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore
from sage.strategies.modality_fusion import CalibrationSample, ModalityCalibrator

POOL_K = 50   # candidate pool per query to calibrate + re-rank
EVAL_K = 10   # nDCG@10 / Recall@10
CONCURRENCY = 8


def _modality_map(manifest_path: str) -> dict[str, str]:
    docs = json.loads(Path(manifest_path).read_text())
    return {d["doc_id"]: d["modality"] for d in docs}


def _modality_of(chunk_id: str, by_doc: dict[str, str]) -> str:
    if chunk_id.startswith("__corpus__") or ":L" in chunk_id:
        return "summary"
    return by_doc.get(chunk_id.rsplit(":", 1)[0], "prose")


async def _pool(pipe, dataset, qids):  # type: ignore[no-untyped-def]
    """Return {qid: [SearchResult, ...]} for the top-POOL_K candidates per query."""
    sem = asyncio.Semaphore(CONCURRENCY)
    qtext = {e.qid: e.question for e in dataset.examples}

    async def one(qid: str):  # type: ignore[no-untyped-def]
        async with sem:
            try:
                results, _ = await pipe.run(qtext[qid], top_k=POOL_K)
                return qid, results
            except Exception:
                return qid, []

    return dict(await asyncio.gather(*(one(q) for q in qids)))


async def main() -> None:
    t0 = time.time()
    ds = build_hetdocqa_dataset(
        "data/hetdocqa/hetdocqa.jsonl", "data/hetdocqa/corpus_manifest.json",
        cache_dir=".cache/hetdoc/docs",
    )
    by_doc = _modality_map("data/hetdocqa/corpus_manifest.json")
    cache = CallCache(".cache/hetdocqa", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hetdocqa/db", dim=dim)
    await index_passages(store, embedder, ds.corpus)

    base = full(PipelineConfig())
    base.raptor.enabled = True
    pipe = build_retrieval_pipeline(base, embedder=embedder, store=store, reranker=reranker)

    cal_ids = [e.qid for e in ds.examples if e.metadata.get("split") == "calibration"]
    dev_ids = [e.qid for e in ds.examples if e.metadata.get("split") == "dev"]
    print(f"calibration {len(cal_ids)} / dev {len(dev_ids)} queries; pool={POOL_K}")

    # ---- candidate pools (the expensive part): retrieve once, cache to disk -------
    pool_path = Path(".cache/hetdocqa/mcf_pools.json")

    def _serialize(pool):  # type: ignore[no-untyped-def]
        return {qid: [[r.chunk_id, float(r.relevance_score), str(r.score_source),
                       _modality_of(r.chunk_id, by_doc),
                       1 if r.chunk_id in ds.qrels.get(qid, {}) else 0]
                      for r in results]
                for qid, results in pool.items()}

    if pool_path.exists():
        pools = json.loads(pool_path.read_text())
        print(f"loaded cached pools ({time.time() - t0:.0f}s)")
    else:
        pools = {
            "calibration": _serialize(await _pool(pipe, ds, cal_ids)),
            "dev": _serialize(await _pool(pipe, ds, dev_ids)),
        }
        pool_path.write_text(json.dumps(pools))
        print(f"retrieved + cached pools ({time.time() - t0:.0f}s)")

    samples = [
        CalibrationSample(modality=mod, score_source=src, score=sc, relevant=rel)
        for rows in pools["calibration"].values()
        for (_cid, sc, src, mod, rel) in rows
    ]
    qrels = {q: ds.qrels[q] for q in pools["dev"] if q in ds.qrels}
    run_base = {qid: {cid: sc for (cid, sc, _src, _mod, _rel) in rows}
                for qid, rows in pools["dev"].items()}

    print(f"\n===== modality-calibrated re-ranking on dev ({time.time() - t0:.0f}s) =====")
    for method in ("platt", "isotonic"):
        calibrator = ModalityCalibrator(method=method).fit(samples)
        run_cal = {
            qid: {cid: calibrator.calibrate(sc, mod, src)
                  for (cid, sc, src, mod, _rel) in rows}
            for qid, rows in pools["dev"].items()
        }
        print(f"\n-- method={method}  buckets={len(calibrator.fitted_buckets)} --")
        print(f"{'metric':10} {'baseline':>10} {'calibrated':>12} {'delta':>9} {'p':>8}")
        for measure in (f"nDCG@{EVAL_K}", f"R@{EVAL_K}", "RR@10"):
            b = retrieval_metrics_per_query(qrels, run_base, measure)
            c = retrieval_metrics_per_query(qrels, run_cal, measure)
            qs = sorted(b)
            mb = sum(b.values()) / len(b)
            mc = sum(c.values()) / len(c)
            p = paired_bootstrap_test([c[q] for q in qs], [b[q] for q in qs], seed=0)
            print(f"{measure:10} {mb:>10.4f} {mc:>12.4f} {mc - mb:>+9.4f} {p:>8.3f}")
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
