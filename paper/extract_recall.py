"""Extract Recall@k curves on HetDocQA dev for the reranker-dominance figure.

All configs use the SAME forced first-stage strategy (semantic) and no correction, so the
curves isolate the *retrieval-stage* components: the cross-encoder reranker vs the
pool-expansion enhancements (RAPTOR, graph, cross-doc). Retrieval-only (no generation);
warm cache makes it fast. Dumps paper/figdata/recall_hetdocqa.json.

    uv run python paper/extract_recall.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.schema import PipelineConfig
from sage.core.types import Strategy, StrategyDecision
from sage.eval.ablate import run_dataset
from sage.eval.dataset import build_raptor_index, index_passages
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.pipeline import build_retrieval_pipeline
from sage.store import LanceDBStore

KS = list(range(1, 31))
CONCURRENCY = 8


class _Forced:
    def __init__(self, s: Strategy) -> None:
        self._s = s

    async def route(self, q: str, v: object, store: object) -> StrategyDecision:
        return StrategyDecision(strategy=self._s)


def _cfg(rerank: bool, raptor: bool, graph: bool, cross_doc: bool) -> PipelineConfig:
    c = full(PipelineConfig())
    c.rerank.enabled = rerank
    c.raptor.enabled = raptor
    c.raptor.cross_doc = cross_doc
    c.graph.enabled = graph
    c.correction.enabled = False          # answer-side; excluded from a retrieval figure
    c.fusion.hyde_expansion = "none"      # strategy is forced to semantic anyway
    return c


CONFIGS = {
    "Dense (bi-encoder)":        _cfg(False, False, False, False),
    "+ Reranker":                _cfg(True, False, False, False),
    "+ Reranker + RAPTOR":       _cfg(True, True, False, True),
    "+ Reranker + Graph":        _cfg(True, False, True, False),
    "+ Reranker + all enh.":     _cfg(True, True, True, True),
}


def recall_at_k(ranking: list[str], gold: set[str]) -> list[float]:
    if not gold:
        return [float("nan")] * len(KS)
    hits = 0
    per_k = {}
    for seen, r in enumerate(ranking, start=1):
        if r in gold:
            hits += 1
        per_k[seen] = hits / len(gold)
    last = 0.0
    out = []
    for k in KS:
        last = per_k.get(k, last)
        out.append(last)
    return out


async def main() -> None:
    ds = build_hetdocqa_dataset(
        "data/hetdocqa/hetdocqa.jsonl", "data/hetdocqa/corpus_manifest.json",
        cache_dir=".cache/hetdoc/docs",
    )
    cache = CallCache(".cache/hetdocqa", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hetdocqa/db", dim=dim)
    await index_passages(store, embedder, ds.corpus)
    await build_raptor_index(store, embedder, None, full(PipelineConfig()).raptor, seed=42)  # cached

    dev = [e for e in ds.examples if e.metadata.get("split") == "dev"]
    sub = type(ds)(name="hetdocqa-dev", examples=dev, corpus=ds.corpus,
                   qrels={e.qid: ds.qrels[e.qid] for e in dev if e.qid in ds.qrels})
    gold = {q: {c for c, g in r.items() if g > 0} for q, r in sub.qrels.items()}

    curves: dict[str, list[float]] = {}
    for name, cfg in CONFIGS.items():
        pipe = build_retrieval_pipeline(cfg, embedder=embedder, store=store, reranker=reranker)
        pipe.router = _Forced(Strategy.SEMANTIC)  # type: ignore[assignment]
        run = dict(await run_dataset(pipe, sub, top_k=max(KS),
                                     semaphore=asyncio.Semaphore(CONCURRENCY)))
        rk = []
        for qid in gold:
            ranking = sorted(run.get(qid, {}), key=lambda c: run[qid][c], reverse=True)
            rk.append(recall_at_k(ranking, gold[qid]))
        curves[name] = np.nanmean(np.array(rk), axis=0).tolist()
        print(f"{name}: R@10={curves[name][9]:.3f}")

    Path("paper/figdata").mkdir(parents=True, exist_ok=True)
    Path("paper/figdata/recall_hetdocqa.json").write_text(
        json.dumps({"ks": KS, "curves": curves})
    )
    print("wrote paper/figdata/recall_hetdocqa.json")


if __name__ == "__main__":
    asyncio.run(main())
