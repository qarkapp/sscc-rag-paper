"""Extract bi- vs cross-encoder relevance-score distributions for the SSCC figure.

SSCC keeps a separate confidence threshold per score source because the bi-encoder
(1/(1+L2)) and cross-encoder (reranker) scores live on different scales. For HetDocQA
calibration+dev queries we record, per retrieved passage, its bi-encoder score and its
cross-encoder score together with whether it is gold. Dumps paper/figdata/sscc.json.

    uv run python paper/extract_sscc.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.reranker import RerankClient
from sage.eval.dataset import index_passages
from sage.hetdocqa.eval_loader import build_hetdocqa_dataset
from sage.store import LanceDBStore

FETCH = 30
CONCURRENCY = 8


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

    ex = [e for e in ds.examples if e.metadata.get("split") in {"calibration", "dev"}]
    gold = {e.qid: {c for c, g in ds.qrels.get(e.qid, {}).items() if g > 0} for e in ex}
    sem = asyncio.Semaphore(CONCURRENCY)
    bi: list[tuple[float, int]] = []
    cross: list[tuple[float, int]] = []

    async def one(e) -> None:  # type: ignore[no-untyped-def]
        async with sem:
            qv = await embedder.embed_query(e.question)
            hits = await store.search(qv, FETCH)
            g = gold.get(e.qid, set())
            for h in hits:
                bi.append((float(h.relevance_score), int(h.chunk_id in g)))
            ranked = await reranker.rerank(e.question, [h.content for h in hits], FETCH)
            for idx, score in ranked:
                cross.append((float(score), int(hits[idx].chunk_id in g)))

    await asyncio.gather(*(one(e) for e in ex))
    Path("paper/figdata").mkdir(parents=True, exist_ok=True)
    Path("paper/figdata/sscc.json").write_text(json.dumps({"bi": bi, "cross": cross}))
    print(f"bi {len(bi)} (gold {sum(g for _, g in bi)}), "
          f"cross {len(cross)} (gold {sum(g for _, g in cross)}) -> paper/figdata/sscc.json")


if __name__ == "__main__":
    asyncio.run(main())
