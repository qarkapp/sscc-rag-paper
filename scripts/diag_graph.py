"""Decompose the GAHR graph regression on the HotpotQA slice.

Reuses the cached HotpotQA index/generations and isolates the two graph operations:
  * wo_graph        -- graph off (baseline)
  * graph_no_expand -- GraphSAGE rescoring only (no PPR expansion)
  * wo_graphsage    -- PPR expansion only (no GNN rescore)
  * full            -- both
Prints nDCG@10 so the culprit is visible.

    uv run python scripts/diag_graph.py
"""

from __future__ import annotations

import asyncio

# Reuse the slice loader.
import importlib.util
import time

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config import full
from sage.config.schema import PipelineConfig
from sage.eval.ablate import run_ablations
from sage.eval.dataset import index_passages
from sage.store import LanceDBStore

_spec = importlib.util.spec_from_file_location("slice_mod", "scripts/run_hotpotqa_slice.py")
_slice = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_slice)


async def main() -> None:
    t0 = time.time()
    dataset = _slice.load_slice()
    cache = CallCache(".cache/hotpot", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    generator = ChatGenerator(
        BackendConfig(provider="openrouter", model="qwen/qwen3.6-35b-a3b", timeout=120), cache
    )
    dim = await embedder.probe()
    store = LanceDBStore(".cache/hotpot/db", dim=dim)
    await index_passages(store, embedder, dataset.corpus)

    base = full(PipelineConfig())
    base.raptor.enabled = False
    names = ["wo_graph", "graph_no_expand", "wo_graphsage", "full"]
    outcomes = await run_ablations(
        names, base, dataset,
        embedder=embedder, store=store, generator=generator, reranker=reranker,
        top_k=10, primary_metric="nDCG@10", measures=("nDCG@10", "Success@10", "R@5"),
        concurrency=4,
    )
    print(f"\n=== GAHR decomposition (HotpotQA, qwen) ({time.time() - t0:.0f}s) ===")
    print(f"{'config':18} {'nDCG@10':>9} {'R@5':>8} {'Success@10':>11}")
    for o in outcomes:
        print(f"{o.name:18} {o.metrics.get('nDCG@10', 0):>9.4f} {o.metrics.get('R@5', 0):>8.4f} {o.metrics.get('Success@10', 0):>11.4f}")


if __name__ == "__main__":
    asyncio.run(main())
