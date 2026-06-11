"""Command-line entry point.

Subcommands grow as the pipelines land. ``doctor`` verifies that the configured
backend is reachable and reports the embedder dimension and a sample rerank.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from sage import __version__


def load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a ``.env`` file without overriding existing vars."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


async def _doctor() -> int:
    from sage.cache.store import CacheMode, CallCache
    from sage.clients.base import BackendConfig
    from sage.clients.embedder import OpenAICompatEmbedder
    from sage.clients.reranker import RerankClient

    cache = CallCache(".cache/doctor", CacheMode.OFF)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    dim = await embedder.probe()
    print(f"embedder  bge-m3-mlx-fp16    ok (dim={dim})")

    reranker = RerankClient(BackendConfig(provider="omlx", model="jina-reranker-v3-mlx"), cache)
    ranked = await reranker.rerank(
        "capital of France",
        ["Paris is the capital of France", "Bananas are yellow"],
        top_n=2,
    )
    top_idx, top_score = ranked[0]
    print(f"reranker  jina-reranker-v3   ok (top={top_idx}, score={top_score:.3f})")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="sage", description=__doc__)
    parser.add_argument("--version", action="version", version=f"sage {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="check backend (oMLX) connectivity")
    args = parser.parse_args()

    if args.command == "doctor":
        load_dotenv(Path(".env"))
        raise SystemExit(asyncio.run(_doctor()))
    parser.print_help()


if __name__ == "__main__":
    main()
