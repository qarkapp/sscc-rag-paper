"""Cross-encoder reranker over an OpenAI-style ``/rerank`` endpoint.

The ``/rerank`` route is not part of the OpenAI schema, so this uses a raw HTTP
call (the Cohere/Jina-compatible body that oMLX speaks). Scores are cached keyed
by the query and the content of the candidate documents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import httpx

from sage.cache.keys import call_key
from sage.cache.store import CallCache
from sage.clients.base import BackendConfig
from sage.core.errors import BackendError

__all__ = ["RerankClient"]


class RerankClient:
    """Implements :class:`sage.core.protocols.Reranker`."""

    def __init__(self, config: BackendConfig, cache: CallCache) -> None:
        self._config = config
        self._cache = cache

    async def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        key = self._key(query, documents, top_n)
        cached = self._cache.get(key)
        if cached is not None:
            return [(int(i), float(s)) for i, s in cached]

        scored = await self._call(query, list(documents), top_n)
        self._cache.put(key, "rerank", self._config.provider, self._config.model, scored)
        return [(int(i), float(s)) for i, s in scored]

    # -- internals ---------------------------------------------------------

    def _key(self, query: str, documents: Sequence[str], top_n: int) -> str:
        doc_hash = hashlib.sha256("".join(documents).encode()).hexdigest()
        return call_key(
            "rerank",
            self._config.provider,
            self._config.model,
            {"query": query, "docs": doc_hash, "n": len(documents)},
            {"top_n": top_n},
        )

    async def _call(self, query: str, documents: list[str], top_n: int) -> list[list[float]]:
        url = self._config.resolved_base_url().rstrip("/") + "/rerank"
        headers = {"Authorization": f"Bearer {self._config.resolved_api_key()}"}
        body = {
            "model": self._config.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
            "return_documents": False,
        }
        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            resp = await client.post(url, json=body, headers=headers)
        if resp.status_code != 200:
            raise BackendError(f"rerank failed {resp.status_code}: {resp.text[:200]}")
        results = resp.json().get("results", [])
        return [[int(r["index"]), float(r["relevance_score"])] for r in results]
