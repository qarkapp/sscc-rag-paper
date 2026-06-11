"""OpenAI-compatible embedder backed by oMLX, with per-item caching.

Caching is per text, not per batch: adding a single new document re-embeds only
its new chunks. The output dimensionality is discovered at construction time by
embedding a probe string -- the same runtime dimension probing the Rust system
uses to prevent silent vector-store corruption.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sage.cache.keys import call_key
from sage.cache.store import CallCache
from sage.clients.base import BackendConfig, make_async_client
from sage.core.errors import EmbeddingDimensionError

__all__ = ["OpenAICompatEmbedder"]


class OpenAICompatEmbedder:
    """Implements :class:`sage.core.protocols.Embedder`."""

    def __init__(
        self,
        config: BackendConfig,
        cache: CallCache,
        *,
        dim: int | None = None,
    ) -> None:
        self._config = config
        self._cache = cache
        self._client = make_async_client(config)
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise EmbeddingDimensionError(
                "dimension not yet probed; call `await embedder.probe()` first"
            )
        return self._dim

    async def probe(self) -> int:
        """Detect the backend's output dimensionality via a probe embedding."""
        vec = await self._embed_one("dimension probe")
        self._dim = int(vec.shape[0])
        return self._dim

    async def embed_query(self, text: str) -> np.ndarray:
        return await self._embed_one(text, task="query")

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        keys = [self._key(t, "document") for t in texts]
        out: list[np.ndarray | None] = [self._cache.get(k) for k in keys]
        missing = [i for i, v in enumerate(out) if v is None]
        if missing:
            fresh = await self._embed_batch([texts[i] for i in missing])
            for slot, vec in zip(missing, fresh, strict=True):
                out[slot] = vec
                self._cache.put(keys[slot], "embed", self._config.provider, self._config.model, vec)
        result = np.vstack([np.asarray(v, dtype=np.float32) for v in out])
        self._check_dim(result.shape[1])
        return result

    # -- internals ---------------------------------------------------------

    def _key(self, text: str, task: str) -> str:
        return call_key(
            "embed",
            self._config.provider,
            self._config.model,
            {"text": text},
            {"task": task},
        )

    async def _embed_one(self, text: str, task: str = "document") -> np.ndarray:
        key = self._key(text, task)
        cached = self._cache.get(key)
        if cached is not None:
            vec = np.asarray(cached, dtype=np.float32)
        else:
            vec = (await self._embed_batch([text]))[0]
            self._cache.put(key, "embed", self._config.provider, self._config.model, vec)
        self._check_dim(vec.shape[0])
        return vec

    async def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        resp = await self._client.embeddings.create(model=self._config.model, input=texts)
        return [np.asarray(item.embedding, dtype=np.float32) for item in resp.data]

    def _check_dim(self, observed: int) -> None:
        if self._dim is None:
            self._dim = observed
        elif observed != self._dim:
            raise EmbeddingDimensionError(f"expected dim {self._dim}, backend returned {observed}")
