"""Chat-completion generator (oMLX local or OpenRouter).

Completions are cached by (system, user, params). Streaming is used by speculative
prefetch; streamed responses are also cached so replay is deterministic regardless
of provider nondeterminism.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sage.cache.keys import call_key
from sage.cache.store import CallCache
from sage.clients.base import BackendConfig, make_async_client

__all__ = ["ChatGenerator"]


class ChatGenerator:
    """Implements :class:`sage.core.protocols.Generator`."""

    def __init__(self, config: BackendConfig, cache: CallCache) -> None:
        self._config = config
        self._cache = cache
        self._client = make_async_client(config)

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        key = self._key(system, user, max_tokens, temperature)
        cached = self._cache.get(key)
        if cached is not None:
            return str(cached)

        resp = await self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=self._config.extra_body or None,
        )
        text = resp.choices[0].message.content or ""
        self._cache.put(key, "generate", self._config.provider, self._config.model, text)
        return text

    async def stream(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Yield tokens; on a cache hit, replay the cached text as one chunk."""
        key = self._key(system, user, max_tokens, temperature)
        cached = self._cache.get(key)
        if cached is not None:
            yield str(cached)
            return

        chunks: list[str] = []
        stream = await self._client.chat.completions.create(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body=self._config.extra_body or None,
        )
        async for event in stream:
            delta = event.choices[0].delta.content or ""
            if delta:
                chunks.append(delta)
                yield delta
        self._cache.put(key, "generate", self._config.provider, self._config.model, "".join(chunks))

    def _key(self, system: str, user: str, max_tokens: int, temperature: float) -> str:
        return call_key(
            "generate",
            self._config.provider,
            self._config.model,
            {"system": system, "user": user},
            {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "extra_body": self._config.extra_body or {},
            },
        )
