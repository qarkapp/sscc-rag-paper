"""Shared backend client configuration and helpers.

The OpenAI SDK (and the raw ``/rerank`` HTTP call) are confined to this package.
Everything else in ``sage`` depends only on the protocols in
:mod:`sage.core.protocols`, which keeps the call-cache and offline replay simple.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sage.core.errors import BackendError

__all__ = ["BackendConfig", "make_async_client", "with_retries"]

# Default endpoints. oMLX is OpenAI-compatible and runs locally; OpenRouter is
# used only when a stronger or closed model is warranted.
_DEFAULT_BASE_URLS = {
    "omlx": "http://127.0.0.1:1234/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
_DEFAULT_KEY_ENV = {
    "omlx": "OMLX_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Connection settings for an OpenAI-compatible backend."""

    provider: str = "omlx"
    model: str = "bge-m3"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0

    def resolved_base_url(self) -> str:
        return self.base_url or _DEFAULT_BASE_URLS.get(self.provider, _DEFAULT_BASE_URLS["openai"])

    def resolved_api_key(self) -> str:
        env = self.api_key_env or _DEFAULT_KEY_ENV.get(self.provider, "OPENAI_API_KEY")
        # Local servers still require a non-empty key; fall back to a placeholder
        # only for providers that explicitly accept one.
        return os.environ.get(env, "")


def make_async_client(config: BackendConfig) -> AsyncOpenAI:
    """Build an :class:`AsyncOpenAI` client for the given backend."""
    key = config.resolved_api_key()
    if not key:
        env = config.api_key_env or _DEFAULT_KEY_ENV.get(config.provider, "OPENAI_API_KEY")
        raise BackendError(
            f"No API key for provider {config.provider!r}: set ${env}. "
            "oMLX requires a key even though it runs locally."
        )
    return AsyncOpenAI(
        base_url=config.resolved_base_url(),
        api_key=key,
        timeout=config.timeout,
    )


def with_retries():  # type: ignore[no-untyped-def]
    """Decorator: retry transient backend errors with exponential backoff."""
    return retry(
        retry=retry_if_exception_type((BackendError, ConnectionError, TimeoutError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
