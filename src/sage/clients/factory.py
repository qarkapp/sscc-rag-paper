"""Construct backend clients from a pipeline configuration."""

from __future__ import annotations

from sage.cache.store import CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient
from sage.config.schema import BackendCfg, PipelineConfig

__all__ = ["Backends", "build_backends"]


def _backend_config(cfg: BackendCfg) -> BackendConfig:
    return BackendConfig(
        provider=cfg.provider,
        model=cfg.model,
        base_url=cfg.base_url,
        api_key_env=cfg.api_key_env,
        timeout=cfg.timeout,
    )


class Backends:
    """Constructed embedder, generator, and reranker sharing one cache."""

    def __init__(
        self,
        embedder: OpenAICompatEmbedder,
        generator: ChatGenerator,
        reranker: RerankClient,
    ) -> None:
        self.embedder = embedder
        self.generator = generator
        self.reranker = reranker


def build_backends(config: PipelineConfig, cache: CallCache) -> Backends:
    """Build oMLX/OpenRouter clients for the given configuration."""
    return Backends(
        embedder=OpenAICompatEmbedder(_backend_config(config.embedder), cache),
        generator=ChatGenerator(_backend_config(config.generator), cache),
        reranker=RerankClient(_backend_config(config.reranker), cache),
    )
