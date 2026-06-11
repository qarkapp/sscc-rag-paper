"""Backend clients for oMLX and OpenRouter (OpenAI-compatible)."""

from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.factory import Backends, build_backends
from sage.clients.generator import ChatGenerator
from sage.clients.reranker import RerankClient

__all__ = [
    "BackendConfig",
    "Backends",
    "ChatGenerator",
    "OpenAICompatEmbedder",
    "RerankClient",
    "build_backends",
]
