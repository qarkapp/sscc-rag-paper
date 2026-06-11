"""Assemble a retrieval pipeline from configuration.

This is the single place that maps a :class:`PipelineConfig` to concrete components.
Routers and correctors are resolved through the registry by their configured
variant, so enabling a different one is a configuration change. Backends (embedder,
store, generator, reranker) are constructed elsewhere and injected here.
"""

from __future__ import annotations

from sage.config.schema import PipelineConfig
from sage.core import registry
from sage.core.protocols import (
    Embedder,
    Generator,
    Reranker,
    RetrievalStrategy,
    VectorStore,
)
from sage.core.types import Strategy
from sage.pipeline.retrieval import RetrievalPipeline

__all__ = ["build_retrieval_pipeline"]


def _ensure_components_registered() -> None:
    # Importing these modules triggers their @register decorators.
    import sage.correction.crag
    import sage.routing.keyword
    import sage.strategies.semantic  # noqa: F401


def build_retrieval_pipeline(
    config: PipelineConfig,
    *,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
) -> RetrievalPipeline:
    """Construct a :class:`RetrievalPipeline` for ``config``."""
    _ensure_components_registered()

    from sage.strategies.hyde import HydeStrategy
    from sage.strategies.semantic import SemanticStrategy
    from sage.strategies.step_back import StepBackStrategy

    router = registry.get("router", config.router.variant)(config.router)
    strategies: dict[Strategy, RetrievalStrategy] = {
        Strategy.SEMANTIC: SemanticStrategy(),
        Strategy.HYDE: HydeStrategy(config.fusion),
        Strategy.STEP_BACK: StepBackStrategy(),
    }
    corrector = None
    if config.correction.enabled:
        corrector = registry.get("corrector", config.correction.variant)(config.correction)

    return RetrievalPipeline(
        config=config,
        embedder=embedder,
        store=store,
        router=router,
        strategies=strategies,
        reranker=reranker if config.rerank.enabled else None,
        corrector=corrector,
        generator=generator,
    )
