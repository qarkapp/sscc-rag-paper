"""Cluster summarization for RAPTOR tree construction."""

from __future__ import annotations

from collections.abc import Sequence

from sage.core.protocols import Generator

__all__ = ["summarize_cluster"]

_LEVEL1_SYSTEM = (
    "You summarize a group of related passages into a single dense summary. "
    "Preserve all key details: entities, relationships, dates, numbers, technical "
    "terms, and specific claims. Be concise and factual."
)
_HIGHER_SYSTEM = (
    "You synthesize a group of summaries into a single coherent overview that "
    "captures the main themes, arguments, and conclusions. Be concise."
)


async def summarize_cluster(
    texts: Sequence[str],
    *,
    level: int,
    generator: Generator,
    context_char_budget: int = 8000,
    max_tokens: int = 512,
) -> str:
    """Summarize the member texts of a cluster into one summary string."""
    system = _LEVEL1_SYSTEM if level == 1 else _HIGHER_SYSTEM
    joined = "\n\n---\n\n".join(texts)
    if len(joined) > context_char_budget:
        joined = joined[:context_char_budget]
    user = f"Summarize the following passages:\n\n{joined}"
    return (await generator.complete(system, user, max_tokens=max_tokens)).strip()
