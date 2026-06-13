"""Modality-aware HyDE: hypothetical documents shaped like each target modality.

Standard HyDE (``sage.strategies.hyde``) writes a single prose answer and embeds it, so
the hypothetical lands near prose chunks and systematically under-retrieves relevant
*code* and *tabular* evidence in a heterogeneous corpus. Modality-aware HyDE instead
generates one hypothetical per content modality -- a code snippet, a table fragment, a
prose passage -- and retrieves with each, unioning the pools before the (shared)
reranker. It changes the candidate pool via the query (upstream of the reranker, where
the headroom is) and exploits corpus heterogeneity, which generic HyDE ignores.
"""

from __future__ import annotations

import asyncio

from sage.core.protocols import Generator

__all__ = ["MODALITY_SYSTEM", "modality_hypotheticals"]

# One generation system prompt per structurally-distinct modality (markdown / pdf read
# as prose). Each asks for a short hypothetical document in that modality's form.
MODALITY_SYSTEM: dict[str, str] = {
    "prose": (
        "Given a question, write a 2-3 sentence factual passage that directly answers "
        "it, as if quoted from the perfect source document. Be specific."
    ),
    "code": (
        "Given a question, write a short, realistic code snippet (with a brief comment) "
        "that implements, configures, or demonstrates the answer. Output only code."
    ),
    "table": (
        "Given a question, write a small markdown/CSV-style table whose header and rows "
        "contain the specific values that answer it. Output only the table."
    ),
}


async def modality_hypotheticals(
    query: str, generator: Generator, *, modalities: tuple[str, ...] = ("prose", "code", "table")
) -> dict[str, str]:
    """Generate one hypothetical document per modality (concurrently; each cached)."""

    async def _one(modality: str) -> tuple[str, str]:
        text = await generator.complete(
            MODALITY_SYSTEM[modality], f"Question: {query}", max_tokens=256
        )
        return modality, text

    pairs = await asyncio.gather(*(_one(m) for m in modalities))
    return {m: t for m, t in pairs if t.strip()}
