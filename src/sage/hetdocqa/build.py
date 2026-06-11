"""Orchestrate HetDocQA candidate generation across collections.

For each collection and question type, documents are selected, a question is drafted,
the answerability and cross-validation filters run, and finally near-duplicates are
removed and collection-disjoint splits assigned. The result is a candidate pool ready
for a human validation pass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np

from sage.config.seed import rng_for
from sage.core.protocols import Embedder, Generator
from sage.hetdocqa.curate import apply_splits, assign_collection_splits, near_duplicate_mask
from sage.hetdocqa.generate import (
    check_answerable_without_context,
    cross_validate,
    draft_question,
)
from sage.hetdocqa.schema import (
    Collection,
    Modality,
    QuestionCandidate,
    QuestionType,
    SourceDoc,
)

__all__ = ["build_candidates", "select_documents"]


def select_documents(
    docs: Sequence[SourceDoc], qtype: QuestionType, rng: np.random.Generator
) -> list[SourceDoc]:
    """Choose documents appropriate for a question type."""
    code = [d for d in docs if d.modality is Modality.CODE]
    non_code = [d for d in docs if d.modality is not Modality.CODE]
    ordered = list(docs)
    rng.shuffle(ordered)

    if qtype is QuestionType.CODE:
        return code[:1] or ordered[:1]
    if qtype is QuestionType.FACTUAL:
        return non_code[:1] or ordered[:1]
    if qtype in (QuestionType.CROSS_DOCUMENT, QuestionType.MULTI_HOP):
        # Prefer two documents of different modality to force a genuine bridge.
        seen: dict[Modality, SourceDoc] = {}
        for d in ordered:
            seen.setdefault(d.modality, d)
        distinct = list(seen.values())
        return distinct[:2] if len(distinct) >= 2 else ordered[:2]
    if qtype is QuestionType.THEMATIC:
        return ordered[:4]
    return ordered[:1]


async def build_candidates(
    collections: Sequence[Collection],
    docs_by_collection: dict[str, list[SourceDoc]],
    *,
    generator: Generator,
    validator: Generator,
    embedder: Embedder,
    per_type: int = 1,
    dedup_threshold: float = 0.9,
    concurrency: int = 8,
    seed: int = 42,
) -> list[QuestionCandidate]:
    """Generate, filter, dedup, and split a candidate pool (concurrently)."""
    semaphore = asyncio.Semaphore(concurrency)

    async def make_one(
        collection: Collection, qtype: QuestionType, i: int
    ) -> QuestionCandidate | None:
        docs = docs_by_collection.get(collection.collection_id, [])
        if not docs:
            return None
        # Per-task RNG keeps document selection deterministic and race-free.
        rng = rng_for(f"hetdocqa-{collection.collection_id}-{qtype.value}-{i}")
        selected = select_documents(docs, qtype, rng)
        if not selected:
            return None
        qid = f"{collection.collection_id}-{qtype.value}-{i}"
        async with semaphore:
            try:
                candidate = await draft_question(generator, qtype, selected, qid=qid)
                if candidate is None:
                    return None
                await check_answerable_without_context(candidate, validator)
                if candidate.answerable_without_context is False:
                    await cross_validate(candidate, selected, validator)
            except Exception:
                return None
        return candidate

    tasks = [
        make_one(collection, qtype, i)
        for collection in collections
        for qtype in QuestionType
        for i in range(per_type)
    ]
    candidates = [c for c in await asyncio.gather(*tasks) if c is not None]

    if candidates:
        embeddings = await embedder.embed_documents([c.question for c in candidates])
        keep = near_duplicate_mask(embeddings, threshold=dedup_threshold)
        candidates = [c for c, k in zip(candidates, keep, strict=True) if k]

    apply_splits(candidates, assign_collection_splits(collections, seed=seed))
    return candidates
