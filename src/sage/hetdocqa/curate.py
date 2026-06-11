"""Deduplication, splitting, and statistics for the candidate pool."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from sage.config.seed import rng_for
from sage.hetdocqa.schema import Collection, QuestionCandidate, QuestionType

__all__ = ["apply_splits", "assign_collection_splits", "dataset_stats", "near_duplicate_mask"]


def near_duplicate_mask(embeddings: np.ndarray, *, threshold: float = 0.9) -> list[bool]:
    """Return a keep-mask removing later near-duplicate questions (cosine > threshold)."""
    n = embeddings.shape[0]
    if n == 0:
        return []
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and sims[i, j] > threshold:
                keep[j] = False  # drop the later duplicate
    return keep


def assign_collection_splits(
    collections: Sequence[Collection],
    *,
    fractions: tuple[float, float, float] = (0.25, 0.25, 0.5),
    seed: int = 42,
) -> dict[str, str]:
    """Partition whole collections into calibration/dev/test (disjoint by collection).

    Splitting by collection -- not by question -- prevents thresholds tuned on dev
    from exploiting corpus structure shared with test.
    """
    ids = [c.collection_id for c in collections]
    rng_for(f"hetdocqa-split-{seed}").shuffle(ids)
    n = len(ids)
    n_calib = max(1, round(fractions[0] * n)) if n >= 3 else 0
    n_dev = max(1, round(fractions[1] * n)) if n >= 3 else 0
    splits: dict[str, str] = {}
    for k, cid in enumerate(ids):
        if k < n_calib:
            splits[cid] = "calibration"
        elif k < n_calib + n_dev:
            splits[cid] = "dev"
        else:
            splits[cid] = "test"
    return splits


def apply_splits(candidates: Sequence[QuestionCandidate], collection_split: dict[str, str]) -> None:
    """Assign each candidate the split of its collection (in place)."""
    for candidate in candidates:
        candidate.split = collection_split.get(candidate.collection_id, "test")


def dataset_stats(candidates: Sequence[QuestionCandidate]) -> dict[str, object]:
    """Summary statistics for the datasheet."""
    by_type = Counter(c.qtype.value for c in candidates)
    by_split = Counter(c.split or "unassigned" for c in candidates)
    multi_evidence = sum(1 for c in candidates if len(c.gold_spans) >= 2)
    return {
        "total": len(candidates),
        "by_type": dict(by_type),
        "by_split": dict(by_split),
        "type_fractions": {
            t.value: round(by_type.get(t.value, 0) / max(1, len(candidates)), 3)
            for t in QuestionType
        },
        "multi_evidence_questions": multi_evidence,
        "avg_gold_spans": round(
            sum(len(c.gold_spans) for c in candidates) / max(1, len(candidates)), 2
        ),
    }
