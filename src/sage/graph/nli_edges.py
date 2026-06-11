"""Directed entailment edges between chunks.

Within the semantic neighbourhood (cosine above a gate), a natural-language-inference
classifier labels ordered chunk pairs as entailment, elaboration, or contradiction.
These directed, typed edges support entailment-chain traversal for multi-hop queries.
The classifier is injected (an oMLX/cross-encoder NLI model in production, a fake in
tests), so this module needs no model dependency.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from sage.core.types import Chunk

__all__ = ["EntailmentEdge", "EntailmentLabel", "NliClassifier", "build_entailment_edges"]


class EntailmentLabel(StrEnum):
    ENTAILMENT = "entailment"
    ELABORATION = "elaboration"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


@dataclass(frozen=True, slots=True)
class EntailmentEdge:
    src: str
    dst: str
    label: EntailmentLabel
    confidence: float


# (premise, hypothesis) -> (label, confidence)
NliClassifier = Callable[[str, str], tuple[EntailmentLabel, float]]


def build_entailment_edges(
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    classify: NliClassifier,
    *,
    cos_gate: float = 0.5,
    max_pairs_per_chunk: int = 10,
) -> list[EntailmentEdge]:
    """Build directed entailment edges among semantically-near chunk pairs."""
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    edges: list[EntailmentEdge] = []
    keep = {EntailmentLabel.ENTAILMENT, EntailmentLabel.ELABORATION, EntailmentLabel.CONTRADICTION}

    for i in range(len(chunks)):
        neighbours = np.argsort(-sims[i])
        considered = 0
        for j in neighbours:
            if i == j or sims[i, j] < cos_gate:
                continue
            label, confidence = classify(chunks[i].content, chunks[int(j)].content)
            if label in keep:
                edges.append(
                    EntailmentEdge(chunks[i].chunk_id, chunks[int(j)].chunk_id, label, confidence)
                )
            considered += 1
            if considered >= max_pairs_per_chunk:
                break
    return edges
