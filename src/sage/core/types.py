"""Core data types shared across the retrieval pipeline.

These types are deliberately plain (dataclasses, not Pydantic models) because they
flow through hot loops and carry NumPy arrays. Configuration objects, which need
validation and serialization, live in :mod:`sage.config.schema` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

__all__ = [
    "Chunk",
    "Confidence",
    "CorrectionOutcome",
    "DocumentSection",
    "RetrievalTrace",
    "ScoreSource",
    "SearchResult",
    "StoreRow",
    "Strategy",
    "StrategyDecision",
]


class Strategy(StrEnum):
    """A first-stage retrieval strategy the router may select."""

    SEMANTIC = "semantic"
    HYDE = "hyde"
    DPHF = "dphf"
    STEP_BACK = "step_back"


class ScoreSource(StrEnum):
    """Provenance of a result's ``relevance_score``.

    The corrector uses this to apply the correct calibrated threshold: bi-encoder
    cosine scores and cross-encoder rerank scores live on different scales.
    """

    BI_ENCODER = "bi_encoder"
    CROSS_ENCODER = "cross_encoder"
    RRF = "rrf"
    PPR = "ppr"


class Confidence(StrEnum):
    """Retrieval confidence emitted by the correction stage."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A parsed span of a source document, prior to chunking."""

    document_id: str
    text: str
    char_start: int
    char_end: int
    page_number: int | None = None
    section_name: str | None = None
    kind: str = "prose"  # "prose" | "code" | "table" | ...


@dataclass(frozen=True, slots=True)
class Chunk:
    """An indexable unit of text with provenance back to its source document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    char_start: int = 0
    char_end: int = 0
    page_number: int | None = None
    section_name: str | None = None
    language: str | None = None  # set for code chunks

    @property
    def token_estimate(self) -> int:
        """Cheap character-based token estimate (``len // 4``)."""
        return len(self.content) // 4


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk with its score and provenance."""

    chunk_id: str
    document_id: str
    content: str
    relevance_score: float
    chunk_index: int = 0
    level: int = 0  # 0 = leaf chunk; >0 = RAPTOR summary node
    score_source: ScoreSource = ScoreSource.BI_ENCODER
    filename: str | None = None
    page_number: int | None = None
    section_name: str | None = None
    embedding: np.ndarray | None = None

    def with_score(self, score: float, source: ScoreSource) -> SearchResult:
        """Return a copy with a new score and source (results are immutable)."""
        return replace(self, relevance_score=score, score_source=source)


@dataclass(frozen=True, slots=True)
class StoreRow:
    """A row persisted in the vector store (chunk or summary node)."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    embedding: np.ndarray
    level: int = 0
    filename: str | None = None
    page_number: int | None = None
    section_name: str | None = None
    language: str | None = None
    parent_id: str | None = None
    child_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """The router's decision for a single query, plus the signals behind it."""

    strategy: Strategy
    entropy: float | None = None
    knn_distances: np.ndarray | None = None
    is_multi_hop: bool = False
    rationale: str = ""


@dataclass(slots=True)
class CorrectionOutcome:
    """Result of the post-retrieval correction stage."""

    confidence: Confidence
    results: list[SearchResult]
    rewritten_query: str | None = None
    raw_score: float | None = None  # e.g. CRAG 1-100 judge score


@dataclass(slots=True)
class RetrievalTrace:
    """Per-query record of every pipeline stage.

    This is the single source of truth for ablation tables and paper figures:
    each stage appends a structured entry rather than logging to text.
    """

    query: str
    stages: list[dict[str, Any]] = field(default_factory=list)

    def record(self, stage: str, **fields: Any) -> None:
        """Append a stage record."""
        self.stages.append({"stage": stage, **fields})

    def last(self, stage: str) -> dict[str, Any] | None:
        """Return the most recent record for ``stage``, if any."""
        for entry in reversed(self.stages):
            if entry["stage"] == stage:
                return entry
        return None
