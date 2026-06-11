"""Data structures for HetDocQA construction.

A heterogeneous benchmark of questions over mixed-format document *collections*
(PDF, code, markdown, tabular, prose). Each question carries character-span gold
evidence so that retrieval relevance is chunker-agnostic at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from sage.eval.span_mapping import GoldSpan

__all__ = ["Collection", "Modality", "QuestionCandidate", "QuestionType", "SourceDoc"]


class Modality(StrEnum):
    PDF = "pdf"
    CODE = "code"
    MARKDOWN = "markdown"
    TABLE = "table"
    PROSE = "prose"


class QuestionType(StrEnum):
    FACTUAL = "factual"
    CODE = "code"
    CROSS_DOCUMENT = "cross_document"
    MULTI_HOP = "multi_hop"
    THEMATIC = "thematic"


# Target quotas per type (fractions), used to balance generation.
TYPE_QUOTAS: dict[QuestionType, float] = {
    QuestionType.FACTUAL: 0.25,
    QuestionType.CODE: 0.20,
    QuestionType.CROSS_DOCUMENT: 0.20,
    QuestionType.MULTI_HOP: 0.20,
    QuestionType.THEMATIC: 0.15,
}


@dataclass(frozen=True, slots=True)
class SourceDoc:
    """A single source document within a collection."""

    doc_id: str
    collection_id: str
    filename: str
    text: str
    modality: Modality
    source_ref: str  # reproducible pointer: arXiv id, repo@sha:path, dataset url
    license: str


@dataclass(frozen=True, slots=True)
class Collection:
    """A realistic mixed-format workspace (e.g. a project's code + docs + a PDF)."""

    collection_id: str
    title: str
    doc_ids: tuple[str, ...]


@dataclass(slots=True)
class QuestionCandidate:
    """A generated question with provenance and validation state."""

    qid: str
    question: str
    answer: str
    qtype: QuestionType
    collection_id: str
    evidence_doc_ids: list[str]
    gold_spans: list[GoldSpan]
    answerable_without_context: bool | None = None
    validation: dict[str, bool] = field(default_factory=dict)
    split: str | None = None

    @property
    def is_clean(self) -> bool:
        """Passed automated filters: needs context and cross-validation agrees."""
        return (
            self.answerable_without_context is False
            and bool(self.gold_spans)
            and all(self.validation.values())
            and len(self.validation) > 0
        )
