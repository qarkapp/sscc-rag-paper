"""HetDocQA: construction toolkit for a heterogeneous multi-format retrieval benchmark."""

from sage.hetdocqa.curate import (
    apply_splits,
    assign_collection_splits,
    dataset_stats,
    near_duplicate_mask,
)
from sage.hetdocqa.generate import (
    check_answerable_without_context,
    cross_validate,
    draft_question,
)
from sage.hetdocqa.release import build_datasheet, to_retrieval_dataset, write_release
from sage.hetdocqa.schema import (
    Collection,
    Modality,
    QuestionCandidate,
    QuestionType,
    SourceDoc,
)
from sage.hetdocqa.spans import locate_span, snippets_to_spans

__all__ = [
    "Collection",
    "Modality",
    "QuestionCandidate",
    "QuestionType",
    "SourceDoc",
    "apply_splits",
    "assign_collection_splits",
    "build_datasheet",
    "check_answerable_without_context",
    "cross_validate",
    "dataset_stats",
    "draft_question",
    "locate_span",
    "near_duplicate_mask",
    "snippets_to_spans",
    "to_retrieval_dataset",
    "write_release",
]
