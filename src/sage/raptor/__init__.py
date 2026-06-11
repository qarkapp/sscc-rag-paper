"""RAPTOR: hierarchical clustering, tree construction, retrieval, cross-document tier."""

from sage.raptor.clustering import cluster_bic, hierarchical_cluster, reduce_dimensions
from sage.raptor.cross_doc import (
    CORPUS_DOCUMENT_ID,
    build_cross_document_tier,
    membership_hash,
)
from sage.raptor.retrieval import raptor_retrieve, select_within_budget
from sage.raptor.summarize import summarize_cluster
from sage.raptor.tree import build_tree

__all__ = [
    "CORPUS_DOCUMENT_ID",
    "build_cross_document_tier",
    "build_tree",
    "cluster_bic",
    "hierarchical_cluster",
    "membership_hash",
    "raptor_retrieve",
    "reduce_dimensions",
    "select_within_budget",
    "summarize_cluster",
]
