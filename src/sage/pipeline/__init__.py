"""Indexing and retrieval pipeline assembly."""

from sage.pipeline.assembly import build_retrieval_pipeline
from sage.pipeline.indexing import Document, IndexingPipeline
from sage.pipeline.retrieval import RetrievalPipeline

__all__ = [
    "Document",
    "IndexingPipeline",
    "RetrievalPipeline",
    "build_retrieval_pipeline",
]
