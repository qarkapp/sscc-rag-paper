"""Benchmark loaders, metrics, span mapping, and the evaluation runner."""

from sage.eval.dataset import (
    QAExample,
    RetrievalDataset,
    index_passages,
    load_beir,
    load_jsonl,
)
from sage.eval.metrics import DEFAULT_MEASURES, exact_match, retrieval_metrics, token_f1
from sage.eval.runner import EvalResult, evaluate_retrieval
from sage.eval.span_mapping import ChunkSpan, GoldSpan, build_qrels, relevant_chunk_ids

__all__ = [
    "DEFAULT_MEASURES",
    "ChunkSpan",
    "EvalResult",
    "GoldSpan",
    "QAExample",
    "RetrievalDataset",
    "build_qrels",
    "evaluate_retrieval",
    "exact_match",
    "index_passages",
    "load_beir",
    "load_jsonl",
    "relevant_chunk_ids",
    "retrieval_metrics",
    "token_f1",
]
