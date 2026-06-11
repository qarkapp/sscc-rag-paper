"""Benchmark loaders, metrics, span mapping, statistics, and experiment runners."""

from sage.eval.ablate import (
    AblationOutcome,
    Comparison,
    compare_to_reference,
    run_ablations,
    run_dataset,
)
from sage.eval.dataset import (
    QAExample,
    RetrievalDataset,
    index_passages,
    load_beir,
    load_jsonl,
)
from sage.eval.metrics import (
    DEFAULT_MEASURES,
    exact_match,
    retrieval_metrics,
    retrieval_metrics_per_query,
    token_f1,
)
from sage.eval.runner import EvalResult, evaluate_retrieval
from sage.eval.span_mapping import ChunkSpan, GoldSpan, build_qrels, relevant_chunk_ids
from sage.eval.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    holm_bonferroni,
    paired_bootstrap_test,
    paired_diff_ci,
)

__all__ = [
    "DEFAULT_MEASURES",
    "AblationOutcome",
    "ChunkSpan",
    "Comparison",
    "EvalResult",
    "GoldSpan",
    "QAExample",
    "RetrievalDataset",
    "benjamini_hochberg",
    "bootstrap_ci",
    "build_qrels",
    "compare_to_reference",
    "evaluate_retrieval",
    "exact_match",
    "holm_bonferroni",
    "index_passages",
    "load_beir",
    "load_jsonl",
    "paired_bootstrap_test",
    "paired_diff_ci",
    "relevant_chunk_ids",
    "retrieval_metrics",
    "retrieval_metrics_per_query",
    "run_ablations",
    "run_dataset",
    "token_f1",
]
