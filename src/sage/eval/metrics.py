"""Retrieval and answer metrics.

Retrieval metrics are thin wrappers over ``ir_measures`` (the trusted pytrec_eval
implementation), so numbers are comparable to published results. The distinction
between Recall@k and Success/Hit@k is made explicit: ``Recall@k`` is the fraction of
relevant documents retrieved; ``Success@k`` (a.k.a. Hit@k) is whether at least one
relevant document is in the top k.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence

__all__ = [
    "DEFAULT_MEASURES",
    "exact_match",
    "retrieval_metrics",
    "token_f1",
]

Qrels = Mapping[str, Mapping[str, int]]
Run = Mapping[str, Mapping[str, float]]

DEFAULT_MEASURES = ("nDCG@10", "R@5", "R@10", "Success@5", "Success@10", "RR@10")


def retrieval_metrics(
    qrels: Qrels, run: Run, measures: Sequence[str] = DEFAULT_MEASURES
) -> dict[str, float]:
    """Compute aggregate retrieval metrics via ir_measures."""
    import ir_measures

    parsed = [ir_measures.parse_measure(m) for m in measures]
    # ir_measures expects integer relevance grades.
    clean_qrels = {q: {d: int(r) for d, r in docs.items()} for q, docs in qrels.items()}
    clean_run = {q: {d: float(s) for d, s in docs.items()} for q, docs in run.items()}
    aggregate = ir_measures.calc_aggregate(parsed, clean_qrels, clean_run)
    return {str(measure): float(value) for measure, value in aggregate.items()}


# -- answer metrics (SQuAD-style normalization) ----------------------------


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
