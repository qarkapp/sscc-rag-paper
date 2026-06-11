"""Export a self-contained review bundle for human validation.

The public release ships only pointers (no third-party content), but a human
reviewer needs to see the gold evidence in context. This writes a local-only
``review.json`` that embeds, for every candidate, the evidence span text and a
surrounding context window from the source document, so the review app can
highlight the evidence without re-fetching anything.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sage.hetdocqa.schema import QuestionCandidate, SourceDoc

__all__ = ["review_records", "write_review_bundle"]

_CONTEXT_CHARS = 320


def review_records(
    candidates: Sequence[QuestionCandidate], sources: Sequence[SourceDoc]
) -> list[dict[str, Any]]:
    """Build review records with evidence text shown in context."""
    by_id = {d.doc_id: d for d in sources}
    records: list[dict[str, Any]] = []
    for c in candidates:
        evidence: list[dict[str, Any]] = []
        for span in c.gold_spans:
            doc = by_id.get(span.document_id)
            if doc is None:
                continue
            start, end = span.char_start, span.char_end
            evidence.append(
                {
                    "document_id": span.document_id,
                    "filename": doc.filename,
                    "modality": doc.modality.value,
                    "char_start": start,
                    "char_end": end,
                    "before": doc.text[max(0, start - _CONTEXT_CHARS) : start],
                    "text": doc.text[start:end],
                    "after": doc.text[end : end + _CONTEXT_CHARS],
                }
            )
        records.append(
            {
                "qid": c.qid,
                "question": c.question,
                "answer": c.answer,
                "type": c.qtype.value,
                "split": c.split,
                "collection_id": c.collection_id,
                "auto": {
                    "answerable_without_context": c.answerable_without_context,
                    "validation": c.validation,
                    "is_clean": c.is_clean,
                },
                "evidence": evidence,
            }
        )
    return records


def write_review_bundle(
    out_path: str | Path,
    candidates: Sequence[QuestionCandidate],
    sources: Sequence[SourceDoc],
) -> None:
    """Write the review bundle JSON (all candidates, including auto-rejected ones)."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(review_records(candidates, sources), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
