"""LLM-driven question drafting and automated cross-validation.

A strong, closed model (distinct from any model evaluated as a SAGE generator, to
avoid self-preference) drafts a question, its answer, the supporting evidence
snippets, and a type label for selected documents. Two automated filters follow:

* **answerability** -- a no-context answer is compared to the gold answer; if the
  question is answerable from parametric knowledge alone it is dropped (retrieval
  must be necessary).
* **cross-validation** -- a separate prompt checks that the evidence supports the
  answer, the type label fits, and the question is natural and unambiguous.

The remaining candidates are flagged for a final human pass.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from sage.core.protocols import Generator
from sage.eval.metrics import token_f1
from sage.hetdocqa.schema import QuestionCandidate, QuestionType, SourceDoc
from sage.hetdocqa.spans import snippets_to_spans

__all__ = ["check_answerable_without_context", "cross_validate", "draft_question"]

_DRAFT_SYSTEM = (
    "You write evaluation questions for a document retrieval benchmark. Given source "
    "documents, write one question whose answer requires reading the provided "
    "evidence (not general knowledge). Respond ONLY with JSON: "
    '{"question": str, "answer": str, "evidence": [verbatim snippets copied from the '
    'documents], "type": one of factual|code|cross_document|multi_hop|thematic}.'
)
_NO_CONTEXT_SYSTEM = (
    "Answer the question from your own knowledge in one short phrase. "
    "If you cannot, reply exactly: UNKNOWN."
)
_VALIDATE_SYSTEM = (
    "You audit benchmark questions. Given a question, a proposed answer, the cited "
    "evidence, and a type label, respond ONLY with JSON: "
    '{"supported": bool, "type_ok": bool, "natural": bool} -- supported means the '
    "evidence entails the answer; type_ok means the label fits; natural means the "
    "question is clear and unambiguous."
)

_DOC_BUDGET = 2500


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text
    brace = re.search(r"\{.*\}", raw, re.S)
    if not brace:
        return None
    try:
        parsed = json.loads(brace.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _render_docs(docs: Sequence[SourceDoc]) -> str:
    blocks = []
    for d in docs:
        blocks.append(f"[{d.filename} | {d.modality}]\n{d.text[:_DOC_BUDGET]}")
    return "\n\n---\n\n".join(blocks)


async def draft_question(
    generator: Generator,
    qtype: QuestionType,
    docs: Sequence[SourceDoc],
    *,
    qid: str,
) -> QuestionCandidate | None:
    """Draft one question over ``docs`` and locate its evidence as spans."""
    user = (
        f"Target question type: {qtype}.\n\nDocuments:\n\n{_render_docs(docs)}\n\n"
        "Write the question now."
    )
    parsed = _extract_json(await generator.complete(_DRAFT_SYSTEM, user, max_tokens=600))
    if not parsed or not parsed.get("question") or not parsed.get("answer"):
        return None

    snippets = [s for s in parsed.get("evidence", []) if isinstance(s, str)]
    gold_spans = []
    evidence_doc_ids = []
    for doc in docs:
        located = snippets_to_spans(doc.doc_id, doc.text, snippets)
        if located:
            gold_spans.extend(located)
            evidence_doc_ids.append(doc.doc_id)
    if not gold_spans:
        return None

    return QuestionCandidate(
        qid=qid,
        question=str(parsed["question"]).strip(),
        answer=str(parsed["answer"]).strip(),
        qtype=qtype,
        collection_id=docs[0].collection_id,
        evidence_doc_ids=evidence_doc_ids,
        gold_spans=gold_spans,
    )


async def check_answerable_without_context(
    candidate: QuestionCandidate, generator: Generator, *, f1_threshold: float = 0.6
) -> bool:
    """Set and return whether the question is answerable with no retrieved context."""
    answer = await generator.complete(_NO_CONTEXT_SYSTEM, candidate.question, max_tokens=64)
    answerable = (
        "unknown" not in answer.lower() and token_f1(answer, candidate.answer) >= f1_threshold
    )
    candidate.answerable_without_context = answerable
    return answerable


async def cross_validate(
    candidate: QuestionCandidate, docs: Sequence[SourceDoc], generator: Generator
) -> dict[str, bool]:
    """Run the LLM cross-validation pass; record and return the verdicts."""
    evidence = "\n".join(
        d.text[s.char_start : s.char_end]
        for d in docs
        for s in candidate.gold_spans
        if s.document_id == d.doc_id
    )
    user = (
        f"Question: {candidate.question}\nAnswer: {candidate.answer}\n"
        f"Type: {candidate.qtype}\nEvidence:\n{evidence[:_DOC_BUDGET]}"
    )
    parsed = _extract_json(await generator.complete(_VALIDATE_SYSTEM, user, max_tokens=120)) or {}
    verdict = {
        "supported": bool(parsed.get("supported", False)),
        "type_ok": bool(parsed.get("type_ok", False)),
        "natural": bool(parsed.get("natural", False)),
    }
    candidate.validation = verdict
    return verdict
