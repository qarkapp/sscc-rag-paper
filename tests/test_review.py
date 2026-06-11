"""Tests for the review-bundle exporter."""

from __future__ import annotations

import json

from sage.eval.span_mapping import GoldSpan
from sage.hetdocqa.review import review_records, write_review_bundle
from sage.hetdocqa.schema import Modality, QuestionCandidate, QuestionType, SourceDoc


def _doc() -> SourceDoc:
    text = "Intro sentence. The reactor uses an XJ-9 coolant loop. Closing remarks here."
    return SourceDoc(
        doc_id="c0:0",
        collection_id="c0",
        filename="spec.md",
        text=text,
        modality=Modality.MARKDOWN,
        source_ref="test://spec",
        license="CC-BY-4.0",
    )


def _candidate(doc: SourceDoc) -> QuestionCandidate:
    start = doc.text.index("XJ-9 coolant loop")
    return QuestionCandidate(
        qid="c0-0",
        question="What coolant loop is used?",
        answer="XJ-9 coolant loop",
        qtype=QuestionType.FACTUAL,
        collection_id="c0",
        evidence_doc_ids=["c0:0"],
        gold_spans=[GoldSpan("c0:0", start, start + len("XJ-9 coolant loop"))],
        answerable_without_context=False,
        validation={"supported": True, "type_ok": True, "natural": True},
        split="test",
    )


def test_review_records_embed_evidence_in_context():
    doc = _doc()
    records = review_records([_candidate(doc)], [doc])
    assert len(records) == 1
    rec = records[0]
    assert rec["qid"] == "c0-0"
    assert rec["auto"]["is_clean"] is True
    ev = rec["evidence"][0]
    assert ev["text"] == "XJ-9 coolant loop"
    assert ev["filename"] == "spec.md"
    assert ev["modality"] == "markdown"
    assert "reactor uses" in ev["before"]  # context window precedes the span
    assert ev["after"].startswith(".")  # context window follows the span


def test_write_review_bundle_round_trip(tmp_path):
    doc = _doc()
    out = tmp_path / "review.json"
    write_review_bundle(out, [_candidate(doc)], [doc])
    loaded = json.loads(out.read_text())
    assert loaded[0]["question"] == "What coolant loop is used?"
    assert loaded[0]["evidence"][0]["text"] == "XJ-9 coolant loop"
