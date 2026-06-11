"""Tests for the HetDocQA construction toolkit (deterministic parts + fake-LLM flow)."""

from __future__ import annotations

import json

import numpy as np

from sage.chunking import ChonkieChunker
from sage.config.schema import ChunkingCfg
from sage.core.types import DocumentSection
from sage.hetdocqa import (
    Collection,
    Modality,
    QuestionType,
    SourceDoc,
    apply_splits,
    assign_collection_splits,
    check_answerable_without_context,
    cross_validate,
    dataset_stats,
    draft_question,
    locate_span,
    near_duplicate_mask,
    to_retrieval_dataset,
    write_release,
)
from sage.hetdocqa.schema import QuestionCandidate
from sage.testing import FakeGenerator

# -- spans -----------------------------------------------------------------


def test_locate_span_exact_and_fuzzy():
    doc = "The melting point of steel is about 1370 degrees Celsius in most alloys."
    assert locate_span(doc, "melting point of steel") is not None
    start, end = locate_span(doc, "melting point of steel")
    assert doc[start:end] == "melting point of steel"
    assert locate_span(doc, "completely unrelated phrase") is None


# -- dedup / splits / stats ------------------------------------------------


def test_near_duplicate_mask_drops_duplicates():
    embeddings = np.array([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float32)
    keep = near_duplicate_mask(embeddings, threshold=0.9)
    assert keep == [True, False, True]


def test_collection_splits_are_disjoint():
    collections = [Collection(f"c{i}", f"Collection {i}", ()) for i in range(8)]
    splits = assign_collection_splits(collections, seed=1)
    assert set(splits.values()) <= {"calibration", "dev", "test"}
    assert len(splits) == 8
    # every collection assigned exactly one split
    assert all(c.collection_id in splits for c in collections)


def _candidate(qid: str, cid: str, qtype: QuestionType) -> QuestionCandidate:
    from sage.eval.span_mapping import GoldSpan

    return QuestionCandidate(
        qid=qid,
        question=f"q {qid}",
        answer="a",
        qtype=qtype,
        collection_id=cid,
        evidence_doc_ids=[f"{cid}:doc"],
        gold_spans=[GoldSpan(f"{cid}:doc", 0, 10)],
    )


def test_apply_splits_and_stats():
    cands = [_candidate(f"q{i}", f"c{i % 2}", QuestionType.FACTUAL) for i in range(4)]
    apply_splits(cands, {"c0": "dev", "c1": "test"})
    assert {c.split for c in cands} == {"dev", "test"}
    stats = dataset_stats(cands)
    assert stats["total"] == 4
    assert stats["by_type"]["factual"] == 4


# -- generation flow (fake LLM) --------------------------------------------


def _source(cid: str, text: str, modality: Modality = Modality.PROSE) -> SourceDoc:
    return SourceDoc(
        doc_id=f"{cid}:0",
        collection_id=cid,
        filename="d.md",
        text=text,
        modality=modality,
        source_ref="test://d",
        license="CC-BY-4.0",
    )


async def test_draft_question_locates_evidence():
    doc_text = "Photosynthesis converts sunlight into chemical energy in chloroplasts."
    gen = FakeGenerator(
        response=json.dumps(
            {
                "question": "Where does photosynthesis occur?",
                "answer": "chloroplasts",
                "evidence": ["chemical energy in chloroplasts"],
                "type": "factual",
            }
        )
    )
    cand = await draft_question(gen, QuestionType.FACTUAL, [_source("c0", doc_text)], qid="c0-0")
    assert cand is not None
    assert cand.gold_spans  # evidence located as a span
    assert cand.evidence_doc_ids == ["c0:0"]


async def test_answerability_and_cross_validation():
    from sage.eval.span_mapping import GoldSpan

    doc = _source("c0", "The reactor uses a custom XJ-9 coolant loop described herein.")
    cand = QuestionCandidate(
        qid="c0-0",
        question="What coolant loop does the reactor use?",
        answer="XJ-9 coolant loop",
        qtype=QuestionType.FACTUAL,
        collection_id="c0",
        evidence_doc_ids=["c0:0"],
        gold_spans=[GoldSpan("c0:0", 16, 45)],
    )
    # No-context model says UNKNOWN -> not answerable without retrieval (kept).
    answerable = await check_answerable_without_context(cand, FakeGenerator(response="UNKNOWN"))
    assert answerable is False

    verdict = await cross_validate(
        cand, [doc], FakeGenerator(response='{"supported": true, "type_ok": true, "natural": true}')
    )
    assert verdict == {"supported": True, "type_ok": True, "natural": True}
    assert cand.is_clean


# -- release / eval integration --------------------------------------------


def test_to_retrieval_dataset_and_release(tmp_path):
    from sage.eval.span_mapping import GoldSpan

    text = "Alpha beta gamma. " * 40  # ~720 chars
    sections = [DocumentSection("c0:0", text, 0, len(text))]
    chunks = ChonkieChunker(ChunkingCfg(prose_target_chars=200)).chunk("c0:0", sections, "d.md")
    # gold span inside the first chunk
    cand = QuestionCandidate(
        qid="c0-0",
        question="what?",
        answer="alpha",
        qtype=QuestionType.FACTUAL,
        collection_id="c0",
        evidence_doc_ids=["c0:0"],
        gold_spans=[GoldSpan("c0:0", 0, 50)],
        answerable_without_context=False,
        validation={"supported": True, "type_ok": True, "natural": True},
        split="test",
    )
    ds = to_retrieval_dataset([cand], chunks, split="test")
    assert ds.examples[0].qid == "c0-0"
    assert ds.qrels["c0-0"]  # at least one chunk is relevant via span overlap

    write_release(
        tmp_path,
        [cand],
        [Collection("c0", "C0", ("c0:0",))],
        [_source("c0", text)],
    )
    written = json.loads((tmp_path / "questions.jsonl").read_text().splitlines()[0])
    assert written["qid"] == "c0-0"
    assert (tmp_path / "DATASHEET.md").exists()
    assert (tmp_path / "corpus_manifest.json").exists()
