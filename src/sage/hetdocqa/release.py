"""Release artifacts: datasheet, JSONL files, and eval-dataset conversion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from sage.core.types import Chunk
from sage.eval.dataset import QAExample, RetrievalDataset
from sage.eval.span_mapping import ChunkSpan, build_qrels
from sage.hetdocqa.curate import dataset_stats
from sage.hetdocqa.schema import Collection, QuestionCandidate, SourceDoc

__all__ = ["build_datasheet", "to_retrieval_dataset", "write_release"]


def build_datasheet(
    candidates: Sequence[QuestionCandidate],
    sources: Sequence[SourceDoc],
    *,
    name: str = "HetDocQA",
) -> str:
    """Render a Gebru-style datasheet (markdown) from the dataset and its sources."""
    stats = dataset_stats(candidates)
    licenses = sorted({d.license for d in sources})
    modalities = sorted({d.modality.value for d in sources})
    lines = [
        f"# Datasheet: {name}",
        "",
        "## Motivation",
        "A heterogeneous, multi-format retrieval benchmark (PDF, code, markdown, "
        "tabular, prose) for evaluating retrieval over realistic mixed-document "
        "collections, where most public benchmarks are homogeneous Wikipedia prose.",
        "",
        "## Composition",
        f"- Questions: {stats['total']}",
        f"- Type distribution: {stats['by_type']}",
        f"- Split distribution: {stats['by_split']}",
        f"- Multi-evidence questions: {stats['multi_evidence_questions']}",
        f"- Average gold spans per question: {stats['avg_gold_spans']}",
        f"- Source modalities: {modalities}",
        f"- Source licenses: {licenses}",
        "",
        "## Collection process",
        "Questions were drafted by a strong closed model (distinct from any model "
        "evaluated as a generator) over selected documents, then automatically "
        "filtered: a no-context answerability check removed questions answerable "
        "without retrieval, near-duplicates were removed by embedding similarity, and "
        "an LLM cross-validation pass checked evidence support, type label, and "
        "naturalness. A final human validation pass is applied before release.",
        "",
        "## Labels",
        "Gold evidence is annotated as character spans in the source documents and "
        "mapped to any system's chunks at evaluation time (>=50% span overlap), so "
        "retrieval metrics are independent of chunking.",
        "",
        "## Provenance and integrity",
        "The corpus is distributed as source pointers (source_ref), not redistributed "
        "content. Each manifest entry pins the materialized document text with a "
        "SHA-256 (content_sha256); the build verifies the reconstructed text against "
        "this hash under the locked environment, so a drifted source or parser is "
        "caught and span offsets stay valid.",
        "",
        "## Splits",
        "Calibration / dev / test splits are disjoint by collection, so thresholds "
        "tuned on dev cannot exploit corpus structure shared with test.",
        "",
        "## Known limitations",
        "Questions are LLM-drafted (then filtered and human-checked); English-only; "
        "domain coverage reflects the chosen sources.",
        "",
    ]
    return "\n".join(lines)


def _span_dict(span: Any) -> dict[str, Any]:
    return asdict(span)


def write_release(
    out_dir: str | Path,
    candidates: Sequence[QuestionCandidate],
    collections: Sequence[Collection],
    sources: Sequence[SourceDoc],
    *,
    clean_only: bool = True,
) -> None:
    """Write questions.jsonl, collections.json, a corpus manifest, and the datasheet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = [c for c in candidates if c.is_clean] if clean_only else list(candidates)

    with (out / "questions.jsonl").open("w", encoding="utf-8") as fh:
        for c in selected:
            fh.write(
                json.dumps(
                    {
                        "qid": c.qid,
                        "question": c.question,
                        "answer": c.answer,
                        "type": c.qtype.value,
                        "collection_id": c.collection_id,
                        "split": c.split,
                        "gold_spans": [_span_dict(s) for s in c.gold_spans],
                    }
                )
                + "\n"
            )

    (out / "collections.json").write_text(
        json.dumps([asdict(c) for c in collections], indent=2), encoding="utf-8"
    )
    # Corpus manifest: reproducible source pointers, not redistributed content.
    (out / "corpus_manifest.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": d.doc_id,
                    "collection_id": d.collection_id,
                    "filename": d.filename,
                    "modality": d.modality.value,
                    "source_ref": d.source_ref,
                    "license": d.license,
                }
                for d in sources
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "DATASHEET.md").write_text(build_datasheet(selected, sources), encoding="utf-8")


def to_retrieval_dataset(
    candidates: Sequence[QuestionCandidate],
    chunks: Sequence[Chunk],
    *,
    name: str = "hetdocqa",
    split: str | None = None,
    min_overlap: float = 0.5,
) -> RetrievalDataset:
    """Convert candidates + chunked corpus into an evaluatable dataset (span->chunk qrels)."""
    selected = [c for c in candidates if split is None or c.split == split]
    corpus = {c.chunk_id: c.content for c in chunks}
    chunk_spans = [ChunkSpan(c.chunk_id, c.document_id, c.char_start, c.char_end) for c in chunks]
    gold_by_query = {c.qid: c.gold_spans for c in selected}
    qrels = build_qrels(gold_by_query, chunk_spans, min_overlap=min_overlap)
    examples = [
        QAExample(
            qid=c.qid, question=c.question, answers=(c.answer,), metadata={"type": c.qtype.value}
        )
        for c in selected
    ]
    return RetrievalDataset(name=name, examples=examples, corpus=corpus, qrels=qrels)
