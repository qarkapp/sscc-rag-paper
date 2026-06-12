"""Build an evaluatable HetDocQA dataset from validated questions.

Re-fetches the source documents referenced by the questions, chunks them with the
configured chunker, and maps the gold character spans to chunks (>=50% overlap),
yielding a :class:`~sage.eval.dataset.RetrievalDataset` whose corpus is the chunked
documents and whose qrels are chunk ids. This makes HetDocQA a first-class benchmark
for the experiment runner, with the same chunker-agnostic relevance as the other sets.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sage.chunking import ChonkieChunker
from sage.config.schema import ChunkingCfg
from sage.core.types import DocumentSection
from sage.eval.dataset import QAExample, RetrievalDataset
from sage.eval.span_mapping import ChunkSpan, GoldSpan, build_qrels
from sage.hetdocqa.sources import (
    fetch_arxiv_pdf,
    fetch_csv,
    fetch_github_file,
    fetch_wikipedia,
)

__all__ = ["build_hetdocqa_dataset", "refetch_from_ref"]


def refetch_from_ref(collection_id: str, source_ref: str, filename: str):  # type: ignore[no-untyped-def]
    """Re-fetch a document from its reproducible ``source_ref`` pointer."""
    if source_ref.startswith("arxiv://"):
        doc = fetch_arxiv_pdf(collection_id, source_ref[len("arxiv://") :], "arXiv")
        time.sleep(2.0)
        return doc
    if source_ref.startswith("wikipedia://"):
        doc = fetch_wikipedia(collection_id, source_ref[len("wikipedia://") :])
        time.sleep(0.5)
        return doc
    if source_ref.startswith("github://"):
        repo_part, path = source_ref[len("github://") :].split(":", 1)
        owner_repo, ref = repo_part.split("@", 1)
        owner, repo = owner_repo.split("/", 1)
        return fetch_github_file(collection_id, owner, repo, ref, path, "")
    if source_ref.startswith("csv://"):
        return fetch_csv(collection_id, source_ref[len("csv://") :], filename, "")
    return None


def _fetch_text(doc_id: str, meta: dict[str, str], cache_dir: Path | None):  # type: ignore[no-untyped-def]
    """Fetch a document's (text, filename), caching the raw text on disk.

    The source fetchers hit the network (arXiv throttled at 2s/request); caching the
    materialized text keeps repeated eval runs fast, reproducible, and offline.
    """
    blob = None
    if cache_dir is not None:
        safe = doc_id.replace("/", "__").replace(":", "_._")
        blob = cache_dir / f"{safe}.json"
        if blob.exists():
            cached = json.loads(blob.read_text(encoding="utf-8"))
            return cached["text"], cached["filename"]
    doc = refetch_from_ref(meta["collection_id"], meta["source_ref"], meta["filename"])
    if doc is None:
        return None, None
    if blob is not None:
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps({"text": doc.text, "filename": doc.filename}), encoding="utf-8")
    return doc.text, doc.filename


def build_hetdocqa_dataset(
    questions_path: str | Path,
    manifest_path: str | Path,
    *,
    chunking: ChunkingCfg | None = None,
    min_overlap: float = 0.5,
    name: str = "hetdocqa",
    cache_dir: str | Path | None = None,
) -> RetrievalDataset:
    """Materialize a retrieval dataset from validated questions + the corpus manifest.

    ``cache_dir`` (recommended) caches fetched document text on disk so the corpus is
    materialized from the network once and reloaded offline thereafter.
    """
    chunker = ChonkieChunker(chunking or ChunkingCfg())
    cache_path = Path(cache_dir) if cache_dir is not None else None
    rows = [
        json.loads(line)
        for line in Path(questions_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = {d["doc_id"]: d for d in json.loads(Path(manifest_path).read_text(encoding="utf-8"))}

    needed = {span["document_id"] for r in rows for span in r["gold_spans"]}
    corpus: dict[str, str] = {}
    chunk_spans: list[ChunkSpan] = []
    for doc_id in sorted(needed):
        meta = manifest.get(doc_id)
        if meta is None:
            continue
        text, filename = _fetch_text(doc_id, meta, cache_path)
        if text is None:
            continue
        sections = [DocumentSection(doc_id, text, 0, len(text))]
        for chunk in chunker.chunk(doc_id, sections, filename):
            corpus[chunk.chunk_id] = chunk.content
            chunk_spans.append(ChunkSpan(chunk.chunk_id, doc_id, chunk.char_start, chunk.char_end))

    gold_by_query = {
        r["qid"]: [
            GoldSpan(s["document_id"], s["char_start"], s["char_end"], s.get("grade", 1))
            for s in r["gold_spans"]
        ]
        for r in rows
    }
    qrels = build_qrels(gold_by_query, chunk_spans, min_overlap=min_overlap)
    # Keep only questions with at least one mapped gold chunk.
    examples = [
        QAExample(
            qid=r["qid"],
            question=r["question"],
            answers=(r["answer"],),
            metadata={"type": r["type"], "split": r.get("split", "test")},
        )
        for r in rows
        if qrels.get(r["qid"])
    ]
    qrels = {q.qid: qrels[q.qid] for q in examples}
    return RetrievalDataset(name=name, examples=examples, corpus=corpus, qrels=qrels)
