"""Retrieval-evaluation dataset abstractions and loaders.

A :class:`RetrievalDataset` bundles questions, a corpus, and relevance judgements
(qrels). Passage-retrieval benchmarks (BEIR, NQ) index each corpus document as a
single passage, so result ids align directly with qrels. Document-with-spans
benchmarks (HetDocQA) chunk documents and derive qrels via span mapping.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sage.core.protocols import Embedder, Generator, VectorStore
from sage.core.types import StoreRow

__all__ = [
    "QAExample",
    "RetrievalDataset",
    "build_raptor_index",
    "index_passages",
    "load_beir",
    "load_jsonl",
]


@dataclass(frozen=True, slots=True)
class QAExample:
    qid: str
    question: str
    answers: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalDataset:
    """Questions, a passage corpus, and qrels."""

    name: str
    examples: list[QAExample]
    corpus: dict[str, str]  # passage_id -> text
    qrels: dict[str, dict[str, int]]  # qid -> {passage_id: grade}


async def index_passages(
    store: VectorStore, embedder: Embedder, corpus: dict[str, str], *, batch_size: int = 128
) -> None:
    """Index each corpus passage as a single leaf row (id = passage id)."""
    ids = list(corpus)
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        embeddings = await embedder.embed_documents([corpus[i] for i in batch])
        await store.upsert(
            [
                StoreRow(
                    chunk_id=pid,
                    document_id=pid,
                    chunk_index=0,
                    content=corpus[pid],
                    embedding=embeddings[k],
                    level=0,
                )
                for k, pid in enumerate(batch)
            ]
        )


async def build_raptor_index(
    store: VectorStore,
    embedder: Embedder,
    generator: Generator,
    cfg: object,
    *,
    seed: int = 42,
) -> int:
    """Build per-document RAPTOR trees + cross-doc tier over already-indexed leaves.

    The benchmark corpus is flat (one leaf per passage); a passage id is ``doc:idx``,
    so leaves are grouped by their source document, a tree is built per document, and a
    cross-document tier is built over the per-document top summaries. Idempotent: a
    no-op if summary levels already exist in the (persistent) store. Returns the number
    of summary nodes added.
    """
    from collections import defaultdict

    from sage.raptor.cross_doc import build_cross_document_tier
    from sage.raptor.tree import build_tree

    leaves = await store.all_leaf_rows()
    if not leaves or await store.count() > len(leaves):
        return 0  # nothing to index, or a tree already exists

    by_doc: dict[str, list[StoreRow]] = defaultdict(list)
    for row in leaves:
        by_doc[row.chunk_id.rsplit(":", 1)[0]].append(row)

    top_nodes: list[StoreRow] = []
    n_summaries = 0
    for doc_id, doc_leaves in by_doc.items():
        summaries = await build_tree(
            sorted(doc_leaves, key=lambda r: r.chunk_id),
            document_id=doc_id,
            embedder=embedder,
            generator=generator,
            store=store,
            cfg=cfg,  # type: ignore[arg-type]
            seed=seed,
        )
        n_summaries += len(summaries)
        if summaries:
            top_level = max(s.level for s in summaries)
            top_nodes.extend(s for s in summaries if s.level == top_level)
        else:
            top_nodes.extend(doc_leaves)  # too small to summarize: represent by leaves

    if getattr(cfg, "cross_doc", False) and top_nodes:
        await build_cross_document_tier(
            top_nodes, embedder=embedder, generator=generator, store=store,
            cfg=cfg, seed=seed,  # type: ignore[arg-type]
        )
    return n_summaries


def load_jsonl(
    questions_path: str | Path, corpus_path: str | Path, qrels_path: str | Path, *, name: str
) -> RetrievalDataset:
    """Load a dataset from JSONL files (questions, corpus) and a qrels JSON."""
    examples = [
        QAExample(
            qid=row["qid"],
            question=row["question"],
            answers=tuple(row.get("answers", [])),
            metadata=row.get("metadata", {}),
        )
        for row in _read_jsonl(questions_path)
    ]
    corpus = {row["id"]: row["text"] for row in _read_jsonl(corpus_path)}
    qrels = json.loads(Path(qrels_path).read_text(encoding="utf-8"))
    return RetrievalDataset(name=name, examples=examples, corpus=corpus, qrels=qrels)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_beir(
    name: str = "scifact", split: str = "test", *, max_queries: int | None = None
) -> RetrievalDataset:
    """Load a BEIR dataset from the Hugging Face hub (network access required).

    Used as a retriever sanity anchor: with a standard embedder, retrieval metrics
    here should reproduce published BEIR numbers.
    """
    from datasets import load_dataset

    corpus_rows = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    corpus = {row["_id"]: (row["title"] + " " + row["text"]).strip() for row in corpus_rows}
    query_rows = load_dataset(f"BeIR/{name}", "queries")["queries"]
    questions = {row["_id"]: row["text"] for row in query_rows}

    qrels_rows = load_dataset(f"BeIR/{name}-qrels")[split]
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qid, did, score = str(row["query-id"]), str(row["corpus-id"]), int(row["score"])
        qrels.setdefault(qid, {})[did] = score

    qids: Sequence[str] = list(qrels)
    if max_queries is not None:
        qids = qids[:max_queries]
    examples = [QAExample(qid=q, question=questions[q]) for q in qids if q in questions]
    return RetrievalDataset(name=f"beir-{name}", examples=examples, corpus=corpus, qrels=qrels)
