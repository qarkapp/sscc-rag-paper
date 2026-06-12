"""Merge validated HetDocQA batches into the final dataset.

Concatenates the human-validated batches (batch 1 takes priority on duplicates),
removes cross-batch near-duplicate questions by embedding similarity, and -- crucially
-- re-assigns collection-disjoint splits over the *merged* collection set so that the
same collection never appears in two splits (the batches were split independently
over different collection pools). Writes the final ``hetdocqa.jsonl`` + datasheet.

    uv run python scripts/merge_hetdocqa.py
"""

from __future__ import annotations

import asyncio
import collections as pycollections
import json
from pathlib import Path

import numpy as np

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.eval.span_mapping import GoldSpan
from sage.hetdocqa.curate import assign_collection_splits, near_duplicate_mask
from sage.hetdocqa.release import build_datasheet
from sage.hetdocqa.schema import Collection, QuestionCandidate, QuestionType

DATA = Path("data/hetdocqa")
BATCHES = [
    "batch1_validated_questions.jsonl",
    "batch2_validated_questions.jsonl",
    "batch3_validated_questions.jsonl",
    "batch4_validated_questions.jsonl",
]
DEDUP_THRESHOLD = 0.93


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


async def main() -> None:
    rows: list[dict] = []
    for name in BATCHES:
        batch = _load(DATA / name)
        print(f"{name}: {len(batch)}")
        rows.extend(batch)
    print(f"combined: {len(rows)}")

    # Cross-batch near-duplicate removal (batch 1 entries come first -> kept).
    cache = CallCache(".cache/hetdoc", CacheMode.READ_WRITE)
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)
    embeddings = await embedder.embed_documents([r["question"] for r in rows])
    keep = near_duplicate_mask(embeddings, threshold=DEDUP_THRESHOLD)
    rows = [r for r, k in zip(rows, keep, strict=True) if k]
    print(f"after cross-batch dedup: {len(rows)} ({keep.count(False)} duplicates removed)")

    # Re-assign collection-disjoint splits over the merged collection set.
    collection_ids = sorted({r["collection_id"] for r in rows})
    collections = [Collection(cid, cid, ()) for cid in collection_ids]
    splits = assign_collection_splits(collections, seed=42)
    for r in rows:
        r["split"] = splits[r["collection_id"]]

    # Write the final dataset and a fresh datasheet.
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "hetdocqa.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    candidates = [
        QuestionCandidate(
            qid=r["qid"],
            question=r["question"],
            answer=r["answer"],
            qtype=QuestionType(r["type"]),
            collection_id=r["collection_id"],
            evidence_doc_ids=[],
            gold_spans=[
                GoldSpan(s["document_id"], s["char_start"], s["char_end"], s.get("grade", 1))
                for s in r["gold_spans"]
            ],
            answerable_without_context=False,
            validation={"human_validated": True},
            split=r["split"],
        )
        for r in rows
    ]
    sources = []  # datasheet license/modality info comes from the manifest if present
    manifest_path = DATA / "corpus_manifest.json"
    if manifest_path.exists():
        from sage.hetdocqa.schema import Modality, SourceDoc

        sources = [
            SourceDoc(
                d["doc_id"],
                d["collection_id"],
                d["filename"],
                "",
                Modality(d["modality"]),
                d["source_ref"],
                d["license"],
            )
            for d in json.loads(manifest_path.read_text())
        ]
    (DATA / "DATASHEET.md").write_text(
        build_datasheet(candidates, sources, name="HetDocQA"), encoding="utf-8"
    )

    print("\n=== final HetDocQA ===")
    print(f"total: {len(rows)}")
    print("by type:", dict(pycollections.Counter(r["type"] for r in rows)))
    print("by split:", dict(pycollections.Counter(r["split"] for r in rows)))
    print(f"collections: {len(collection_ids)}")
    print(f"avg gold spans: {np.mean([len(r['gold_spans']) for r in rows]):.2f}")
    print("wrote data/hetdocqa/hetdocqa.jsonl + DATASHEET.md")


if __name__ == "__main__":
    asyncio.run(main())
