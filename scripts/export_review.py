"""Build data/hetdocqa/review.json from an existing questions.jsonl.

Re-fetches only the source documents referenced by the questions (throttled, with a
retry) and embeds the gold evidence in context for the review app. Use this when a
build wrote the release files but not the review bundle.

    uv run python scripts/export_review.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sage.eval.span_mapping import GoldSpan
from sage.hetdocqa.review import write_review_bundle
from sage.hetdocqa.schema import QuestionCandidate, QuestionType, SourceDoc
from sage.hetdocqa.sources import (
    fetch_arxiv_pdf,
    fetch_csv,
    fetch_github_file,
    fetch_wikipedia,
)

DATA = Path("data/hetdocqa")


def refetch(collection_id: str, source_ref: str, filename: str) -> SourceDoc | None:
    """Re-fetch a document from its reproducible source_ref pointer (with one retry)."""
    for _ in range(2):
        doc: SourceDoc | None = None
        if source_ref.startswith("arxiv://"):
            doc = fetch_arxiv_pdf(collection_id, source_ref[len("arxiv://") :], "arXiv")
            time.sleep(2.0)
        elif source_ref.startswith("wikipedia://"):
            doc = fetch_wikipedia(collection_id, source_ref[len("wikipedia://") :])
            time.sleep(1.0)
        elif source_ref.startswith("github://"):
            spec = source_ref[len("github://") :]
            repo_part, path = spec.split(":", 1)
            owner_repo, ref = repo_part.split("@", 1)
            owner, repo = owner_repo.split("/", 1)
            doc = fetch_github_file(collection_id, owner, repo, ref, path, "")
        elif source_ref.startswith("csv://"):
            doc = fetch_csv(collection_id, source_ref[len("csv://") :], filename, "")
        if doc is not None:
            return doc
        time.sleep(2.0)
    return None


def main() -> None:
    rows = [
        json.loads(line)
        for line in (DATA / "questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manifest = {d["doc_id"]: d for d in json.loads((DATA / "corpus_manifest.json").read_text())}

    needed = {span["document_id"] for r in rows for span in r["gold_spans"]}
    print(f"re-fetching {len(needed)} source documents referenced by {len(rows)} questions...")
    sources: list[SourceDoc] = []
    for doc_id in sorted(needed):
        meta = manifest.get(doc_id)
        if meta is None:
            continue
        doc = refetch(meta["collection_id"], meta["source_ref"], meta["filename"])
        if doc is not None:
            sources.append(doc)
    print(f"fetched {len(sources)}/{len(needed)} documents")

    candidates = [
        QuestionCandidate(
            qid=r["qid"],
            question=r["question"],
            answer=r["answer"],
            qtype=QuestionType(r["type"]),
            collection_id=r["collection_id"],
            evidence_doc_ids=sorted({s["document_id"] for s in r["gold_spans"]}),
            gold_spans=[
                GoldSpan(s["document_id"], s["char_start"], s["char_end"], s.get("grade", 1))
                for s in r["gold_spans"]
            ],
            answerable_without_context=False,
            validation={"passed_auto_filters": True},
            split=r["split"],
        )
        for r in rows
    ]
    write_review_bundle(DATA / "review.json", candidates, sources)
    located = sum(1 for rec in json.loads((DATA / "review.json").read_text()) if rec["evidence"])
    print(
        f"wrote data/hetdocqa/review.json ({located}/{len(rows)} questions have located evidence)"
    )


if __name__ == "__main__":
    main()
