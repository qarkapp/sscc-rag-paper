"""Carve a per-batch review bundle out of the freshly built ``review.json``.

The build is cumulative: ``data/hetdocqa/review.json`` holds every clean candidate
across all collections. This script subtracts the questions already handled by prior
batches -- those human-validated (``batch*_validated_questions.jsonl``) and those
still queued in an earlier review bundle (``review_batch*.json``) -- and writes the
genuinely new remainder to ``review_batch<N>.json`` for the review app.

    uv run python scripts/export_review_batch.py 4
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATA = Path("data/hetdocqa")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _qids(items: list[dict]) -> set[str]:
    return {row["qid"] for row in items}


def main() -> None:
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    review = json.loads((DATA / "review.json").read_text())

    # Questions already settled by previous batches: validated answers + queued reviews.
    seen: set[str] = set()
    for path in sorted(DATA.glob("batch*_validated_questions.jsonl")):
        seen |= _qids(_load_jsonl(path))
    for path in sorted(DATA.glob("review_batch*.json")):
        if path.name == f"review_batch{batch}.json":
            continue
        seen |= _qids(json.loads(path.read_text()))

    fresh = [row for row in review if row["qid"] not in seen]

    out = DATA / f"review_batch{batch}.json"
    out.write_text(json.dumps(fresh, ensure_ascii=False, indent=2))

    by_type = dict(Counter(row["type"] for row in fresh))
    by_split = dict(Counter(row["split"] for row in fresh))
    print(f"review.json candidates : {len(review)}")
    print(f"already settled (skip) : {len(seen)}")
    print(f"new for batch {batch}        : {len(fresh)}")
    print(f"  by type  : {by_type}")
    print(f"  by split : {by_split}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
