"""Pin and verify source-document content hashes in the corpus manifest.

HetDocQA ships the corpus as *pointers*, not content; the corpus is reconstructed
from the original sources by the build. To make that reconstruction verifiable, each
manifest entry carries a ``content_sha256`` over the materialized document text (the
text that gold character spans index into). A rebuild that fetches and parses the
same source under the locked environment (uv.lock) reproduces the same text and the
same hash; a drifted source or parser is caught.

    # maintainer: compute hashes from the materialized doc cache and write them in
    uv run python scripts/hash_corpus.py --write

    # verify a manifest against the doc cache (no network)
    uv run python scripts/hash_corpus.py --verify

The doc cache is the one populated by build_hetdocqa_dataset(cache_dir=...), default
``.cache/hetdoc/docs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST = Path("data/hetdocqa/corpus_manifest.json")
DOC_CACHE = Path(".cache/hetdoc/docs")
# Field order for each manifest entry (hashes appended after the pointer fields).
_ORDER = ["doc_id", "collection_id", "filename", "modality", "source_ref", "license",
          "content_sha256", "content_chars"]


def _cache_path(doc_id: str, cache: Path) -> Path:
    safe = doc_id.replace("/", "__").replace(":", "_._")
    return cache / f"{safe}.json"


def _text_hash(doc_id: str, cache: Path) -> tuple[str, int] | None:
    blob = _cache_path(doc_id, cache)
    if not blob.exists():
        return None
    text = json.loads(blob.read_text(encoding="utf-8"))["text"]
    return hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)


def _ordered(entry: dict) -> dict:
    return {k: entry[k] for k in _ORDER if k in entry} | {
        k: v for k, v in entry.items() if k not in _ORDER
    }


def run(write: bool, manifest_path: Path, cache: Path) -> int:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashed = missing = mismatched = 0
    for e in entries:
        res = _text_hash(e["doc_id"], cache)
        if res is None:
            missing += 1
            print(f"  MISSING from cache: {e['doc_id']}")
            continue
        sha, n = res
        if write:
            e["content_sha256"], e["content_chars"] = sha, n
            hashed += 1
        else:
            want = e.get("content_sha256")
            if want is None:
                print(f"  no hash in manifest: {e['doc_id']}")
            elif want != sha:
                mismatched += 1
                print(f"  MISMATCH {e['doc_id']}\n    manifest {want}\n    cache    {sha}")
            else:
                hashed += 1
    if write:
        manifest_path.write_text(
            json.dumps([_ordered(e) for e in entries], indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {hashed} hashes, {missing} missing -> {manifest_path}")
        return 1 if missing else 0
    print(f"\nverified {hashed} ok, {mismatched} mismatched, {missing} missing")
    return 1 if (mismatched or missing) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="compute and write hashes into the manifest")
    ap.add_argument("--verify", action="store_true", help="check manifest hashes against the doc cache")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--cache", default=str(DOC_CACHE))
    ns = ap.parse_args()
    if ns.write == ns.verify:
        sys.exit("choose exactly one of --write / --verify")
    sys.exit(run(ns.write, Path(ns.manifest), Path(ns.cache)))


if __name__ == "__main__":
    main()
