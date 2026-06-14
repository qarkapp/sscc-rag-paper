"""Fetch large release artifacts from a Zenodo record and verify checksums.

The benchmark itself ships in git. Any large convenience artifact (e.g. the
embedding cache) is deposited in the project's Zenodo record and fetched here,
verified by SHA-256 against ``data_manifest.json`` so a corrupted or replaced file
is caught. No account or key is needed: Zenodo files have stable public URLs.

    uv run python scripts/fetch_data.py                       # download + verify all
    uv run python scripts/fetch_data.py --make-manifest <dir>  # (maintainer) hash files

Manifest (``data_manifest.json`` at the repo root):

    {
      "zenodo_record": "1234567",
      "files": [
        {"name": "hetdocqa_embeddings.tar.zst", "sha256": "...", "dest": ".cache"}
      ]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "data_manifest.json"
_CHUNK = 1 << 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        while block := resp.read(_CHUNK):
            fh.write(block)
            done += len(block)
            if total:
                print(f"\r  {dest.name}: {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def fetch() -> int:
    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST}; run with --make-manifest <dir> first")
    man = json.loads(MANIFEST.read_text())
    record = str(man.get("zenodo_record", ""))
    if not record or record.startswith("<"):
        sys.exit("manifest has no real zenodo_record yet -- deposit on Zenodo and fill it in")

    failures = 0
    for entry in man["files"]:
        name, want = entry["name"], entry["sha256"]
        dest = Path(entry.get("dest", ".")) / name
        if dest.exists() and _sha256(dest) == want:
            print(f"  {name}: already present, checksum ok")
            continue
        url = f"https://zenodo.org/records/{record}/files/{name}?download=1"
        _download(url, dest)
        got = _sha256(dest)
        if got != want:
            print(f"  {name}: CHECKSUM MISMATCH\n    want {want}\n    got  {got}")
            failures += 1
        else:
            print(f"  {name}: ok")
    if failures:
        print(f"\n{failures} file(s) failed.")
    return 1 if failures else 0


def make_manifest(directory: str) -> int:
    """Maintainer helper: hash every file in ``directory`` into a manifest skeleton."""
    d = Path(directory)
    files = [
        {"name": p.name, "sha256": _sha256(p), "dest": ".cache",
         "size_mb": round(p.stat().st_size / 1e6, 1)}
        for p in sorted(d.iterdir()) if p.is_file()
    ]
    skeleton = {"zenodo_record": "<fill in the Zenodo record id>", "files": files}
    print(json.dumps(skeleton, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + verify Zenodo-hosted release artifacts.")
    ap.add_argument("--make-manifest", metavar="DIR",
                    help="(maintainer) print a manifest with sha256 of every file in DIR")
    ns = ap.parse_args()
    sys.exit(make_manifest(ns.make_manifest) if ns.make_manifest else fetch())


if __name__ == "__main__":
    main()
