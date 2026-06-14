"""Fetch large release artifacts from a pCloud public link and verify checksums.

Large artifacts (e.g. the embedding cache) are hosted on a pCloud *public* folder
link rather than in git. This script resolves that link through the pCloud API,
downloads each expected file, and verifies its SHA-256 against ``data_manifest.json``
so a replaced or corrupted file is caught.

It does NOT need a pCloud account or key -- a public link is enough.

    uv run python scripts/fetch_data.py                 # download + verify all
    uv run python scripts/fetch_data.py --dest .cache    # custom destination
    uv run python scripts/fetch_data.py --make-manifest <dir>   # (maintainer) hash local files

The manifest (``data_manifest.json`` at the repo root) looks like:

    {
      "pcloud_code": "XXXXXXX",          # the code= from the public link
      "region": "us",                     # "us" (api.pcloud.com) or "eu" (eapi.pcloud.com)
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
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = Path(__file__).resolve().parent.parent / "data_manifest.json"
_HOSTS = {"us": "https://api.pcloud.com", "eu": "https://eapi.pcloud.com"}
_CHUNK = 1 << 20


def _api(host: str, method: str, **params: str) -> dict:
    url = f"{host}/{method}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted host)
        return json.loads(resp.read().decode())


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
    code = man.get("pcloud_code")
    if not code or code.startswith("<"):
        sys.exit("manifest has no real pcloud_code yet -- create the public link and fill it in")
    host = _HOSTS.get(man.get("region", "us"), _HOSTS["us"])

    # Resolve the public link to a {name -> fileid} map.
    info = _api(host, "showpublink", code=code)
    if info.get("result", 0) != 0:
        sys.exit(f"showpublink failed: {info.get('error', info)}")
    meta = info["metadata"]
    contents = meta.get("contents", [meta])  # single-file links have no 'contents'
    by_name = {c["name"]: c["fileid"] for c in contents if not c.get("isfolder")}

    failures = 0
    for entry in man["files"]:
        name, want = entry["name"], entry["sha256"]
        dest = Path(entry.get("dest", ".")) / name
        if dest.exists() and _sha256(dest) == want:
            print(f"  {name}: already present, checksum ok")
            continue
        fid = by_name.get(name)
        if fid is None:
            print(f"  {name}: NOT FOUND in public link")
            failures += 1
            continue
        link = _api(host, "getpublinkdownload", code=code, fileid=str(fid))
        if link.get("result", 0) != 0:
            print(f"  {name}: getpublinkdownload failed: {link.get('error', link)}")
            failures += 1
            continue
        url = "https://" + link["hosts"][0] + link["path"]
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
    skeleton = {"pcloud_code": "<fill in the code= from the pCloud public link>",
                "region": "us", "files": files}
    print(json.dumps(skeleton, indent=2))
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + verify pCloud-hosted release artifacts.")
    ap.add_argument("--make-manifest", metavar="DIR",
                    help="(maintainer) print a manifest with sha256 of every file in DIR")
    ap.parse_args(namespace=(ns := argparse.Namespace()))
    sys.exit(make_manifest(ns.make_manifest) if ns.make_manifest else fetch())


if __name__ == "__main__":
    main()
