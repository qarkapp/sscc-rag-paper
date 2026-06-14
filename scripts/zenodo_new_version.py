"""Publish a new version of an existing Zenodo record with a replacement file.

Run this LOCALLY with your own token; never paste the token into shared logs.

    export ZENODO_TOKEN=...        # personal token, scopes: deposit:write deposit:actions
    uv run python scripts/zenodo_new_version.py \\
        --record 20693144 --file ~/Downloads/sscc-rag-paper-zenodo.zip --version v1.1

The concept DOI (10.5281/zenodo.20693143) is preserved; a new version DOI is minted.
Use --sandbox to test against sandbox.zenodo.org first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", required=True, help="latest published version record id")
    ap.add_argument("--file", required=True, help="path to the replacement archive")
    ap.add_argument("--version", required=True, help="new version label, e.g. v1.1")
    ap.add_argument("--sandbox", action="store_true", help="use sandbox.zenodo.org")
    ns = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("set ZENODO_TOKEN in your environment (do not paste it into shared logs)")
    path = Path(ns.file).expanduser()
    if not path.exists():
        sys.exit(f"no such file: {path}")

    base = "https://sandbox.zenodo.org/api" if ns.sandbox else "https://zenodo.org/api"
    params = {"access_token": token}
    cli = httpx.Client(timeout=300, params=params)

    def check(r: httpx.Response, what: str) -> dict:
        if r.status_code >= 300:
            sys.exit(f"{what} failed [{r.status_code}]: {r.text[:300]}")
        return r.json() if r.content else {}

    print(f"new version of record {ns.record} ...")
    nv = check(cli.post(f"{base}/deposit/depositions/{ns.record}/actions/newversion"),
               "newversion")
    draft = check(cli.get(nv["links"]["latest_draft"]), "get draft")
    did, bucket = draft["id"], draft["links"]["bucket"]
    print(f"  draft deposition {did}")

    for f in draft.get("files", []):  # drop inherited files so only the new one remains
        check(cli.delete(f"{base}/deposit/depositions/{did}/files/{f['id']}"), "delete old file")
        print(f"  removed inherited {f.get('filename', f['id'])}")

    with path.open("rb") as fh:
        check(cli.put(f"{bucket}/{path.name}", content=fh.read()), "upload")
    print(f"  uploaded {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

    meta = draft.get("metadata", {})
    meta["version"] = ns.version
    check(cli.put(f"{base}/deposit/depositions/{did}",
                  json={"metadata": meta}), "update metadata")

    pub = check(cli.post(f"{base}/deposit/depositions/{did}/actions/publish"), "publish")
    print(f"published: {pub.get('doi_url', pub.get('links', {}).get('record_html', 'ok'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
