"""Source fetchers for HetDocQA collections.

Each fetcher returns a :class:`SourceDoc` with a reproducible ``source_ref`` pointer
(repo@sha:path, arXiv id, dataset URL) so the corpus can be rebuilt from the released
manifest without redistributing third-party content. Fetchers return ``None`` on
failure so collection assembly is resilient to transient network errors.
"""

from __future__ import annotations

import csv
import io

import httpx

from sage.chunking.languages import is_code_file
from sage.hetdocqa.schema import Modality, SourceDoc

__all__ = [
    "fetch_arxiv_pdf",
    "fetch_csv",
    "fetch_github_file",
    "fetch_wikipedia",
]

_TIMEOUT = 30.0
_MAX_CHARS = 20000


def _truncate(text: str) -> str:
    return text[:_MAX_CHARS]


def fetch_github_file(
    collection_id: str, owner: str, repo: str, ref: str, path: str, license: str
) -> SourceDoc | None:
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    modality = Modality.CODE if is_code_file(path) else Modality.MARKDOWN
    return SourceDoc(
        doc_id=f"{collection_id}:{path}",
        collection_id=collection_id,
        filename=path.rsplit("/", 1)[-1],
        text=_truncate(resp.text),
        modality=modality,
        source_ref=f"github://{owner}/{repo}@{ref}:{path}",
        license=license,
    )


def fetch_wikipedia(collection_id: str, title: str) -> SourceDoc | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": title,
        "redirects": "1",
    }
    try:
        resp = httpx.get(
            url, params=params, timeout=_TIMEOUT, headers={"User-Agent": "sage-hetdocqa/0.1"}
        )
        resp.raise_for_status()
        pages = resp.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
    except (httpx.HTTPError, KeyError, ValueError):
        return None
    if not extract:
        return None
    return SourceDoc(
        doc_id=f"{collection_id}:wiki-{title.replace(' ', '_')}",
        collection_id=collection_id,
        filename=f"{title}.txt",
        text=_truncate(extract),
        modality=Modality.PROSE,
        source_ref=f"wikipedia://{title}",
        license="CC-BY-SA-4.0",
    )


def fetch_csv(
    collection_id: str, url: str, name: str, license: str, *, max_rows: int = 60
) -> SourceDoc | None:
    try:
        resp = httpx.get(url, timeout=_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    reader = csv.reader(io.StringIO(resp.text))
    rows = [", ".join(row) for _, row in zip(range(max_rows), reader, strict=False)]
    if not rows:
        return None
    return SourceDoc(
        doc_id=f"{collection_id}:{name}",
        collection_id=collection_id,
        filename=name,
        text=_truncate("\n".join(rows)),
        modality=Modality.TABLE,
        source_ref=f"csv://{url}",
        license=license,
    )


def fetch_arxiv_pdf(collection_id: str, arxiv_id: str, license: str) -> SourceDoc | None:
    try:
        resp = httpx.get(
            f"https://arxiv.org/pdf/{arxiv_id}", timeout=_TIMEOUT, follow_redirects=True
        )
        resp.raise_for_status()
        import pymupdf

        with pymupdf.open(stream=resp.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception:  # noqa: BLE001 - network or PDF-parse failures are non-fatal
        return None
    if not text.strip():
        return None
    return SourceDoc(
        doc_id=f"{collection_id}:arxiv-{arxiv_id}",
        collection_id=collection_id,
        filename=f"{arxiv_id}.pdf",
        text=_truncate(text),
        modality=Modality.PDF,
        source_ref=f"arxiv://{arxiv_id}",
        license=license,
    )
