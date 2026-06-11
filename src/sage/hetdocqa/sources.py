"""Source fetchers for HetDocQA collections.

Each fetcher returns a :class:`SourceDoc` with a reproducible ``source_ref`` pointer
(repo@sha:path, arXiv id, dataset URL) so the corpus can be rebuilt from the released
manifest without redistributing third-party content. Fetchers return ``None`` on
failure so collection assembly is resilient to transient network errors.
"""

from __future__ import annotations

import csv
import io
import re

import httpx

from sage.chunking.languages import is_code_file
from sage.hetdocqa.schema import Modality, SourceDoc

__all__ = [
    "arxiv_search",
    "fetch_arxiv_pdf",
    "fetch_csv",
    "fetch_github_file",
    "fetch_wikipedia",
]

_TIMEOUT = 30.0
_MAX_CHARS = 20000


def _truncate(text: str) -> str:
    return text[:_MAX_CHARS]


def _get(url: str, *, retries: int = 4, **kwargs: object) -> httpx.Response | None:
    """GET with retry/backoff on rate limits and transient errors."""
    import time

    kwargs.setdefault("timeout", _TIMEOUT)
    kwargs.setdefault("follow_redirects", True)
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = httpx.get(url, **kwargs)  # type: ignore[arg-type]
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = float(resp.headers.get("retry-after", delay))
                time.sleep(min(wait, 30.0))
                delay *= 2
                continue
            resp.raise_for_status()
            return resp
        except httpx.HTTPError:
            if attempt == retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def arxiv_search(query: str, max_results: int = 10) -> list[str]:
    """Return arXiv ids matching a query (via the public arXiv API)."""
    try:
        resp = httpx.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except httpx.HTTPError:
        return []
    ids = re.findall(r"<id>http://arxiv\.org/abs/([^<]+)</id>", resp.text)
    # The first <id> is the feed itself; entry ids carry a version suffix.
    return [i.split("v")[0] if re.search(r"v\d+$", i) else i for i in ids]


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
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": title,
        "redirects": "1",
    }
    resp = _get(
        "https://en.wikipedia.org/w/api.php",
        params=params,
        headers={"User-Agent": "sage-hetdocqa/0.1"},
    )
    if resp is None:
        return None
    try:
        pages = resp.json()["query"]["pages"]
        extract = next(iter(pages.values())).get("extract", "")
    except (KeyError, ValueError):
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
    resp = _get(f"https://arxiv.org/pdf/{arxiv_id}")
    if resp is None:
        return None
    try:
        import pymupdf

        with pymupdf.open(stream=resp.content, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
    except Exception:
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
