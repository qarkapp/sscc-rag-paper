"""Build the full HetDocQA candidate set.

Assembles ~50 mixed-format collections from public sources (arXiv PDFs paired with
topical Wikipedia prose, permissive GitHub repos for code + markdown, and open CSVs
for tabular data), then drafts and filters questions with DeepSeek.

Drafting uses ``deepseek-v4-pro`` with thinking disabled (fast, cheap); cross-
validation uses a distinct model (``deepseek-reasoner``) for an independent check.
Run from the repo root with the DeepSeek key in ``.env``:

    uv run python scripts/build_hetdocqa.py
"""

from __future__ import annotations

import asyncio
import json
import time

from sage.cache.store import CacheMode, CallCache
from sage.clients.base import BackendConfig
from sage.clients.embedder import OpenAICompatEmbedder
from sage.clients.generator import ChatGenerator
from sage.hetdocqa import (
    Collection,
    SourceDoc,
    build_candidates,
    dataset_stats,
    fetch_arxiv_pdf,
    fetch_csv,
    fetch_github_file,
    fetch_wikipedia,
    write_release,
    write_review_bundle,
)

# Curated, well-known arXiv ids (avoids the rate-limited search API), each paired
# with a topical Wikipedia article so the collection mixes a PDF with prose.
ARXIV_PAPERS = [
    ("2005.11401", "Retrieval-augmented generation"),  # RAG
    ("2004.04906", "Information retrieval"),  # DPR
    ("2212.10496", "Information retrieval"),  # HyDE
    ("2401.18059", "Retrieval-augmented generation"),  # RAPTOR
    ("2401.15884", "Retrieval-augmented generation"),  # CRAG
    ("2310.11511", "Retrieval-augmented generation"),  # Self-RAG
    ("2112.09118", "Information retrieval"),  # Contriever
    ("2004.12832", "Information retrieval"),  # ColBERT
    ("1706.02216", "Graph neural network"),  # GraphSAGE
    ("1609.02907", "Graph neural network"),  # GCN
    ("1710.10903", "Graph neural network"),  # GAT
    ("1810.00826", "Graph neural network"),  # GIN
    ("1706.03762", "Transformer (deep learning architecture)"),  # Attention is all you need
    ("1810.04805", "BERT (language model)"),  # BERT
    ("2005.14165", "GPT-3"),  # GPT-3
    ("1907.11692", "BERT (language model)"),  # RoBERTa
    ("1802.03426", "Dimensionality reduction"),  # UMAP
    ("1908.10084", "Word embedding"),  # Sentence-BERT
    ("2212.03533", "Word embedding"),  # E5
    ("1310.4546", "Word embedding"),  # word2vec
    ("1809.09600", "Question answering"),  # HotpotQA
    ("2108.00573", "Question answering"),  # MuSiQue
    ("1705.03551", "Question answering"),  # TriviaQA
    ("1412.6980", "Stochastic gradient descent"),  # Adam
    ("1512.03385", "Residual neural network"),  # ResNet
    ("1406.2661", "Generative adversarial network"),  # GAN
    ("1312.6114", "Variational autoencoder"),  # VAE
    ("2403.05530", "Large language model"),  # Gemini 1.5
]
ARXIV_THROTTLE_S = 2.0  # be polite to arXiv between PDF downloads

# (collection_id, owner, repo, ref, [paths], license, wiki article)
REPOS = [
    (
        "requests",
        "psf",
        "requests",
        "main",
        ["README.md", "src/requests/api.py", "src/requests/sessions.py"],
        "Apache-2.0",
        "Hypertext Transfer Protocol",
    ),
    (
        "flask",
        "pallets",
        "flask",
        "main",
        ["README.md", "src/flask/app.py"],
        "BSD-3-Clause",
        "Web framework",
    ),
    (
        "click",
        "pallets",
        "click",
        "main",
        ["README.md", "src/click/core.py"],
        "BSD-3-Clause",
        "Command-line interface",
    ),
    (
        "fastapi",
        "fastapi",
        "fastapi",
        "master",
        ["README.md", "fastapi/applications.py"],
        "MIT",
        "Web API",
    ),
    (
        "httpx",
        "encode",
        "httpx",
        "master",
        ["README.md", "httpx/_client.py"],
        "BSD-3-Clause",
        "HTTP",
    ),
    (
        "pydantic",
        "pydantic",
        "pydantic",
        "main",
        ["README.md", "pydantic/main.py"],
        "MIT",
        "Data validation",
    ),
    (
        "rich",
        "Textualize",
        "rich",
        "master",
        ["README.md", "rich/console.py"],
        "MIT",
        "Terminal emulator",
    ),
]

# (collection_id, csv_url, name, wiki article)
CSVS = [
    (
        "iris",
        "https://raw.githubusercontent.com/plotly/datasets/master/iris.csv",
        "iris.csv",
        "Iris flower data set",
    ),
    (
        "tips",
        "https://raw.githubusercontent.com/plotly/datasets/master/tips.csv",
        "tips.csv",
        "Gratuity",
    ),
    (
        "gapminder",
        "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv",
        "gapminder.csv",
        "Gapminder Foundation",
    ),
    (
        "ag-exports",
        "https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv",
        "ag_exports.csv",
        "Agriculture in the United States",
    ),
]

PLOTLY_LICENSE = "MIT"


def assemble() -> tuple[list[Collection], dict[str, list[SourceDoc]]]:
    collections: list[Collection] = []
    docs: dict[str, list[SourceDoc]] = {}

    def register(cid: str, title: str, fetched: list[SourceDoc | None]) -> None:
        got = [d for d in fetched if d is not None]
        if len(got) >= 2:  # a collection needs at least two documents
            docs[cid] = got
            collections.append(Collection(cid, title, tuple(d.doc_id for d in got)))

    # arXiv + Wikipedia (curated ids, throttled downloads)
    for arxiv_id, wiki in ARXIV_PAPERS:
        cid = f"arxiv-{arxiv_id.replace('/', '_')}"
        register(
            cid,
            f"{wiki} ({arxiv_id})",
            [
                fetch_arxiv_pdf(cid, arxiv_id, "arXiv (see source_ref)"),
                fetch_wikipedia(cid, wiki),
            ],
        )
        time.sleep(ARXIV_THROTTLE_S)

    # GitHub code + markdown + Wikipedia
    for cid, owner, repo, ref, paths, lic, wiki in REPOS:
        fetched = [fetch_github_file(cid, owner, repo, ref, p, lic) for p in paths]
        fetched.append(fetch_wikipedia(cid, wiki))
        register(cid, f"{repo} + {wiki}", fetched)

    # CSV + Wikipedia
    for cid, url, name, wiki in CSVS:
        register(
            cid,
            f"{name} + {wiki}",
            [
                fetch_csv(cid, url, name, PLOTLY_LICENSE),
                fetch_wikipedia(cid, wiki),
            ],
        )

    return collections, docs


async def main() -> None:
    t0 = time.time()
    collections, docs = assemble()
    print(
        f"assembled {len(collections)} collections, "
        f"{sum(len(v) for v in docs.values())} docs ({time.time() - t0:.0f}s)"
    )

    cache = CallCache(".cache/hetdoc", CacheMode.READ_WRITE)
    drafter = ChatGenerator(
        BackendConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            timeout=120,
            extra_body={"thinking": {"type": "disabled"}},
        ),
        cache,
    )
    validator = ChatGenerator(
        BackendConfig(provider="deepseek", model="deepseek-reasoner", timeout=180), cache
    )
    embedder = OpenAICompatEmbedder(BackendConfig(provider="omlx", model="bge-m3-mlx-fp16"), cache)

    candidates = await build_candidates(
        collections,
        docs,
        generator=drafter,
        validator=validator,
        embedder=embedder,
        per_type=8,
        concurrency=8,
    )
    clean = [c for c in candidates if c.is_clean]
    print(f"\n{len(candidates)} candidates, {len(clean)} clean ({time.time() - t0:.0f}s)")
    print("stats:", json.dumps(dataset_stats(clean)))

    all_docs = [d for v in docs.values() for d in v]
    write_release("data/hetdocqa", candidates, collections, all_docs)
    # Local-only review bundle (embeds evidence text for the human-review app).
    write_review_bundle("data/hetdocqa/review.json", candidates, all_docs)
    print(
        "wrote data/hetdocqa/{questions.jsonl,DATASHEET.md,collections.json,"
        "corpus_manifest.json,review.json}"
    )


if __name__ == "__main__":
    asyncio.run(main())
