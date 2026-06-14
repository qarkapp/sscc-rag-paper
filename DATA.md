# Data and large artifacts

This repo ships the **benchmark** (questions, labels, datasheet, source *pointers*)
in git, and hosts **large convenience artifacts** on an external public link. The
copyrighted source corpus is **never redistributed** — it is reconstructed from the
original sources by the build script.

## What lives where

| Artifact | Location | Notes |
|---|---|---|
| Questions + span labels (`questions.jsonl`) | git (`data/hetdocqa/`) | your license (see below) |
| Collections + corpus **pointers** (`corpus_manifest.json`) | git | `source_ref` only, no content |
| Datasheet (`DATASHEET.md`) | git | Gebru-style |
| Per-query results (`results/*.json`) | git | metrics only, no passage text |
| Reconstructed corpus (PDFs, code, prose) | **not distributed** | rebuilt by `scripts/build_hetdocqa.py` |
| Vector store / call cache (`.cache/`) | **not distributed** | contains verbatim source text |
| Embedding cache (vectors only, optional) | pCloud public link | convenience; see below |

> **Why the cache is not published:** the LanceDB store and the reconstructed
> documents contain the verbatim source text of copyrighted PDFs, code, and prose.
> Publishing them would re-distribute content we deliberately keep pointer-only. Only
> derived numeric artifacts (embedding vectors) may be shared as a convenience.

## Reproducing from scratch (no download needed)

```bash
uv sync --extra index --extra graph --extra eval
cp .env.example .env            # add OMLX_API_KEY (local models) + OPENROUTER_API_KEY
uv run python scripts/build_hetdocqa.py     # reconstruct corpus from source pointers
uv run python scripts/run_benchmark.py hetdocqa test   # or run_retrieval_only.py
```

## Optional: fetch the large convenience bundle

Large numeric artifacts (optional, e.g. the embedding cache) are deposited on
**Zenodo** alongside the benchmark and verified by SHA-256. Put the record id in
[`data_manifest.json`](data_manifest.json).

```bash
# maintainer: hash the bundle and print a manifest skeleton
uv run python scripts/fetch_data.py --make-manifest path/to/bundle

# user: download + verify everything in the manifest
uv run python scripts/fetch_data.py
```

`scripts/fetch_data.py` downloads from the record's stable public URLs
(`zenodo.org/records/<id>/files/<name>`), so no account is required.

## Licensing

- **Code:** Apache-2.0 (`LICENSE`).
- **Benchmark annotations** (questions, answers, span labels): CC BY 4.0
  (`data/hetdocqa/LICENSE`).
- **Source corpus:** each document keeps its original license; see `license` /
  `source_ref` in `corpus_manifest.json`. We distribute pointers, not content.

## Persistent archive

The benchmark is deposited on **Zenodo** (free, CERN-run, DOI, versioned, up to
50 GB) for a citable, persistent snapshot: `data/hetdocqa/`, `results/`, the paper,
and the harness. The Zenodo DOI is the canonical citation; GitHub holds the working
copy.
