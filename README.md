# Beyond the Reranker

Benchmark, harness, and paper for a controlled study of retrieval-augmented
generation (RAG) enhancements on heterogeneous documents.

RAG is routinely extended with methods meant to improve retrieval: query
expansion, hierarchical and cross-document summarization, graph-based expansion,
per-query routing, rank fusion, and corrective re-retrieval. The evidence for
these methods comes almost entirely from homogeneous corpora, predominantly
Wikipedia prose. This repository tests whether they hold on mixed-format
collections, where code, tables, scientific PDFs, and prose are interleaved in one
corpus.

## What we find

On a shared backbone (one bi-encoder, one cross-encoder reranker, one generator,
all held fixed), eight methods are ablated one at a time across HetDocQA
(heterogeneous) and two homogeneous controls (MuSiQue, QASPER), with bootstrap
confidence intervals, Holm-Bonferroni correction, and a dev/test sign-consistency
rule.

- The cross-encoder reranker accounts for most of the pipeline's retrieval
  quality.
- Beyond the reranker, only two methods give a reliable gain: query expansion
  (HyDE) and SSCC, a per-source calibrated corrector introduced here that sets a
  separate acceptance threshold per score source and helps only on heterogeneous
  data.
- The reranking and pool-expansion methods in common use (hierarchical
  summarization, the cross-document tier, graph expansion, per-query routing, and
  rank fusion) give no reliable gain once a strong reranker is present, on
  heterogeneous and homogeneous data alike.

See [`paper/main.pdf`](paper/main.pdf) for the full study.

## HetDocQA

A question-answering benchmark over heterogeneous, multi-format document
collections (code, markdown, prose, tables, and PDFs).

- **Chunker-agnostic labels.** Gold evidence is annotated as character spans in the
  source documents and matched to each system's own chunks at evaluation time
  (>=50% span overlap), so retrieval metrics do not depend on how a system chunks
  the corpus.
- **Collection-disjoint splits.** Calibration, dev, and test share no collection, so
  a threshold tuned on dev cannot exploit corpus structure that recurs in test.
- **Pointers, not content.** The corpus is released as source pointers plus build
  scripts (`data/hetdocqa/corpus_manifest.json`), not as redistributed documents,
  so it reconstructs from the original sources under their own licenses.

Benchmark files are in [`data/hetdocqa/`](data/hetdocqa/); see the
[datasheet](data/hetdocqa/DATASHEET.md) and [DATA.md](DATA.md) for composition,
construction, and reproduction details.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Retrieval uses a local
[oMLX](http://127.0.0.1:1234) server (OpenAI-compatible) for embeddings and
reranking; generation uses an OpenRouter key.

```bash
uv sync --extra index --extra graph --extra eval
cp .env.example .env          # fill in OMLX_API_KEY and OPENROUTER_API_KEY
```

## Reproducing the study

```bash
# 1. Rebuild the corpus from source pointers and assemble the benchmark
uv run python scripts/build_hetdocqa.py

# 2. Full ablation with answer metrics (retrieval + EM/F1)
uv run python scripts/run_benchmark.py hetdocqa test

# 2b. Fast retrieval-only ablation (nDCG / Success@10, no generation)
uv run python scripts/run_retrieval_only.py hetdocqa dev
```

Backend calls (embeddings, reranks, generations) are content-addressed and cached
on disk, so a full ablation suite replays offline and deterministically. The cache
is not redistributed (it contains the verbatim source corpus); see
[DATA.md](DATA.md).

## Repository layout

```
data/hetdocqa/   the benchmark: questions, span labels, datasheet, source pointers
paper/           the paper (main.tex, figures, references)
results/         per-query retrieval and answer metrics (no passage text)
scripts/         build, run, retrieval-only, and data-fetch entry points
src/sage/        the shared retrieval harness used in the study
  core/          types, protocols, registry
  config/        configuration schema, presets, seeding
  cache/         content-addressed call cache
  clients/       oMLX / OpenRouter backends
  raptor/        hierarchical clustering, tree retrieval, cross-document tier
  chunking/      prose and code (AST) chunking
  routing/       strategy routing (keyword baseline + entropy-gated)
  strategies/    semantic, HyDE, step-back, dual-path fusion
  correction/    CRAG and score-source-calibrated correction (SSCC)
  graph/         typed chunk graph, GraphSAGE refinement, PageRank expansion
  pipeline/      indexing and retrieval pipeline assembly
  eval/          benchmark loaders, metrics, statistics, experiment runner
```

Every component is reached through a small set of protocols
(`sage.core.protocols`) and assembled from configuration, so any method can be
toggled or swapped for an ablation without changing code.

## Development

```bash
uv run pytest            # tests (offline, deterministic)
uv run ruff check .      # lint
uv run mypy              # type-check
```

## License

- **Code:** Apache-2.0. See [LICENSE](LICENSE).
- **HetDocQA annotations** (questions, answers, span labels, splits, datasheet):
  CC BY 4.0. See [data/hetdocqa/LICENSE](data/hetdocqa/LICENSE).
- **Source documents:** not redistributed; each retains its original license, as
  recorded in `corpus_manifest.json`.

## Citation

Benchmark, code, and paper are archived on Zenodo (DOI
[10.5281/zenodo.20693143](https://doi.org/10.5281/zenodo.20693143), all versions).

```bibtex
@misc{beyondthereranker,
  title     = {Beyond the Reranker: Do RAG Retrieval Enhancements Help Once a
               Strong Reranker Is Present?},
  author    = {Singh, Sadanand and Reddy, Allam and Chopra, Manan},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Cascade Research},
  doi       = {10.5281/zenodo.20693143},
  url       = {https://doi.org/10.5281/zenodo.20693143}
}
```
