# SAGE

A compositional retrieval-augmented generation framework. SAGE treats multi-strategy
retrieval as a per-query composition problem: it routes each query to an appropriate
retrieval strategy, fuses multiple hypotheses, calibrates post-retrieval correction to
the score source, and augments retrieval with graph-structured and hierarchical
indexing.

## Status

Under active development. The package is being built in phases.

## Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- An [oMLX](http://127.0.0.1:1234) server (OpenAI-compatible) for local embeddings and
  reranking; optionally an OpenRouter key for stronger generation models

## Setup

```bash
uv sync                 # base + dev dependencies
uv sync --extra index   # add the indexing stack (LanceDB, UMAP, sklearn, chunkers)
uv sync --extra graph   # add the graph stack (torch, torch-geometric)
uv sync --extra eval    # add the evaluation stack (ir-measures, datasets, ragas)

cp .env.example .env     # then fill in OMLX_API_KEY
```

## Development

```bash
uv run pytest            # tests (fully offline via deterministic fakes)
uv run ruff check .      # lint
uv run ruff format .     # format
uv run mypy              # type-check
```

## Architecture

Every component is reached through a small set of protocols (`sage.core.protocols`)
and assembled from configuration, so any component can be toggled or swapped for an
ablation without changing code. Backend calls (embeddings, reranking, generation) are
content-addressed and cached on disk, so experiments re-run offline and deterministically.

```
sage/
  core/        types, protocols, registry
  config/      configuration schema, presets, seeding
  cache/       content-addressed call cache
  clients/     oMLX / OpenRouter backends (the only place the OpenAI SDK is used)
  raptor/      hierarchical clustering + tree retrieval + cross-document tier
  chunking/    prose and code (AST) chunking + adaptive chunking
  routing/     strategy routing (keyword baseline + entropy-gated)
  strategies/  semantic, HyDE, step-back, dual-path fusion
  correction/  CRAG + score-source-calibrated correction
  graph/       typed chunk graph, GraphSAGE refinement, PageRank expansion, NLI edges
  prefetch/    speculative retrieval prefetching
  pipeline/    indexing and retrieval pipeline assembly
  eval/        benchmark loaders, metrics, experiment runner
```

## License

Apache-2.0. See [LICENSE](LICENSE).
