"""Typed chunk-graph construction.

Four edge types connect chunks:

* ``sequential`` -- consecutive chunks within a document.
* ``semantic``   -- chunk pairs whose embeddings exceed a cosine threshold.
* ``xref``       -- chunks that share an explicit cross-reference marker
  (e.g. ``Section 3``, ``Figure 2``, ``[12]``).
* ``ast``        -- code chunks in the same file that share a defined identifier.

Edges are undirected and deduplicated. Semantic-edge construction is O(n^2) in the
number of chunks; for very large corpora it should be batched or approximated, which
is noted where it is built.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from sage.core.types import Chunk

__all__ = ["EDGE_TYPES", "ChunkGraph", "build_chunk_graph"]

EDGE_TYPES = ("sequential", "semantic", "xref", "ast")

_XREF = re.compile(r"\b(?:section|fig(?:ure)?|table|chapter|eq(?:uation)?)\s+\d+\b", re.I)
_CITATION = re.compile(r"\[\d+\]")
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")


@dataclass(slots=True)
class ChunkGraph:
    chunk_ids: list[str]
    embeddings: np.ndarray
    edges: dict[str, set[tuple[int, int]]] = field(default_factory=dict)

    @property
    def index_of(self) -> dict[str, int]:
        return {cid: i for i, cid in enumerate(self.chunk_ids)}

    def active_edges(self, edge_types: Sequence[str]) -> set[tuple[int, int]]:
        out: set[tuple[int, int]] = set()
        for et in edge_types:
            out |= self.edges.get(et, set())
        return out

    def neighbors(self, idx: int, edge_types: Sequence[str]) -> set[int]:
        result: set[int] = set()
        for a, b in self.active_edges(edge_types):
            if a == idx:
                result.add(b)
            elif b == idx:
                result.add(a)
        return result


def _undirected(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def build_chunk_graph(
    chunks: Sequence[Chunk], embeddings: np.ndarray, *, semantic_threshold: float = 0.7
) -> ChunkGraph:
    """Construct a typed chunk graph from chunks and their embeddings."""
    chunk_ids = [c.chunk_id for c in chunks]
    graph = ChunkGraph(chunk_ids=chunk_ids, embeddings=np.asarray(embeddings))
    graph.edges = {
        "sequential": _sequential_edges(chunks),
        "semantic": _semantic_edges(embeddings, semantic_threshold),
        "xref": _xref_edges(chunks),
        "ast": _ast_edges(chunks),
    }
    return graph


def _sequential_edges(chunks: Sequence[Chunk]) -> set[tuple[int, int]]:
    by_doc: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        by_doc.setdefault(c.document_id, []).append(i)
    edges: set[tuple[int, int]] = set()
    for indices in by_doc.values():
        ordered = sorted(indices, key=lambda i: chunks[i].chunk_index)
        for a, b in itertools.pairwise(ordered):
            edges.add(_undirected(a, b))
    return edges


def _semantic_edges(embeddings: np.ndarray, threshold: float) -> set[tuple[int, int]]:
    n = embeddings.shape[0]
    if n < 2:
        return set()
    norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sims = norm @ norm.T
    edges: set[tuple[int, int]] = set()
    rows, cols = np.where(np.triu(sims, k=1) > threshold)
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        edges.add((int(i), int(j)))
    return edges


def _markers(text: str) -> set[str]:
    return {m.lower() for m in _XREF.findall(text)} | set(_CITATION.findall(text))


def _xref_edges(chunks: Sequence[Chunk]) -> set[tuple[int, int]]:
    marker_to_nodes: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        for marker in _markers(c.content):
            marker_to_nodes.setdefault(marker, []).append(i)
    edges: set[tuple[int, int]] = set()
    for nodes in marker_to_nodes.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                edges.add(_undirected(nodes[a], nodes[b]))
    return edges


def _ast_edges(chunks: Sequence[Chunk]) -> set[tuple[int, int]]:
    # Code chunks in the same file that share a defined identifier.
    code = [(i, c) for i, c in enumerate(chunks) if c.language]
    ident_to_nodes: dict[tuple[str, str], list[int]] = {}
    for i, c in code:
        for ident in set(_IDENTIFIER.findall(c.content)):
            ident_to_nodes.setdefault((c.document_id, ident), []).append(i)
    edges: set[tuple[int, int]] = set()
    for nodes in ident_to_nodes.values():
        if len(nodes) < 2 or len(nodes) > 12:  # skip ubiquitous identifiers
            continue
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                edges.add(_undirected(nodes[a], nodes[b]))
    return edges
