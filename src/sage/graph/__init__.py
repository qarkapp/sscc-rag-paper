"""Graph-augmented hierarchical retrieval (GAHR).

Typed chunk graph + GraphSAGE embedding refinement + Personalized PageRank
expansion. ``build`` and ``ppr`` need no deep-learning stack; ``gnn`` imports torch
lazily.
"""

from sage.graph.build import EDGE_TYPES, ChunkGraph, build_chunk_graph
from sage.graph.nli_edges import (
    EntailmentEdge,
    EntailmentLabel,
    build_entailment_edges,
)
from sage.graph.ppr import expand_by_ppr, personalized_pagerank
from sage.graph.traversal import entailment_chains

__all__ = [
    "EDGE_TYPES",
    "ChunkGraph",
    "EntailmentEdge",
    "EntailmentLabel",
    "build_chunk_graph",
    "build_entailment_edges",
    "entailment_chains",
    "expand_by_ppr",
    "personalized_pagerank",
]
