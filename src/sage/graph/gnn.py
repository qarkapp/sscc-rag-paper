"""GraphSAGE embedding refinement with an unsupervised objective.

A typed GraphSAGE (one aggregator per edge type, summed) refines chunk embeddings
using the chunk graph. There are no labels, so training uses the GraphSAGE
neighbour-sampling objective: connected chunks should have similar refined
embeddings, random pairs should not.

Design choice: the network projects its output back into the original embedding
space (R^d) rather than concatenating, so refined chunk vectors remain directly
comparable to query embeddings with no separate query projection. Residual
connections, layer norm, and edge dropout control over-smoothing. With ``gnn_layers
= 0`` refinement is the identity (the ablation baseline), and late fusion with
``beta = 0`` recovers the bi-encoder exactly.

torch is imported lazily so that ``sage.graph.build`` and ``sage.graph.ppr`` (which
need no deep-learning stack) stay importable without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from sage.config.schema import GraphCfg
from sage.graph.build import ChunkGraph

__all__ = ["late_fusion_score", "refine_embeddings"]


def refine_embeddings(
    graph: ChunkGraph, cfg: GraphCfg, edge_types: Sequence[str], *, seed: int = 42
) -> np.ndarray:
    """Return GraphSAGE-refined embeddings in the original embedding space."""
    embeddings = np.asarray(graph.embeddings, dtype=np.float32)
    if cfg.gnn_layers <= 0 or embeddings.shape[0] < 3:
        return embeddings  # identity refinement (ablation / too small to train)

    import torch

    torch.manual_seed(seed)
    edge_index = _edge_index_by_type(graph, edge_types)
    if not edge_index:
        return embeddings

    model = _make_model(
        in_dim=embeddings.shape[1],
        hidden=cfg.gnn_hidden,
        layers=cfg.gnn_layers,
        edge_types=list(edge_index),
        dropout=0.2,
    )
    x = torch.from_numpy(embeddings)
    _train_unsupervised(model, x, edge_index, seed=seed)
    model.eval()
    with torch.no_grad():
        refined = model(x, edge_index).cpu().numpy()
    return np.asarray(refined, dtype=np.float32)


def late_fusion_score(
    query_vector: np.ndarray,
    base_embedding: np.ndarray,
    refined_embedding: np.ndarray,
    beta: float,
) -> float:
    """Blend bi-encoder and graph-refined cosine similarity (``beta=0`` -> baseline)."""
    return (1.0 - beta) * _cos(query_vector, base_embedding) + beta * _cos(
        query_vector, refined_embedding
    )


# -- internals -------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _edge_index_by_type(graph: ChunkGraph, edge_types: Sequence[str]) -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {}
    for et in edge_types:
        edges = graph.edges.get(et, set())
        if not edges:
            continue
        src = [a for a, _ in edges] + [b for _, b in edges]  # undirected -> both ways
        dst = [b for _, b in edges] + [a for a, _ in edges]
        out[et] = torch.tensor([src, dst], dtype=torch.long)
    return out


def _make_model(
    *, in_dim: int, hidden: int, layers: int, edge_types: list[str], dropout: float
) -> Any:
    import torch
    from torch import nn
    from torch_geometric.nn import SAGEConv

    class TypedGraphSAGE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.edge_types = edge_types
            self.dropout = dropout
            self.in_proj = nn.Linear(in_dim, hidden)
            self.convs = nn.ModuleList(
                nn.ModuleDict({et: SAGEConv(hidden, hidden) for et in edge_types})
                for _ in range(layers)
            )
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
            self.out_proj = nn.Linear(hidden, in_dim)

        def forward(self, x: Any, edge_index: dict[str, Any]) -> Any:
            h = self.in_proj(x)
            for conv_dict, norm in zip(self.convs, self.norms, strict=True):
                convs: Any = conv_dict  # ModuleDict indexing is dynamic
                agg = torch.zeros_like(h)
                for et in self.edge_types:
                    if et in edge_index:
                        agg = agg + convs[et](h, edge_index[et])
                h = norm(h + torch.relu(agg))  # residual + norm control over-smoothing
                h = torch.dropout(h, self.dropout, train=self.training)
            return self.out_proj(h)

    return TypedGraphSAGE()


def _train_unsupervised(
    model: Any, x: Any, edge_index: dict[str, Any], *, seed: int, epochs: int = 100
) -> None:
    import torch
    import torch.nn.functional as functional

    generator = torch.Generator().manual_seed(seed)
    all_src = torch.cat([ei[0] for ei in edge_index.values()])
    all_dst = torch.cat([ei[1] for ei in edge_index.values()])
    n = x.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model(x, edge_index)
        pos = (z[all_src] * z[all_dst]).sum(-1)
        neg_dst = torch.randint(0, n, (all_src.shape[0],), generator=generator)
        neg = (z[all_src] * z[neg_dst]).sum(-1)
        loss = -functional.logsigmoid(pos).mean() - functional.logsigmoid(-neg).mean()
        loss.backward()
        optimizer.step()
