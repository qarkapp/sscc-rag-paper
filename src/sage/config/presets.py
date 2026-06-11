"""Named configuration presets and the ablation registry.

A preset is a function ``PipelineConfig -> PipelineConfig`` that returns a modified
copy. Each ablation row in the paper corresponds to one entry in :data:`ABLATIONS`,
so the experiment runner can enumerate them mechanically.

Conventions:
    * ``full``        -- every contribution enabled (the reference system)
    * ``baseline``    -- the simple, faithful reproduction of the reference app
                         (keyword routing, merge-dedup fusion, single-threshold CRAG,
                         no graph) -- the starting point the contributions improve on
    * ``semantic_only`` -- dense retrieval with no composition (the lower bound)
"""

from __future__ import annotations

from collections.abc import Callable

from sage.config.schema import PipelineConfig

__all__ = ["ABLATIONS", "apply", "baseline", "full", "semantic_only"]

Preset = Callable[[PipelineConfig], PipelineConfig]


def _copy(cfg: PipelineConfig) -> PipelineConfig:
    return cfg.model_copy(deep=True)


def full(cfg: PipelineConfig) -> PipelineConfig:
    """Enable every contribution."""
    out = _copy(cfg)
    out.router.variant = "egr"
    out.fusion.variant = "rrf"
    out.correction.variant = "sscc"
    out.graph.enabled = True
    out.raptor.enabled = True
    out.raptor.cross_doc = True
    out.prefetch.enabled = True
    out.raac.enabled = True
    if "nli" not in out.graph.edges:
        out.graph.edges = [*out.graph.edges, "nli"]
    return out


def baseline(cfg: PipelineConfig) -> PipelineConfig:
    """Faithful reproduction of the reference app's behaviour."""
    out = _copy(cfg)
    out.router.variant = "keyword"
    out.fusion.variant = "merge_dedup"
    out.correction.variant = "crag"
    out.graph.enabled = False
    out.prefetch.enabled = False
    out.raac.enabled = False
    return out


def semantic_only(cfg: PipelineConfig) -> PipelineConfig:
    """Dense retrieval with no composition."""
    out = _copy(cfg)
    out.router.enabled = False
    out.router.variant = "keyword"
    out.fusion.variant = "merge_dedup"
    out.rerank.enabled = False
    out.correction.enabled = False
    out.graph.enabled = False
    out.raptor.enabled = False
    out.prefetch.enabled = False
    out.raac.enabled = False
    return out


def _disable_graph(cfg: PipelineConfig) -> PipelineConfig:
    out = _copy(cfg)
    out.graph.enabled = False
    return out


def _drop_edge(edge: str) -> Preset:
    def preset(cfg: PipelineConfig) -> PipelineConfig:
        out = _copy(cfg)
        out.graph.edges = [e for e in out.graph.edges if e != edge]
        return out

    return preset


def _set(**changes: object) -> Preset:
    """Build a preset that applies dotted-path assignments, e.g. ``router.variant``."""

    def preset(cfg: PipelineConfig) -> PipelineConfig:
        out = _copy(cfg)
        for dotted, value in changes.items():
            obj: object = out
            *parents, leaf = dotted.split(".")
            for part in parents:
                obj = getattr(obj, part)
            setattr(obj, leaf, value)
        return out

    return preset


# Each entry takes the *full* config and returns the ablated variant.
ABLATIONS: dict[str, Preset] = {
    "full": full,
    "baseline": baseline,
    "semantic_only": semantic_only,
    # routing
    "router_keyword": _set(**{"router.variant": "keyword"}),
    "router_oracle": _set(**{"router.variant": "oracle"}),
    "router_learned": _set(**{"router.variant": "learned"}),
    # fusion / correction
    "wo_dphf": _set(**{"fusion.variant": "single"}),
    "wo_sscc": _set(**{"correction.variant": "crag"}),
    "wo_crag": _set(**{"correction.enabled": False}),
    "wo_rerank": _set(**{"rerank.enabled": False}),
    # graph
    "wo_graph": _disable_graph,
    "wo_graphsage": _set(**{"graph.gnn_layers": 0}),  # PPR expansion only (no GNN rescore)
    "graph_no_expand": _set(**{"graph.ppr_expand": False}),  # GNN rescore only (no expansion)
    "wo_nli_edges": _drop_edge("nli"),
    "wo_semantic_edges": _drop_edge("semantic"),
    "wo_ast_edges": _drop_edge("ast"),
    # hierarchy
    "wo_raptor": _set(**{"raptor.enabled": False}),
    "wo_cross_doc": _set(**{"raptor.cross_doc": False}),
    "raptor_kmeans": _set(**{"raptor.cluster_covariance": "spherical"}),
    # systems
    "wo_srp": _set(**{"prefetch.enabled": False}),
    "wo_raac": _set(**{"raac.enabled": False}),
}


def apply(name: str, base: PipelineConfig | None = None) -> PipelineConfig:
    """Return the named ablation applied on top of the full configuration."""
    if name not in ABLATIONS:
        raise KeyError(f"unknown ablation {name!r}; known: {sorted(ABLATIONS)}")
    start = full(base or PipelineConfig())
    return ABLATIONS[name](start)
