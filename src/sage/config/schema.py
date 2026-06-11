"""Configuration schema for the full pipeline.

Every component has its own validated config model with an ``enabled`` flag and,
where there are alternatives, a ``variant`` selector. The pipeline is assembled
entirely from a :class:`PipelineConfig`, so an ablation is a configuration change
rather than a code change.

Defaults reflect the locally available oMLX models and the constants recovered from
the reference implementation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from sage.cache.store import CacheMode

EdgeType = Literal["sequential", "semantic", "xref", "ast", "nli"]
_DEFAULT_EDGES: list[EdgeType] = ["sequential", "semantic", "xref", "ast"]

__all__ = [
    "BackendCfg",
    "CacheCfg",
    "ChunkingCfg",
    "CorrectionCfg",
    "EmbedderCfg",
    "ExpansionCfg",
    "FusionCfg",
    "GeneratorCfg",
    "GraphCfg",
    "PipelineConfig",
    "PrefetchCfg",
    "RaacCfg",
    "RaptorCfg",
    "RerankCfg",
    "RerankerCfg",
    "RouterCfg",
]


class BackendCfg(BaseModel):
    """Connection settings for an OpenAI-compatible backend (oMLX/OpenRouter)."""

    provider: Literal["omlx", "openrouter", "openai"] = "omlx"
    model: str = "bge-m3-mlx-fp16"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0


class EmbedderCfg(BackendCfg):
    model: str = "bge-m3-mlx-fp16"
    batch_size: int = 64
    probe_dim: bool = True


class RerankerCfg(BackendCfg):
    enabled: bool = True
    model: str = "jina-reranker-v3-mlx"


class GeneratorCfg(BackendCfg):
    """Generator for HyDE/step-back/CRAG and summarization."""

    model: str = "Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
    max_tokens: int = 512
    temperature: float = 0.0


class CacheCfg(BaseModel):
    root: str = ".cache/calls"
    mode: CacheMode = CacheMode.READ_WRITE


class ChunkingCfg(BaseModel):
    prose_target_chars: int = 1500
    prose_overlap_chars: int = 200
    prose_min_chars: int = 100
    code_target_chars: int = 1500  # ~375 tokens
    prose_backend: Literal["chonkie", "semantic", "naive"] = "chonkie"


class RaptorCfg(BaseModel):
    enabled: bool = True
    max_levels: int = 3
    min_nodes_for_level: int = 4
    soft_cluster_threshold: float = 0.1
    max_clusters: int = 50
    # Clusters larger than this are refined with a second, local clustering pass.
    # The default keeps a single global pass for typical sizes (local re-projection
    # of a small homogeneous cluster can introduce spurious sub-structure).
    local_refine_min_nodes: int = 64
    umap_target_dim: int = 10
    umap_n_epochs: int = 100
    max_cluster_tokens: int = 20000
    retrieval_mode: Literal["collapsed", "tree_traversal"] = "tree_traversal"
    retrieval_token_budget: int = 2000
    traversal_top_k: int = 7
    cross_doc: bool = True
    cluster_covariance: Literal["full", "diag", "tied", "spherical"] = "full"


class RouterCfg(BaseModel):
    enabled: bool = True
    variant: Literal["keyword", "egr", "oracle", "learned"] = "egr"
    egr_k: int = 20
    egr_temperature: float = 1.0
    egr_tau_low: float = 1.8
    egr_tau_high: float = 2.6
    calibrate: bool = False
    calibration_set: str | None = None


class FusionCfg(BaseModel):
    # "single" = hypothesis path only (single-path HyDE baseline); "merge_dedup" =
    # dual-path union by id (reference behaviour); "rrf" = dual-path reciprocal-rank
    # fusion (DPHF).
    variant: Literal["single", "merge_dedup", "rrf"] = "rrf"
    rrf_k: int = 60


class RerankCfg(BaseModel):
    enabled: bool = True
    over_fetch: int = 3  # m: retrieve m*k candidates before reranking
    top_k: int = 5


class CorrectionCfg(BaseModel):
    enabled: bool = True
    variant: Literal["crag", "sscc"] = "sscc"
    # CRAG (single-threshold baseline), scores on a 1-100 scale.
    crag_upper: float = 60.0
    crag_lower: float = 25.0
    # SSCC: tau(source) = tau0 * (1 + alpha * 1[source == bi_encoder]).
    sscc_tau0: float = 0.5
    sscc_alpha: float = 0.6
    enable_query_rewrite: bool = True


class GraphCfg(BaseModel):
    enabled: bool = False
    edges: list[EdgeType] = Field(default_factory=lambda: list(_DEFAULT_EDGES))
    semantic_threshold: float = 0.7
    gnn_layers: int = 2
    gnn_hidden: int = 128
    gnn_objective: Literal["graphsage_neighbor", "dgi"] = "graphsage_neighbor"
    query_fusion: Literal["project_query", "late_fusion", "query_proj_head"] = "project_query"
    late_fusion_beta: float = 0.5
    ppr_steps: int = 20
    ppr_alpha: float = 0.15
    ppr_expand_frac: float = 0.5
    nli_model: str = "jina-reranker-v3-mlx"
    nli_cos_gate: float = 0.5


class PrefetchCfg(BaseModel):
    enabled: bool = False
    entity_window_tokens: int = 32
    hit_cosine_threshold: float = 0.8


class RaacCfg(BaseModel):
    enabled: bool = False
    reindex_cycles: int = 3
    split_precision_threshold: float = 0.4
    merge_coretrieval_threshold: float = 0.2


class ExpansionCfg(BaseModel):
    """Parent-chunk neighbour expansion of final results."""

    enabled: bool = True
    window_chars: int = 200


class PipelineConfig(BaseModel):
    """Top-level configuration assembled into a runnable pipeline."""

    seed: int = 42
    top_k: int = 5
    cache: CacheCfg = Field(default_factory=CacheCfg)
    embedder: EmbedderCfg = Field(default_factory=EmbedderCfg)
    reranker: RerankerCfg = Field(default_factory=RerankerCfg)
    generator: GeneratorCfg = Field(default_factory=GeneratorCfg)
    chunking: ChunkingCfg = Field(default_factory=ChunkingCfg)
    raptor: RaptorCfg = Field(default_factory=RaptorCfg)
    router: RouterCfg = Field(default_factory=RouterCfg)
    fusion: FusionCfg = Field(default_factory=FusionCfg)
    rerank: RerankCfg = Field(default_factory=RerankCfg)
    correction: CorrectionCfg = Field(default_factory=CorrectionCfg)
    graph: GraphCfg = Field(default_factory=GraphCfg)
    prefetch: PrefetchCfg = Field(default_factory=PrefetchCfg)
    raac: RaacCfg = Field(default_factory=RaacCfg)
    expansion: ExpansionCfg = Field(default_factory=ExpansionCfg)
