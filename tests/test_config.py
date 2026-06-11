"""Tests for the configuration schema, presets, and loader."""

from __future__ import annotations

from sage.config import PipelineConfig, apply, load_config
from sage.config.presets import ABLATIONS, baseline, full, semantic_only


def test_defaults_match_local_backend():
    cfg = PipelineConfig()
    assert cfg.embedder.model == "bge-m3-mlx-fp16"
    assert cfg.reranker.model == "jina-reranker-v3-mlx"
    assert cfg.embedder.provider == "omlx"
    assert cfg.rerank.over_fetch == 3
    assert cfg.fusion.rrf_k == 60


def test_full_enables_contributions():
    cfg = full(PipelineConfig())
    assert cfg.graph.enabled
    assert cfg.router.variant == "egr"
    assert cfg.fusion.variant == "rrf"
    assert cfg.correction.variant == "sscc"
    assert "nli" in cfg.graph.edges


def test_baseline_disables_contributions():
    cfg = baseline(PipelineConfig())
    assert cfg.router.variant == "keyword"
    assert cfg.fusion.variant == "merge_dedup"
    assert cfg.correction.variant == "crag"
    assert not cfg.graph.enabled


def test_semantic_only_is_minimal():
    cfg = semantic_only(PipelineConfig())
    assert not cfg.rerank.enabled
    assert not cfg.correction.enabled
    assert not cfg.raptor.enabled


def test_every_ablation_applies_cleanly():
    for name in ABLATIONS:
        cfg = apply(name)
        assert isinstance(cfg, PipelineConfig)


def test_ablation_isolates_one_change():
    full_cfg = full(PipelineConfig())
    wo_dphf = apply("wo_dphf")
    assert wo_dphf.fusion.variant == "single"
    # graph stays on; only fusion changed relative to full
    assert wo_dphf.graph.enabled == full_cfg.graph.enabled


def test_drop_edge_ablation():
    cfg = apply("wo_nli_edges")
    assert "nli" not in cfg.graph.edges
    assert "semantic" in cfg.graph.edges


def test_loader_round_trip_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_MODEL", "Qwen3.6-35B-A3B-4bit")
    path = tmp_path / "cfg.yaml"
    path.write_text("seed: 7\ntop_k: 8\ngenerator:\n  model: ${MY_MODEL}\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.top_k == 8
    assert cfg.generator.model == "Qwen3.6-35B-A3B-4bit"
