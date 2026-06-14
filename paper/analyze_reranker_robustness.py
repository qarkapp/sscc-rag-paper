"""Reranker robustness check (review issue M1).

Recompute the survival-grid verdict for every method under two rerankers --- the
listwise jina-reranker-v3 (default, no suffix) and a plain pairwise cross-encoder
bge-reranker-v2-m3 (``_bge`` suffix) --- and report whether the conclusions change.

Verdict logic mirrors the paper: effect of a method = F1(full) - F1(full without
the method); a paired bootstrap p-value vs ``full``; Holm-Bonferroni over the
eight-method family per (benchmark, reranker, split); and, for HetDocQA, a
sign-consistency check across dev and test (\\yes only if Holm-significant on test
and the sign agrees on dev; \\unstable if significant on test but the sign flips).
MuSiQue and QASPER have a single split, so the verdict is Holm-significance alone.

    uv run python paper/analyze_reranker_robustness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sage.eval.stats import holm_bonferroni, paired_bootstrap_test

RESULTS = Path("results")

# Survival-grid rows: display name -> ablation key (removing the method).
METHODS = {
    "reranker": "wo_rerank",
    "HyDE": "wo_hyde",
    "SSCC": "wo_sscc",
    "RAPTOR": "wo_raptor",
    "DPHF/RRF": "wo_dphf",
    "GAHR graph": "wo_graph",
    "cross-document": "wo_cross_doc",
    "CRAG": "wo_crag",
}

# Retrieval-only robustness check (nDCG@10 per query). The jina baseline and the
# bge variant are produced by the SAME code path (run_retrieval_only.py), so the
# comparison is apples-to-apples regardless of any generation nondeterminism.
# (benchmark, split, jina-file, bge-file)
METRIC = "ndcg"
DATASETS = [
    ("hetdocqa", "dev", "hetdocqa_dev_retonly.json", "hetdocqa_dev_retonly_bge.json"),
    ("hetdocqa", "test", "hetdocqa_test_retonly.json", "hetdocqa_test_retonly_bge.json"),
    ("musique", "dev", "musique_dev_retonly.json", "musique_dev_retonly_bge.json"),
    ("qasper", "dev", "qasper_dev_retonly.json", "qasper_dev_retonly_bge.json"),
]


def _metric_arrays(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    d = json.loads(path.read_text())
    qids = d["qids"]
    out: dict[str, np.ndarray] = {}
    for cfg, payload in d["configs"].items():
        scores = payload[METRIC]
        out[cfg] = np.array([scores.get(q, 0.0) for q in qids], dtype=np.float64)
    return qids, out


def _effects(path: Path) -> dict[str, dict[str, float]]:
    """For each method: delta = mean(full) - mean(wo_method), and paired p vs full."""
    _, f1 = _metric_arrays(path)
    full = f1["full"]
    pvals, deltas = [], []
    keys = list(METHODS)
    for name in keys:
        wo = f1[METHODS[name]]
        deltas.append(float(full.mean() - wo.mean()))
        pvals.append(paired_bootstrap_test(wo, full, seed=0))
    reject = holm_bonferroni(pvals, alpha=0.05)
    return {
        name: {"delta": deltas[i], "p": pvals[i], "holm": bool(reject[i])}
        for i, name in enumerate(keys)
    }


def _verdict_hetdoc(dev: dict, test: dict, name: str) -> str:
    t, d = test[name], dev[name]
    if not t["holm"]:
        return "no"
    same_sign = (t["delta"] >= 0) == (d["delta"] >= 0)
    return "yes" if same_sign else "unstable"


def _verdict_single(eff: dict, name: str) -> str:
    return "yes" if eff[name]["holm"] else "no"


def main() -> None:
    # Load whatever exists, per reranker.
    eff: dict[tuple[str, str, str], dict] = {}  # (rerank, bench, split) -> effects
    for bench, split, jfile, bfile in DATASETS:
        for tag, fname in (("jina", jfile), ("bge", bfile)):
            p = RESULTS / fname
            if p.exists():
                try:
                    eff[(tag, bench, split)] = _effects(p)
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"  ! skip {fname}: {e}")

    # ---- HetDocQA headline table (dev+test verdict), per reranker -------------
    for tag in ("jina", "bge"):
        dev = eff.get((tag, "hetdocqa", "dev"))
        test = eff.get((tag, "hetdocqa", "test"))
        if not (dev and test):
            print(f"\n[{tag}] HetDocQA: missing dev or test -- skipping")
            continue
        print(f"\n===== HetDocQA headline, reranker = {tag} =====")
        print(f"{'method':16} {'dF1(test)':>10} {'p(test)':>9} {'holm':>5} "
              f"{'dF1(dev)':>10} {'verdict':>9}")
        for name in METHODS:
            t, d = test[name], dev[name]
            print(f"{name:16} {t['delta']:>+10.4f} {t['p']:>9.3f} "
                  f"{str(t['holm']):>5} {d['delta']:>+10.4f} "
                  f"{_verdict_hetdoc(dev, test, name):>9}")

    # ---- Cross-benchmark survival grid: jina vs bge --------------------------
    def grid(tag: str) -> dict[str, str]:
        out = {}
        for name in METHODS:
            if (tag, "hetdocqa", "dev") in eff and (tag, "hetdocqa", "test") in eff:
                out[f"{name}@HetDocQA"] = _verdict_hetdoc(
                    eff[(tag, "hetdocqa", "dev")], eff[(tag, "hetdocqa", "test")], name)
            for bench in ("musique", "qasper"):
                if (tag, bench, "dev") in eff:
                    out[f"{name}@{bench}"] = _verdict_single(eff[(tag, bench, "dev")], name)
        return out

    gj, gb = grid("jina"), grid("bge")
    keys = sorted(set(gj) | set(gb))
    print("\n===== Survival grid: jina vs bge (verdict per method x benchmark) =====")
    print(f"{'method @ benchmark':28} {'jina':>9} {'bge':>9} {'changed?':>9}")
    changed = 0
    for k in keys:
        vj, vb = gj.get(k, "-"), gb.get(k, "-")
        flag = "" if vj == vb else "<-- CHANGED"
        if vj != vb and "-" not in (vj, vb):
            changed += 1
        print(f"{k:28} {vj:>9} {vb:>9}   {flag}")
    print(f"\nverdict cells that changed between rerankers: {changed}")


if __name__ == "__main__":
    main()
