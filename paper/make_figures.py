"""Generate paper figures from the committed result tables.

DRAFT figures use the per-row F1 +/- 95% CI in results/*_ablation.txt. Final versions
will use per-query paired-delta CIs once the warm-cache re-run writes the per-query JSON
(scripts/run_benchmark.py now persists results/<name>_<split>_perquery.json).

    uv run --group paper python paper/make_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path("paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)
ROW = re.compile(
    r"^(\S+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\[([\d.]+),\s*([\d.]+)\]\s+(\S+)"
)

# Friendlier labels for the components we ablate.
PRETTY = {
    "full": "Full system", "wo_rerank": "− Reranker", "wo_hyde": "− HyDE",
    "wo_sscc": "− SSCC (→CRAG)", "wo_crag": "− Correction", "wo_dphf": "− DPHF/RRF",
    "wo_graph": "− Graph (GAHR)", "wo_raptor": "− RAPTOR", "wo_cross_doc": "− Cross-doc tier",
    "wo_modality_hyde": "− Modality-HyDE", "hyde_multi_prose": "Multi-prose HyDE",
    "semantic_only": "Semantic only",
}


def parse_component(path: Path) -> dict[str, tuple[float, float, float, float, float]]:
    """Return {config: (nDCG, F1, ci_lo, ci_hi, p)} from the component-ablation block."""
    rows: dict[str, tuple[float, float, float, float, float]] = {}
    in_block = False
    for line in path.read_text().splitlines():
        if "component ablation" in line:
            in_block = True
            continue
        if in_block and line.startswith("total"):
            break
        m = ROW.match(line.strip()) if in_block else None
        if m:
            cfg, ndcg, f1, lo, hi, p = m.groups()
            rows[cfg] = (float(ndcg), float(f1), float(lo), float(hi),
                         float("nan") if p == "nan" else float(p))
    return rows


def fig_forest(rows: dict, out: Path) -> None:
    """F3 draft: F1 +/- 95% CI per config; full as a reference band; sig vs full marked."""
    full_f1 = rows["full"][1]
    order = sorted(rows, key=lambda c: rows[c][1])  # ascending F1
    ys = range(len(order))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(rows["full"][2], rows["full"][3], color="0.85", zorder=0, label="Full 95% CI")
    ax.axvline(full_f1, color="0.4", ls="--", lw=1, zorder=1)
    for y, cfg in zip(ys, order):
        ndcg, f1, lo, hi, p = rows[cfg]
        sig = (not (p != p)) and p < 0.05 and cfg != "full"  # p==p false for nan
        color = "#c0392b" if sig else ("#2c3e50" if cfg == "full" else "#7f8c8d")
        ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2)
        ax.plot(f1, y, "o", color=color, ms=6, zorder=3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([PRETTY.get(c, c) for c in order], fontsize=9)
    ax.set_xlabel("Answer token-F1 (HetDocQA test, 363 q)")
    ax.set_title("Effect of removing each component (DRAFT: per-row CI)", fontsize=10)
    ax.text(0.98, 0.02, "red = sig. vs Full (raw p<.05)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="#c0392b")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_gradient(out: Path) -> None:
    """F5 draft: SSCC effect (full_F1 - wo_sscc_F1) across homogeneous -> heterogeneous."""
    benches = ["MuSiQue\n(prose)", "QASPER\n(sci prose)", "HetDocQA\n(heterogeneous)"]
    sscc = [0.3925 - 0.3727, 0.2178 - 0.2210, 0.5481 - 0.5234]  # +helps
    sig = [False, False, True]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(benches, sscc, color=["#bdc3c7" if not s else "#27ae60" for s in sig])
    ax.axhline(0, color="0.3", lw=1)
    for b, s in zip(bars, sig):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                ("*" if s else "n.s."), ha="center",
                va="bottom" if b.get_height() >= 0 else "top", fontsize=11)
    ax.set_ylabel("SSCC effect on F1 (full − ablate)")
    ax.set_title("SSCC helps only with heterogeneity (DRAFT)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_dominance(rows: dict, out: Path) -> None:
    """F4 draft: nDCG@10 full vs −reranker vs −every-other (mean), showing the collapse."""
    full_nd = rows["full"][0]
    no_rr = rows["wo_rerank"][0]
    others = [v[0] for c, v in rows.items() if c not in ("full", "wo_rerank", "semantic_only")]
    mean_other = sum(others) / len(others)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(["Full", "− Reranker", "− any other\n(mean)"],
                  [full_nd, no_rr, mean_other],
                  color=["#2c3e50", "#c0392b", "#7f8c8d"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_ylabel("nDCG@10 (HetDocQA test)")
    ax.set_title("The reranker carries the system (DRAFT)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    rows = parse_component(Path("results/hetdocqa_test_ablation.txt"))
    fig_forest(rows, FIGS / "F3_forest_DRAFT.png")
    fig_gradient(FIGS / "F5_gradient_DRAFT.png")
    fig_dominance(rows, FIGS / "F4_dominance_DRAFT.png")
    print(f"wrote {sorted(p.name for p in FIGS.glob('*.png'))}")


if __name__ == "__main__":
    main()
