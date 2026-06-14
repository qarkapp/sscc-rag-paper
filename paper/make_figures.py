"""Build publication figures from extracted per-query data in paper/figdata/.

Figures show distributions / mechanisms / relationships -- not re-plots of the ablation
means (those live in tables). Vector PDF + PNG preview via paper/figstyle.

    uv run --group paper python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIGDATA = Path("paper/figdata")
_LABEL_NAMES = {"semantic": "Semantic", "dphf": "DPHF", "stepback": "Step-back"}


def fig_routing_degeneracy(data: dict, name: str) -> None:
    """Three panels: distance saturation -> entropy at ceiling -> no class separation."""
    qids = list(data["entropy"])
    H = np.array([data["entropy"][q] for q in qids])
    logk, tlo, thi = data["log_k"], data["tau_low"], data["tau_high"]
    dists = {q: np.sort(np.asarray(data["knn_distances"][q])) for q in qids}
    labels = data["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDE, 2.55))

    # (a) kNN distance profiles are near-flat -> softmax(-d/T) is ~uniform.
    ax = axes[0]
    k = len(next(iter(dists.values())))
    xs = np.arange(1, k + 1)
    prof = np.vstack([dists[q] for q in qids])
    med = np.median(prof, axis=0)
    q25, q75 = np.percentile(prof, [25, 75], axis=0)
    rng = np.random.default_rng(0)
    for q in rng.choice(qids, size=min(60, len(qids)), replace=False):
        ax.plot(xs, dists[q], color=fs.NULL, lw=0.4, alpha=0.18, zorder=1)
    ax.fill_between(xs, q25, q75, color=fs.COOL, alpha=0.25, lw=0, zorder=2)
    ax.plot(xs, med, color=fs.INK, lw=1.6, zorder=3)
    span = float(np.median(prof[:, -1] - prof[:, 0]))
    ax.annotate(rf"span $\Delta d \approx {span:.2f}$",
                xy=(k * 0.72, med[int(k * 0.72)]), xytext=(4.5, 0.265),
                fontsize=7, color=fs.ACCENT, va="center",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=fs.ACCENT))
    ax.set_ylim(0.18, 0.72)
    ax.set_xlabel("nearest-neighbour rank $i$")
    ax.set_ylabel("$L_2$ distance $d_i$")
    ax.set_title("(a) Neighbour distances are near-flat", loc="left")

    # (b) the actual entropy distribution: a tight spike just below the log K ceiling, far
    # above both EGR thresholds. The consequence (all route to step-back) is stated, not
    # drawn at full range (which would put the thresholds off in empty space).
    axb = axes[1]
    n_sb = sum(1 for q in qids if data["egr"][q] == "stepback")
    counts, _, _ = axb.hist(H, bins=np.linspace(H.min() - 0.0009, logk + 0.0009, 30),
                            color=fs.BAR, edgecolor="white", linewidth=0.4, zorder=3)
    m = max(counts)
    axb.set_ylim(0, m * 1.34)
    axb.axvline(logk, color=fs.ACCENT, lw=1.2, ls=(0, (4, 2)), zorder=4)
    fs.halo(axb.text(logk, m * 1.16, r"$\log K$", color=fs.ACCENT, fontsize=7.5,
            ha="center", va="bottom", zorder=5))
    axb.text(0.05, 0.93, rf"$\sigma_H = {fs.sci(H.std())}$", transform=axb.transAxes,
             va="top", ha="left", fontsize=7.5, color=fs.INK)
    axb.set_xlim(H.min() - 0.0011, logk + 0.0011)
    ticks = [round(H.min(), 3), round((H.min() + logk) / 2, 3), round(logk, 3)]
    axb.set_xticks(ticks)
    axb.set_xticklabels([f"{t:.3f}" for t in ticks])
    axb.set_xlabel("routing entropy $H$")
    axb.set_ylabel("queries")
    axb.set_title("(b) Routing entropy is near-constant", loc="left")

    # (c) entropy does not separate the oracle-best classes: identical violins, and the
    # per-class means all land on a single grand-mean line.
    ax = axes[2]
    groups = [[data["entropy"][q] for q in qids if data["oracle"][q] == s] for s in labels]
    parts = ax.violinplot(groups, showextrema=False, widths=0.8)
    for b in parts["bodies"]:
        b.set_facecolor(fs.COOL)
        b.set_alpha(0.25)
        b.set_edgecolor(fs.COOL)
        b.set_linewidth(0.6)
    for i, g in enumerate(groups, start=1):
        ax.scatter(np.full(len(g), i) + rng.uniform(-0.08, 0.08, len(g)), g,
                   s=2.2, color=fs.INK, alpha=0.28, zorder=3, lw=0)
    grand = H.mean()
    ax.axhline(grand, color=fs.ACCENT, lw=1.0, ls=(0, (4, 2)), zorder=4)
    for i, g in enumerate(groups, start=1):
        ax.plot(i, np.mean(g), "D", color=fs.ACCENT, ms=4, mec="white", mew=0.5, zorder=6)
    between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if g)
    within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups if g)
    _ = between, within  # reported in the caption, not on the plot
    handles = [
        Line2D([0], [0], marker="D", color="none", markerfacecolor=fs.ACCENT,
               markeredgecolor="white", markersize=5, label="class mean"),
        Line2D([0], [0], color=fs.ACCENT, lw=1.2, ls=(0, (4, 2)), label="overall mean"),
    ]
    ax.legend(handles=handles, loc="lower center", fontsize=6.2, frameon=False,
              handletextpad=0.5, borderaxespad=0.3, labelspacing=0.25)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_LABEL_NAMES[s] for s in labels])
    ax.set_xlim(0.5, len(labels) + 0.5)
    ax.set_xlabel("oracle-best strategy")
    ax.set_ylabel("routing entropy $H$")
    ax.set_title("(c) Identical across oracle classes", loc="left")

    fig.tight_layout(w_pad=1.6)
    fs.save(fig, "F_routing_degeneracy")
    print("wrote F_routing_degeneracy")


def fig_recall_dominance(data: dict) -> None:
    """Two panels: the reranker lifts Recall@k; pool-expansion enhancements pile on top."""
    ks = np.array(data["ks"])
    c = data["curves"]
    dense = np.array(c["Dense (bi-encoder)"])
    rerank = np.array(c["+ Reranker"])
    enh = {"+ RAPTOR": ("+ Reranker + RAPTOR", fs.COOL),
           "+ Graph": ("+ Reranker + Graph", fs.GREEN),
           "+ all enh.": ("+ Reranker + all enh.", fs.ACCENT)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(fs.WIDE * 0.74, 2.5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) the whole story: dense vs +reranker vs +enhancements.
    axA.plot(ks, dense, color=fs.NULL, ls=(0, (4, 2)), lw=1.6, label="Dense (bi-encoder)", zorder=2)
    axA.plot(ks, rerank, color=fs.INK, lw=2.2, label="+ Reranker", zorder=4)
    for lbl, (key, col) in enh.items():
        axA.plot(ks, np.array(c[key]), color=col, lw=1.0, alpha=0.95, label=lbl, zorder=3)
    k0 = 9  # k=10
    axA.annotate("", xy=(10, rerank[k0]), xytext=(10, dense[k0]),
                 arrowprops=dict(arrowstyle="<->", color=fs.ACCENT, lw=1.0))
    axA.text(11.5, (rerank[k0] + dense[k0]) / 2,
             f"reranker\n$+{rerank[k0] - dense[k0]:.2f}$ R@10", color=fs.ACCENT, fontsize=7, va="center")
    axA.set_xlabel("$k$")
    axA.set_ylabel("Recall@$k$")
    axA.set_xlim(1, ks.max())
    axA.set_ylim(0, 1)
    axA.legend(loc="lower right", fontsize=6.6)
    axA.set_title("(a) The reranker carries retrieval", loc="left")

    # (b) zoom on the reranker-on curves: the enhancements are indistinguishable.
    band = np.vstack([rerank] + [np.array(c[k]) for k, _ in enh.values()])
    axB.fill_between(ks, band.min(0), band.max(0), color=fs.NULL_L, alpha=0.7, lw=0, zorder=1,
                     label="enh. spread")
    axB.plot(ks, rerank, color=fs.INK, lw=1.8, zorder=4, label="+ Reranker")
    for lbl, (key, col) in enh.items():
        axB.plot(ks, np.array(c[key]), color=col, lw=1.0, alpha=0.95, zorder=3)
    maxspread = float(np.max(band.max(0) - band.min(0)))
    axB.text(0.5, 0.08, f"max spread across\nenhancements $\\leq {maxspread:.03f}$",
             transform=axB.transAxes, ha="center", fontsize=7, color=fs.ACCENT)
    axB.set_xlabel("$k$")
    axB.set_ylabel("Recall@$k$")
    axB.set_xlim(ks.min(), ks.max())
    lo = float(band[:, 3:].min()) - 0.02
    hi = float(band[:, 3:].max()) + 0.03
    axB.set_ylim(lo, hi)
    axB.set_title("(b) Enhancements add nothing on top", loc="left")

    fig.tight_layout(w_pad=1.4)
    fs.save(fig, "F_recall_dominance")
    print("wrote F_recall_dominance")


def main() -> None:
    fs.use_style()
    routing = FIGDATA / "routing_musique.json"
    if routing.exists():
        fig_routing_degeneracy(json.loads(routing.read_text()), "musique")
    recall = FIGDATA / "recall_hetdocqa.json"
    if recall.exists():
        fig_recall_dominance(json.loads(recall.read_text()))


if __name__ == "__main__":
    main()
