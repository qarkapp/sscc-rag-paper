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
    ax.annotate(f"median span $\\Delta d\\approx{span:.2f}$\n($T=1\\Rightarrow$ softmax $\\approx$ uniform)",
                xy=(k * 0.7, med[int(k * 0.7)]), xytext=(3.2, 0.26),
                fontsize=6.8, color=fs.ACCENT,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=fs.ACCENT))
    ax.set_ylim(0.18, 0.72)
    ax.set_xlabel("nearest-neighbour rank $i$")
    ax.set_ylabel("$L_2$ distance $d_i$")
    ax.set_title("(a) Neighbour distances are near-flat", loc="left")

    # (b) full entropy range [0, log K]: EGR's three decision regions are wide and labelled,
    # and every query's entropy is jammed at the far-right ceiling (a zoom inset shows the
    # spike's shape). So EGR always picks step-back.
    axb = axes[1]
    n_sb = sum(1 for q in qids if data["egr"][q] == "stepback")
    top = logk + 0.06
    axb.axvspan(-0.05, tlo, color=fs.NULL_L, alpha=0.55, lw=0)
    axb.axvspan(tlo, thi, color="#dde6ec", lw=0)
    axb.axvspan(thi, top, color="#f4d9dc", lw=0)
    counts, _, _ = axb.hist(H, bins=np.linspace(-0.05, top, 90), color=fs.BAR,
                            edgecolor=fs.BAR, zorder=4)
    ymax = max(counts) * 1.2
    axb.set_ylim(0, ymax)
    for xc, lab, col in [(tlo / 2, "semantic", "#566069"),
                         (2.12, "DPHF", "#566069"),
                         (2.84, "step-back", fs.ACCENT)]:
        axb.text(xc, ymax * 0.965, lab, fontsize=6.5, ha="center", va="top", color=col)
    for x, lab in [(tlo, r"$\tau_\ell$"), (thi, r"$\tau_h$")]:
        axb.axvline(x, color="#7c8186", lw=0.8, zorder=5)
        axb.text(x, ymax * 0.62, lab, fontsize=7, ha="center", va="center", color="#566069",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"))
    axb.annotate(f"every query\n$H \\approx \\log K$\n($\\sigma_H = {fs.sci(H.std())}$)",
                 xy=(H.mean(), ymax * 0.4), xytext=((tlo + thi) / 2, ymax * 0.45),
                 fontsize=6.3, color=fs.ACCENT, ha="center", va="center",
                 arrowprops=dict(arrowstyle="->", lw=0.8, color=fs.ACCENT))
    axb.set_xlim(0, top)
    axb.set_xlabel("routing entropy $H$")
    axb.set_ylabel("queries")
    axb.set_title(r"(b) Entropy jammed at the $\log K$ ceiling", loc="left")

    # zoom inset over the empty low-entropy region: the spike's actual shape.
    axin = axb.inset_axes([0.085, 0.30, 0.42, 0.44])
    axin.hist(H, bins=np.linspace(H.min() - 0.0008, logk + 0.0008, 24), color=fs.BAR,
              edgecolor="white", linewidth=0.25)
    axin.axvline(logk, color=fs.ACCENT, lw=1.0, ls=(0, (3, 2)))
    axin.set_xlim(H.min() - 0.0008, logk + 0.001)
    axin.set_xticks([round(H.min(), 3), round(logk, 3)])
    axin.tick_params(labelsize=5.5, length=2, pad=1)
    axin.set_yticks([])
    axin.set_title("zoom near ceiling", fontsize=5.8, color=fs.INK, pad=2)

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
    # label the grand-mean line at its right end; the stat sits in the empty lower-right.
    fs.halo(ax.text(len(groups) + 0.42, grand, r"$\bar H$", fontsize=7.5, color=fs.ACCENT,
            va="center", ha="left", zorder=7))
    fs.halo(ax.text(0.97, 0.09, f"between/within $= {between / max(within, 1e-9):.3f}$",
            transform=ax.transAxes, fontsize=6.5, ha="right", va="bottom", color=fs.ACCENT, zorder=7))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_LABEL_NAMES[s] for s in labels])
    ax.set_xlim(0.45, len(labels) + 0.7)
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
    axB.set_xlim(1, ks.max())
    lo = float(min(rerank[4:].min(), band[:, 4:].min())) - 0.02
    axB.set_ylim(lo, 1.0)
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
