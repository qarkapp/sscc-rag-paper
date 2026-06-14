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

    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDE, 2.35))

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

    # (b) entropy zoomed to the data (a spike at the ceiling); the EGR thresholds sit far
    # to the left (shown in a full-range context strip on top), so every query lands in
    # the step-back region. Two stacked axes: context strip + zoomed distribution.
    gs = axes[1].get_subplotspec().subgridspec(2, 1, height_ratios=[1, 4], hspace=0.05)
    axes[1].remove()
    axc = fig.add_subplot(gs[0])   # full-range context strip
    axz = fig.add_subplot(gs[1])   # zoomed distribution
    n_sb = sum(1 for q in qids if data["egr"][q] == "stepback")

    # context strip: full EGR axis with the three decision regions + where mass sits.
    axc.axvspan(1.6, tlo, color=fs.NULL_L, alpha=0.6, lw=0)
    axc.axvspan(tlo, thi, color="#dfe7ec", alpha=0.8, lw=0)
    axc.axvspan(thi, logk + 0.02, color="#f4d9dc", alpha=0.9, lw=0)
    for x in (tlo, thi):
        axc.axvline(x, color="#777", lw=0.7, zorder=2)
    # region names INSIDE the strip (centered), so nothing collides with the title.
    axc.text((1.6 + tlo) / 2, 0.5, "semantic", fontsize=5.5, ha="center", va="center",
             color="#555", transform=axc.get_xaxis_transform())
    axc.text((tlo + thi) / 2, 0.5, "DPHF", fontsize=5.5, ha="center", va="center",
             color="#555", transform=axc.get_xaxis_transform())
    axc.text((thi + logk) / 2 - 0.06, 0.5, "step-back", fontsize=5.5, ha="center", va="center",
             color=fs.ACCENT, transform=axc.get_xaxis_transform())
    axc.axvline(H.mean(), color=fs.ACCENT, lw=1.6, zorder=4)  # where all mass sits
    axc.text(tlo - 0.04, -0.45, r"$\tau_\ell$", fontsize=6, ha="center", color="#555",
             transform=axc.get_xaxis_transform())
    axc.text(thi - 0.04, -0.45, r"$\tau_h$", fontsize=6, ha="center", color="#555",
             transform=axc.get_xaxis_transform())
    axc.set_xlim(1.6, logk + 0.03)
    axc.set_ylim(0, 1)
    axc.set_yticks([])
    axc.tick_params(labelbottom=False, length=0)
    axc.set_title(r"(b) Entropy pinned to the $\log K$ ceiling", loc="left")

    # zoomed distribution near the ceiling.
    axz.hist(H, bins=np.linspace(H.min() - 0.0008, logk + 0.0008, 30), color=fs.INK, zorder=3)
    axz.axvline(logk, color=fs.ACCENT, lw=1.1, ls="--", zorder=4)
    axz.text(logk - 0.0006, axz.get_ylim()[1] * 0.92, r"$\log K$", color=fs.ACCENT,
             fontsize=6.5, va="top", ha="right")
    axz.text(0.04, 0.95, fr"$\bar H/\log K={H.mean()/logk:.3f}$" + "\n"
             + fr"$\sigma_H={H.std():.1e}$" + "\n" + f"{100*n_sb/len(qids):.0f}% $\\to$ step-back",
             transform=axz.transAxes, fontsize=6.8, va="top", ha="left", color=fs.INK)
    axz.set_xlim(H.min() - 0.0008, logk + 0.0010)
    axz.set_xlabel("routing entropy $H$ (zoom)")
    axz.set_ylabel("queries")
    axz.ticklabel_format(axis="x", useOffset=False)
    axz.set_xticks(np.round(np.linspace(H.min(), logk, 3), 3))

    # (c) entropy does not separate the oracle-best classes.
    ax = axes[2]
    groups = [[data["entropy"][q] for q in qids if data["oracle"][q] == s] for s in labels]
    parts = ax.violinplot(groups, showextrema=False, widths=0.85)
    for b in parts["bodies"]:
        b.set_facecolor(fs.COOL)
        b.set_alpha(0.35)
        b.set_edgecolor(fs.INK)
        b.set_linewidth(0.6)
    for i, g in enumerate(groups, start=1):
        ax.scatter(np.full(len(g), i) + rng.uniform(-0.07, 0.07, len(g)), g,
                   s=3, color=fs.INK, alpha=0.5, zorder=3, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.mean(g)] * 2, color=fs.ACCENT, lw=1.3, zorder=4)
    # between/within dispersion ratio (an F-like separation statistic).
    grand = H.mean()
    between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups if g)
    within = sum(sum((x - np.mean(g)) ** 2 for x in g) for g in groups if g)
    ax.text(0.5, 0.04, f"between/within $= {between / max(within, 1e-9):.3f}$",
            transform=ax.transAxes, fontsize=7, ha="center", color=fs.ACCENT)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([_LABEL_NAMES[s] for s in labels])
    ax.set_xlabel("oracle-best strategy")
    ax.set_ylabel("routing entropy $H$")
    ax.set_title("(c) Identical across oracle classes", loc="left")

    fig.tight_layout(w_pad=1.6)
    fs.save(fig, "F_routing_degeneracy")
    print("wrote F_routing_degeneracy")


def main() -> None:
    fs.use_style()
    routing = FIGDATA / "routing_musique.json"
    if routing.exists():
        fig_routing_degeneracy(json.loads(routing.read_text()), "musique")
    else:
        print(f"missing {routing}; run scripts/diag_router.py musique")


if __name__ == "__main__":
    main()
