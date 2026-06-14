"""Build publication figures from extracted per-query data in paper/figdata/.

Figures show distributions / mechanisms / relationships -- not re-plots of the ablation
means (those live in tables). Vector PDF + PNG preview via paper/figstyle.

    uv run --group paper python paper/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import figstyle as fs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIGDATA = Path("paper/figdata")
_LABEL_NAMES = {"semantic": "Semantic", "dphf": "DPHF", "stepback": "Step-back"}


def fig_routing_degeneracy(data: dict, name: str) -> None:
    """Three panels: distance saturation -> entropy at ceiling -> no class separation."""
    qids = list(data["entropy"])
    H = np.array([data["entropy"][q] for q in qids])
    logk = data["log_k"]
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
    for _lbl, (key, col) in enh.items():
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


def _youden_threshold(rel: np.ndarray, irr: np.ndarray) -> float:
    """Score threshold maximizing TPR - FPR (best separation of gold vs non-gold)."""
    cand = np.unique(np.concatenate([rel, irr]))
    j = [(rel >= t).mean() - (irr >= t).mean() for t in cand]
    return float(cand[int(np.argmax(j))])


def fig_sscc_calibration(data: dict) -> None:
    """Why per-source thresholds (SSCC): bi- and cross-encoder scores separate gold at
    different score values, so one global threshold (CRAG) cannot serve both."""
    from scipy.stats import gaussian_kde

    panels = [("bi", "(a) Bi-encoder  $1/(1{+}L_2)$", "bi-encoder relevance score", "upper left"),
              ("cross", "(b) Cross-encoder reranker", "cross-encoder relevance score", "upper right")]
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDE * 0.72, 2.5))
    taus = {}
    for ax, (key, title, xlab, loc) in zip(axes, panels):
        arr = np.array(data[key], dtype=np.float64)
        rel, irr = arr[arr[:, 1] == 1, 0], arr[arr[:, 1] == 0, 0]
        lo, hi = arr[:, 0].min(), arr[:, 0].max()
        xs = np.linspace(lo, hi, 200)
        for vals, col, lab in [(irr, fs.NULL, "non-relevant"), (rel, fs.GREEN, "relevant")]:
            d = gaussian_kde(vals)(xs)
            ax.fill_between(xs, d, color=col, alpha=0.35, lw=0, zorder=2)
            ax.plot(xs, d, color=col, lw=1.3, zorder=3, label=lab)
        tau = _youden_threshold(rel, irr)
        taus[key] = tau
        ax.axvline(tau, color=fs.ACCENT, lw=1.4, ls=(0, (4, 2)), zorder=4,
                   label="calibrated $\\tau$")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(xlab)
        ax.set_ylabel("density")
        ax.set_yticks([])
        ax.set_title(title, loc="left")
        ax.legend(loc=loc, fontsize=6.4, handlelength=1.2)
    fig.tight_layout(w_pad=1.6)
    fs.save(fig, "F_sscc_calibration")
    print(f"wrote F_sscc_calibration  (tau_bi={taus['bi']:.3f}, tau_cross={taus['cross']:.3f})")


def fig_heterogeneity(perq: dict) -> None:
    """Effect (delta-F1 (full minus ablate) with 95% CI for three components across benchmarks ordered
    by heterogeneity. SSCC rises with heterogeneity; HyDE helps throughout; RAPTOR is null."""
    from sage.eval.stats import paired_diff_ci

    order = ["musique", "qasper", "hetdocqa"]
    xlabels = ["MuSiQue\n(prose)", "QASPER\n(sci-prose)", "HetDocQA\n(heterog.)"]
    comps = [("wo_sscc", "SSCC", fs.GREEN, "o"),
             ("wo_hyde", "HyDE", fs.COOL, "s"),
             ("wo_raptor", "RAPTOR", fs.NULL, "^")]
    fig, ax = plt.subplots(figsize=(fs.COL * 1.5, 2.7))
    ax.axhline(0, color="#8a9098", lw=0.9, zorder=1)
    for j, (key, label, col, mk) in enumerate(comps):
        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(order):
            d = perq.get(b)
            if d is None or key not in d["configs"]:
                continue
            qids = d["qids"]
            a = [d["configs"]["full"]["f1"].get(q, 0.0) for q in qids]
            c = [d["configs"][key]["f1"].get(q, 0.0) for q in qids]
            delta, clo, chi = paired_diff_ci(a, c, seed=0)
            xs.append(i + (j - 1) * 0.09)
            ys.append(delta)
            lo.append(delta - clo)
            hi.append(chi - delta)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, ms=4.5, lw=1.3,
                    capsize=2.2, capthick=0.8, label=label, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(xlabels)
    ax.set_xlim(-0.4, len(order) - 0.6)
    ax.set_ylabel(r"$\Delta$F1  (full $-$ ablate)")
    ax.set_xlabel(r"increasing corpus heterogeneity $\rightarrow$")
    ax.legend(loc="upper left", fontsize=6.6, handlelength=1.4)
    ax.set_title("Component benefit vs. corpus heterogeneity", loc="left")
    fig.tight_layout()
    fs.save(fig, "F_heterogeneity")
    print("wrote F_heterogeneity")


def fig_difficulty(perq: dict) -> None:
    """Appendix: where HetDocQA is hard -- answer F1 by question type, retrieval nDCG@10
    by the modality of a question's gold evidence."""
    from sage.eval.stats import bootstrap_ci

    rows = [json.loads(line) for line in
            Path("data/hetdocqa/hetdocqa.jsonl").read_text().splitlines() if line.strip()]
    manifest = {d["doc_id"]: d["modality"]
                for d in json.loads(Path("data/hetdocqa/corpus_manifest.json").read_text())}
    meta = {r["qid"]: (r["type"],
                       {manifest.get(s["document_id"], "prose") for s in r["gold_spans"]})
            for r in rows}
    full = perq["configs"]["full"]
    qids = [q for q in perq["qids"] if q in meta]

    def bars(ax, groups, score_key, xlabel):
        labels, means, los, his = [], [], [], []
        for name, qs in groups:
            vals = [full[score_key][q] for q in qs if q in full[score_key]]
            if len(vals) < 3:
                continue
            mean, lo, hi = bootstrap_ci(vals, seed=0)
            labels.append(f"{name}  ($n{{=}}{len(vals)}$)")
            means.append(mean)
            los.append(mean - lo)
            his.append(hi - mean)
        order = np.argsort(means)
        y = np.arange(len(order))
        ax.barh(y, [means[i] for i in order], color=fs.BAR, edgecolor="white", lw=0.5,
                zorder=3, height=0.66)
        ax.errorbar([means[i] for i in order], y, xerr=[[los[i] for i in order],
                    [his[i] for i in order]], fmt="none", ecolor=fs.INK, elinewidth=0.8,
                    capsize=2, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[i] for i in order])
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, max(means) + max(his) + 0.06)

    types = ["factual", "code", "cross_document", "multi_hop", "thematic"]
    type_names = {"cross_document": "cross-doc", "multi_hop": "multi-hop"}
    by_type = [(type_names.get(t, t), [q for q in qids if meta[q][0] == t]) for t in types]
    mods = ["prose", "code", "table", "markdown", "pdf"]
    by_mod = [(m, [q for q in qids if m in meta[q][1]]) for m in mods]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(fs.WIDE, 2.5))
    bars(axA, by_type, "f1", "answer F1")
    axA.set_title("(a) Answer F1 by question type", loc="left")
    bars(axB, by_mod, "ndcg", "retrieval nDCG@10")
    axB.set_title("(b) Retrieval by gold-evidence modality", loc="left")
    fig.tight_layout(w_pad=2.0)
    fs.save(fig, "A_difficulty")
    print("wrote A_difficulty")


def fig_mahyde_control() -> None:
    """Appendix: modality typing vs. a same-size prose ensemble on the code/table test
    subset (n=56). Modality is numerically best but not significantly above multi-prose."""
    # from results/hetdocqa_mahyde_control.txt (frozen test, code/table subgroup).
    metrics = ["nDCG@10", "Recall@10", "F1"]
    arms = [("generic ($\\times$1 prose)", [0.7425, 0.7777, 0.5602], fs.NULL),
            ("multi-prose ($\\times$3)", [0.7672, 0.8060, 0.5519], fs.COOL),
            ("modality ($\\times$3)", [0.7849, 0.8134, 0.5962], fs.GREEN)]
    fig, ax = plt.subplots(figsize=(fs.COL * 1.6, 2.6))
    x = np.arange(len(metrics))
    w = 0.26
    for j, (name, vals, col) in enumerate(arms):
        ax.bar(x + (j - 1) * w, vals, w, color=col, label=name, edgecolor="white", lw=0.5,
               zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("score (code/table test subset, $n{=}56$)")
    ax.legend(loc="upper right", fontsize=6.4)
    ax.set_title("Modality typing vs. ensemble (code/table questions)", loc="left")
    fig.tight_layout()
    fs.save(fig, "A_mahyde_control")
    print("wrote A_mahyde_control")


def main() -> None:
    fs.use_style()
    fig_mahyde_control()
    hetq = Path("results/hetdocqa_dev_perquery.json")
    if hetq.exists():
        fig_difficulty(json.loads(hetq.read_text()))
    routing = FIGDATA / "routing_musique.json"
    if routing.exists():
        fig_routing_degeneracy(json.loads(routing.read_text()), "musique")
    recall = FIGDATA / "recall_hetdocqa.json"
    if recall.exists():
        fig_recall_dominance(json.loads(recall.read_text()))
    sscc = FIGDATA / "sscc.json"
    if sscc.exists():
        fig_sscc_calibration(json.loads(sscc.read_text()))
    perq = {b: json.loads(p.read_text())
            for b in ("musique", "qasper", "hetdocqa")
            for p in [Path(f"results/{b}_dev_perquery.json")] if p.exists()}
    if {"musique", "qasper", "hetdocqa"} <= set(perq):
        fig_heterogeneity(perq)


if __name__ == "__main__":
    main()
