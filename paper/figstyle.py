"""Publication figure style: serif typography, restrained palette, vector PDF output.

Import and call ``use_style()`` before plotting; ``save(fig, name)`` writes a vector PDF
(for LaTeX) plus a PNG preview at column / double-column widths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path("paper/figs")
FIGS.mkdir(parents=True, exist_ok=True)

# Single- and double-column widths for a typical two-column proceedings template (inches).
COL = 3.33
WIDE = 7.0

# Restrained, colorblind-safe palette: ink for primary, one warm accent for emphasis,
# a cool secondary, neutral grays for null/context.
INK = "#1b2838"
ACCENT = "#b1283a"     # crimson -- the one thing that matters per figure
COOL = "#2f6f8f"       # slate blue -- secondary series
GREEN = "#2e7d52"      # the lone positive (SSCC)
NULL = "#9aa3ab"       # null / "everything else"
NULL_L = "#cfd4d9"
GRID = "#e6e8eb"


def use_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,            # embed TrueType (editor-friendly, no Type3)
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "TeX Gyre Heros", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#3a3f44",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
    })


def panel_label(ax, text: str, dx: float = -0.02, dy: float = 1.0) -> None:
    """Bold (a)/(b)/(c) panel tag at the top-left, outside the axes."""
    ax.text(dx, dy + 0.04, text, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png", dpi=200)
    plt.close(fig)
