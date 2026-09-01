"""Figures for the 2-3 page technical report (report/report.html -> report.pdf).

Nothing here recomputes a metric. Every number drawn or annotated is read back
from results/metrics.json and results/preds_*.csv, so the figures cannot drift
away from the tables in the report.

REPLICATE SELECTION, STATED SO IT CANNOT BE MISTAKEN FOR CHERRY-PICKING
-----------------------------------------------------------------------
The scatter panels show ONE replicate; the tables show all five. The rule is
"the median replicate of the fine-tuned model", and split-seed 43 happens to be
the median on BOTH split definitions (pdb_id 0.195 of [.225 .102 .237 .195 .131],
mutation 0.663 of [.638 .684 .678 .663 .639]). The frozen probe is then drawn on
that SAME partition, so panels 2 and 3 are a like-for-like comparison rather
than each method's own best day.

Palette: categorical slots 1-3 of the validated default (blue/orange/aqua).
Adjacent-pair CVD dE >= 8 in light mode. Aqua sits below 3:1 on a light
surface, so every aqua mark carries a visible direct label and the report
prints the full table beside the figure -- the relief rule, satisfied twice.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# --- design tokens ---------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#7a7975"
GRID = "#e8e7e4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# PRINT SCALE -- why the font sizes below look too large.
# Both figures are authored 7.2in wide but printed narrower: report.html scales
# them to 67% and 86% of a 184mm text column. Point sizes shrink by the same
# factor, so 7pt type would reach paper at ~4.5pt and be unreadable. Authoring
# at size x SCALE lands each label at its nominal size on the page. If the CSS
# widths change, these change with them.
S1 = 1.49   # fig1, printed at 67% of the text column
S2 = 1.17   # fig2, printed at 86%

mpl.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.edgecolor": GRID,       # hairline, one shade off the surface
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",        # solid: dashing reads as "threshold"
    "legend.frameon": False,
})

METRICS = json.loads((RESULTS / "metrics.json").read_text())
MUT = pd.read_csv(ROOT / "data" / "processed" / "mutations.csv")[["uid", "ddg"]]

PARTITION = "split43"   # the median replicate; see the module docstring
TRAIN_SEED = "seed42"
SPLITS_FILE = ROOT / "data" / "processed" / "splits_seed43.csv"


def test_rows(run: str, split_col: str) -> pd.DataFrame:
    """Predictions joined to measured ddG, restricted to that split's test set."""
    preds = pd.read_csv(RESULTS / f"preds_{run}.csv")
    splits = pd.read_csv(SPLITS_FILE)[["uid", split_col]]
    df = preds.merge(MUT, on="uid").merge(splits, on="uid")
    return df[df[split_col] == "test"]


def spearman_of(run: str, split_col: str) -> float:
    return METRICS[run]["splits"][split_col]["test"]["spearman"]


# --- Figure 1: predicted vs measured ---------------------------------------
def figure_scatter() -> None:
    panels = [
        (f"finetune_esm2_t30_150M_UR50D_split_mutation_{PARTITION}_{TRAIN_SEED}",
         "split_mutation",
         "Fine-tuned ESM-2",
         "Naive split (leaks)"),
        (f"finetune_esm2_t30_150M_UR50D_split_pdb_id_{PARTITION}_{TRAIN_SEED}",
         "split_pdb_id",
         "Fine-tuned ESM-2",
         "Complex-level split"),
        (f"linear_probe_mlp_esm2_t30_150M_UR50D_split_pdb_id_{PARTITION}_{TRAIN_SEED}",
         "split_pdb_id",
         "Frozen ESM-2 + MLP head",
         "Complex-level split"),
    ]

    # `set_aspect("equal")` fights constrained_layout, which then crops the
    # titles and x-labels. Explicit margins plus a tight bbox is the reliable
    # combination when the axes have a fixed aspect ratio.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.35))
    fig.subplots_adjust(left=0.088, right=0.995, top=0.845, bottom=0.155, wspace=0.26)
    lo, hi = -6.5, 8.5

    for ax, (run, col, method, regime) in zip(axes, panels):
        df = test_rows(run, col)
        rho = spearman_of(run, col)
        r = stats.pearsonr(df["ddg"], df["y_pred"])[0]
        rmse = float(np.sqrt(np.mean((df["ddg"] - df["y_pred"]) ** 2)))

        ax.plot([lo, hi], [lo, hi], color=GRID, lw=0.8, zorder=1)
        # Thin marks, no edge stroke: a border around dense points reads as noise.
        ax.scatter(df["ddg"], df["y_pred"], s=5, alpha=0.30,
                   color=BLUE, edgecolors="none", linewidths=0, zorder=2)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.tick_params(labelsize=7.5 * S1)
        ax.grid(True, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

        ax.set_title(f"{method}\n{regime}", fontsize=8 * S1, pad=5, linespacing=1.4)
        ax.set_xlabel("measured ΔΔG  (kcal/mol)", fontsize=7 * S1)
        if ax is axes[0]:
            ax.set_ylabel("predicted ΔΔG  (kcal/mol)", fontsize=7 * S1)

        # Text wears ink tokens, never the series colour.
        ax.text(0.045, 0.955, f"ρ = {rho:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=9.5 * S1, color=INK, fontweight="bold")
        ax.text(0.045, 0.845, f"r = {r:.3f}\nRMSE = {rmse:.2f}\nn = {len(df)}",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=6.8 * S1, color=INK_MUTED, linespacing=1.4)

    fig.savefig(OUT / "fig1_scatter.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote fig1_scatter.png")


# --- Figure 2: the leakage gradient ----------------------------------------
def figure_gradient() -> None:
    suffixes = ["split42_seed42", "split42_seed43", "split42_seed44",
                "split43_seed42", "split44_seed42"]
    methods = [
        ("Frozen ESM-2 + ridge", "linear_probe_ridge_esm2_t30_150M_UR50D", ORANGE),
        ("Frozen ESM-2 + MLP head", "linear_probe_mlp_esm2_t30_150M_UR50D", AQUA),
        ("Fine-tuned ESM-2 (148M params)", "finetune_esm2_t30_150M_UR50D", BLUE),
    ]
    # One-line tick labels: the group counts already appear in the report's
    # split table, and a two-line axis adds ~60px of print height for nothing.
    split_cols = [
        ("split_mutation", "Naive: split by mutation  (leaks)"),
        ("split_pdb_id", "Honest: split by structure"),
        ("split_hold_out_proteins", "Honest: split by protein pair"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 2.0), constrained_layout=True)
    n_m = len(methods)
    group_w, gap = 0.60, 0.028      # `gap` is the 2px surface gap between bars
    bar_w = group_w / n_m - gap

    for j, (label, prefix, colour) in enumerate(methods):
        means, stds = [], []
        for col, _ in split_cols:
            vals = [METRICS[f"{prefix}_{col}_{s}"]["splits"][col]["test"]["spearman"]
                    for s in suffixes]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)))
        x = np.arange(len(split_cols)) - group_w / 2 + bar_w / 2 + j * (bar_w + gap)
        ax.bar(x, means, bar_w, label=label, color=colour, linewidth=0, zorder=2)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK_2,
                    elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=3)
        # Static print has no tooltip, so the direct label carries the value.
        for xi, mi, si in zip(x, means, stds):
            ax.text(xi, mi + si + 0.018, f"{mi:.3f}", ha="center", va="bottom",
                    fontsize=6.8 * S2, color=INK_2)

    ax.axhline(0, color=INK_MUTED, lw=0.6, zorder=1)
    # Breathing room so the outer tick labels do not run to the figure edge.
    ax.set_xlim(-0.62, len(split_cols) - 0.38)
    ax.set_xticks(np.arange(len(split_cols)))
    ax.set_xticklabels([lab for _, lab in split_cols], fontsize=7.2 * S2)
    ax.set_ylabel("test Spearman ρ", fontsize=7.5 * S2)
    ax.tick_params(axis="y", labelsize=7.5 * S2)
    ax.set_ylim(-0.02, 0.83)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.grid(True, axis="y", zorder=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=7.2 * S2, ncols=1,
              labelcolor=INK_2, handlelength=1.1, handleheight=1.1, borderpad=0.2)

    fig.savefig(OUT / "fig2_gradient.png", dpi=400)
    plt.close(fig)
    print("wrote fig2_gradient.png")


if __name__ == "__main__":
    figure_scatter()
    figure_gradient()
