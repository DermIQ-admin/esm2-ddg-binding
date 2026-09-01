"""Evaluation harness: metrics, all three splits, error analysis, plots.


Every model — zero-shot, frozen-embedding, fine-tuned — reports through this
one harness, so the numbers in the README are directly comparable rather than
three slightly different computations that happen to share a name.

THE INTERFACE
-------------
A model produces a two-column frame: `uid` and `y_pred`. That is all. This
module joins it against `mutations.csv` for the truth and `splits.csv` for the
assignments, so a model never has to know how the splits were built and can
never accidentally evaluate on the wrong rows.

    preds = pd.DataFrame({"uid": [...], "y_pred": [...]})
    report = evaluate_predictions(preds, name="zero_shot_35M")

METRICS
  PRIMARY: Spearman. Rank order is what matters downstream — you are ranking
  candidate mutations, not calibrating an absolute energy.
  Pearson and RMSE/MAE are reported ALONGSIDE it, never instead.

  `ranking_only=True` suppresses RMSE/MAE. The zero-shot baseline outputs a
  log-likelihood ratio, not kcal/mol; an RMSE between those two would be a
  number with no meaning, and printing it would invite exactly the comparison
  it cannot support.

THE LEAKAGE GRADIENT
--------------------
Every run is scored against all three split definitions:

    split_mutation           naive row-level. Leaks by construction.
    split_pdb_id             grouped by structure.
    split_hold_out_proteins  SKEMPI's curated grouping. Strictest.

Expect a trained model to look best on `split_mutation` and worst on
`split_hold_out_proteins`. That gap is the headline result of the project, not
an embarrassment.

A CONTROL WORTH UNDERSTANDING: the zero-shot baseline never trains, so it
cannot leak. Its numbers should come out roughly EQUAL across all three split
definitions. If they do, that establishes the three test sets are of similar
intrinsic difficulty — and therefore that any gap the fine-tuned model shows
later is caused by training on correlated complexes, not by the splits being
unequal to begin with. Run zero-shot first and this control is free.

Usage:
    python -m src.evaluate --predictions results/preds_zero_shot.csv --name zero_shot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SPLIT_COLUMNS = ["split_mutation", "split_pdb_id", "split_hold_out_proteins"]
SPLIT_NAMES = ["train", "val", "test"]

# Below this many rows a Spearman is noise, not a measurement.
MIN_ROWS_FOR_METRIC = 10
# Per-complex error analysis needs enough mutations to mean anything.
MIN_MUTATIONS_PER_COMPLEX = 10


# --------------------------------------------------------------------------
# Core metrics
# --------------------------------------------------------------------------

def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, ranking_only: bool = False
) -> dict:
    """Spearman (primary), Pearson, RMSE, MAE.

    Returns NaN rather than raising on a degenerate input — a constant
    prediction vector has undefined correlation, and that is a result worth
    seeing in the table (it usually means a dead model) rather than a crash
    halfway through a report.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    out: dict[str, float | int | None] = {"n": int(len(y_true))}

    if len(y_true) < MIN_ROWS_FOR_METRIC or np.std(y_pred) == 0 or np.std(y_true) == 0:
        out.update(spearman=float("nan"), pearson=float("nan"))
    else:
        out["spearman"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["pearson"] = float(stats.pearsonr(y_true, y_pred).statistic)

    if ranking_only:
        # See the module docstring: these units do not exist for a scale-free
        # score, so we omit them rather than print a meaningless number.
        out.update(rmse=None, mae=None)
    else:
        residual = y_pred - y_true
        out["rmse"] = float(np.sqrt(np.mean(residual ** 2)))
        out["mae"] = float(np.mean(np.abs(residual)))

    return out


def load_truth(processed_dir: Path = PROCESSED_DIR,
               splits_file: str = "splits.csv") -> pd.DataFrame:
    """uid -> ddg plus the metadata the error analysis needs."""
    mutations = pd.read_csv(processed_dir / "mutations.csv")
    splits = pd.read_csv(processed_dir / splits_file)
    columns = [
        "uid", "pdb_field", "pdb_id", "hold_out_type", "mutation",
        "wt_aa", "mut_aa", "interface_location", "ddg",
    ]
    return mutations[columns].merge(splits, on="uid", validate="one_to_one")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def evaluate_predictions(
    predictions: pd.DataFrame,
    name: str,
    ranking_only: bool = False,
    trained_on: str | None = None,
    processed_dir: Path = PROCESSED_DIR,
    splits_file: str = "splits.csv",
) -> dict:
    """Score one run against every split definition.

    `predictions` needs exactly two columns: `uid` and `y_pred`. Rows the model
    did not predict are simply absent — an inner join — but the coverage is
    reported so a silently-truncated prediction set cannot masquerade as a
    complete one.

    `trained_on` IS A CORRECTNESS GUARD, NOT A LABEL
    ------------------------------------------------
    The three split definitions disagree about which rows are held out. A row in
    `split_pdb_id`'s test set is very often in `split_mutation`'s TRAINING set.
    So a model trained against one definition cannot be honestly scored against
    the other two — its "test" numbers there include rows it was fitted on, and
    they would look wonderful.

    A model that never trains (the zero-shot baseline) has no such problem, so
    it passes `trained_on=None` and all three tables are valid — which is what
    makes it the free control described at the top of this module.

    Anything trained MUST pass the split column it was trained against. The
    other two are then computed but marked `contaminated: true` and printed
    struck out, so the leakage gradient has to be assembled from three separately
    trained models rather than faked from one.
    """
    if trained_on is not None and trained_on not in SPLIT_COLUMNS:
        raise ValueError(f"trained_on must be one of {SPLIT_COLUMNS}, got {trained_on!r}")
    required = {"uid", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions frame is missing {sorted(missing)}")
    if predictions["uid"].duplicated().any():
        raise ValueError("predictions contain duplicate uids")

    # MUST match the partition the model trained against, or every
    # split label in the report is wrong.
    truth = load_truth(processed_dir, splits_file)
    merged = truth.merge(predictions[["uid", "y_pred"]], on="uid", how="inner",
                         validate="one_to_one")

    unknown = set(predictions["uid"]) - set(truth["uid"])
    if unknown:
        raise KeyError(f"{len(unknown)} predicted uids are not in mutations.csv, "
                       f"e.g. {sorted(unknown)[:3]}")

    report = {
        "name": name,
        "ranking_only": ranking_only,
        "trained_on": trained_on,
        "coverage": {
            "predicted": int(len(merged)),
            "available": int(len(truth)),
            "fraction": round(len(merged) / len(truth), 4),
        },
        "overall": regression_metrics(merged["ddg"], merged["y_pred"], ranking_only),
        "splits": {},
    }

    for column in SPLIT_COLUMNS:
        report["splits"][column] = {
            split: regression_metrics(sub["ddg"], sub["y_pred"], ranking_only)
            for split, sub in merged.groupby(column)
            if split in SPLIT_NAMES
        }
        # See the docstring: valid only if nothing was trained, or if this is
        # the definition the model was actually trained against.
        report["splits"][column]["contaminated"] = (
            trained_on is not None and column != trained_on
        )

    # Error analysis follows the model's own split, not a fixed default —
    # analysing failures on rows the model trained on would measure memorisation.
    report["error_analysis"] = error_analysis(
        merged, split_column=trained_on or "split_pdb_id", ranking_only=ranking_only
    )
    return report


def error_analysis(
    merged: pd.DataFrame,
    split_column: str = "split_pdb_id",
    ranking_only: bool = False,
) -> dict:
    """Section 9's error analysis, computed on the held-out TEST rows only.

    Doing this on train would mostly measure how well the model memorised, which
    tells us nothing about where it actually fails.
    """
    test = merged[merged[split_column] == "test"]
    if len(test) < MIN_ROWS_FOR_METRIC:
        return {"note": f"too few test rows ({len(test)}) for error analysis"}

    out: dict = {"basis": f"{split_column} == 'test'", "n": int(len(test))}
    residual = (test["y_pred"] - test["ddg"]).abs()

    # --- worst complexes -------------------------------------------------
    by_complex = (
        test.assign(abs_error=residual)
        .groupby("pdb_field")
        .agg(n=("ddg", "size"), mae=("abs_error", "mean"), ddg_std=("ddg", "std"))
    )
    by_complex = by_complex[by_complex["n"] >= MIN_MUTATIONS_PER_COMPLEX]
    out["worst_complexes"] = [
        {"pdb_field": field, "n": int(r["n"]), "mae": round(float(r["mae"]), 3),
         "ddg_std": round(float(r["ddg_std"]), 3)}
        for field, r in by_complex.sort_values("mae", ascending=False).head(8).iterrows()
    ]

    # --- structurally unusual residues -----------------------------------
    # Proline breaks backbone geometry and glycine is uniquely flexible, so
    # mutations touching either are a plausible, physically-motivated failure
    # mode rather than an arbitrary slice of the data.
    categories = {
        "involves_proline": (test["wt_aa"] == "P") | (test["mut_aa"] == "P"),
        "involves_glycine": (test["wt_aa"] == "G") | (test["mut_aa"] == "G"),
        "to_alanine": test["mut_aa"] == "A",
        "other": ~((test["wt_aa"].isin(["P", "G"])) | (test["mut_aa"].isin(["P", "G", "A"]))),
    }
    out["by_mutation_type"] = {
        label: regression_metrics(test[mask]["ddg"], test[mask]["y_pred"], ranking_only)
        for label, mask in categories.items()
    }

    # --- interface location (SKEMPI's own annotation) ---------------------
    # COR = core, SUP = support, RIM = rim, INT = interior, SUR = surface.
    # A model that understands binding should do best in the core.
    out["by_interface_location"] = {
        str(loc): regression_metrics(sub["ddg"], sub["y_pred"], ranking_only)
        for loc, sub in test.groupby("interface_location")
        if len(sub) >= MIN_ROWS_FOR_METRIC
    }

    # --- large-effect mutations ------------------------------------------
    # 11% of the dataset exceeds |ddG| > 3. These are the ones that matter for
    # a real screen, and the ones Huber deliberately down-weights.
    large = test[test["ddg"].abs() > 3]
    out["large_effect"] = {
        "threshold_kcal_per_mol": 3.0,
        **regression_metrics(large["ddg"], large["y_pred"], ranking_only),
    }

    # --- antibody subset (SKEMPI labels it, so this is free) --------------
    ab = test[test["hold_out_type"].astype(str).str.contains("AB/AG", na=False)]
    if len(ab) >= MIN_ROWS_FOR_METRIC:
        out["antibody_antigen"] = regression_metrics(ab["ddg"], ab["y_pred"], ranking_only)

    return out


def format_report(report: dict) -> str:
    """Human-readable table. The leakage gradient reads down each column."""
    lines = [
        f"\n{'=' * 72}",
        f"  {report['name']}",
        f"{'=' * 72}",
        f"  coverage: {report['coverage']['predicted']} / "
        f"{report['coverage']['available']} rows "
        f"({100 * report['coverage']['fraction']:.1f}%)",
    ]
    if report["ranking_only"]:
        lines.append("  ranking-only output: RMSE/MAE omitted (not in kcal/mol)")

    header = f"  {'split definition':<26}{'set':<7}{'n':>6}{'spearman':>11}{'pearson':>10}{'rmse':>8}{'mae':>8}"
    lines += ["", header, "  " + "-" * (len(header) - 2)]

    def cell(value, width, fmt):
        if value is None:
            return " " * (width - 1) + "-"
        if isinstance(value, float) and np.isnan(value):
            return " " * (width - 3) + "n/a"
        return f"{value:>{width}{fmt}}"

    for column in SPLIT_COLUMNS:
        contaminated = report["splits"].get(column, {}).get("contaminated", False)
        for i, split in enumerate(SPLIT_NAMES):
            metrics = report["splits"].get(column, {}).get(split)
            if metrics is None:
                continue
            label = column if i == 0 else ""
            row = (
                f"  {label:<26}{split:<7}{metrics['n']:>6}"
                f"{cell(metrics['spearman'], 11, '.3f')}"
                f"{cell(metrics['pearson'], 10, '.3f')}"
                f"{cell(metrics.get('rmse'), 8, '.3f')}"
                f"{cell(metrics.get('mae'), 8, '.3f')}"
            )
            # Marked, not hidden: seeing the inflated number next to the honest
            # one is the whole pedagogical point of the split comparison.
            lines.append(row + ("   <- INVALID, trained on other rows" if contaminated and split == "test" else ""))
        lines.append("")

    if report.get("trained_on"):
        lines.append(f"  trained against {report['trained_on']}; the other two split "
                     f"definitions are marked INVALID because their\n  test rows overlap "
                     f"this model's training set. Build the leakage gradient from three "
                     f"separately\n  trained models, never from one.\n")

    ea = report.get("error_analysis", {})
    if "by_mutation_type" in ea:
        lines.append(f"  error analysis on {ea['basis']}  (n={ea['n']})")
        lines.append(f"    {'slice':<26}{'n':>6}{'spearman':>11}")
        for label, m in ea["by_mutation_type"].items():
            lines.append(f"    {label:<26}{m['n']:>6}{cell(m['spearman'], 11, '.3f')}")
        if ea.get("worst_complexes"):
            lines.append(f"\n    worst complexes by MAE (>= {MIN_MUTATIONS_PER_COMPLEX} mutations)")
            for c in ea["worst_complexes"][:5]:
                lines.append(f"      {c['pdb_field']:<20} n={c['n']:<4} "
                             f"mae={c['mae']:.2f}  ddG std={c['ddg_std']:.2f}")

    return "\n".join(lines)


def save_report(report: dict, results_dir: Path = RESULTS_DIR) -> Path:
    """Merge this run into results/metrics.json, keyed by run name.

    Merging rather than overwriting means the zero-shot, linear-probe and
    fine-tuned numbers accumulate into one file that the README can quote from,
    and re-running one model does not wipe the others.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.json"

    all_reports = json.loads(path.read_text()) if path.exists() else {}
    all_reports[report["name"]] = report
    path.write_text(json.dumps(all_reports, indent=2))
    return path


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_predictions(
    predictions: pd.DataFrame,
    name: str,
    split_column: str = "split_pdb_id",
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> Path:
    """Predicted vs true scatter, train / val / test side by side."""
    import matplotlib
    matplotlib.use("Agg")  # no display on a headless run; write straight to file
    import matplotlib.pyplot as plt

    merged = load_truth(processed_dir).merge(predictions[["uid", "y_pred"]], on="uid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, split in zip(axes, SPLIT_NAMES):
        sub = merged[merged[split_column] == split]
        if sub.empty:
            continue
        m = regression_metrics(sub["ddg"], sub["y_pred"])
        ax.scatter(sub["ddg"], sub["y_pred"], s=8, alpha=0.35, edgecolors="none")
        ax.axhline(0, lw=0.5, color="grey")
        ax.axvline(0, lw=0.5, color="grey")
        ax.set_title(f"{split}  (n={m['n']})\nSpearman {m['spearman']:.3f}")
        ax.set_xlabel("true ddG (kcal/mol)")
    axes[0].set_ylabel("predicted")

    fig.suptitle(f"{name} — {split_column}")
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}_{split_column}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="CSV with columns uid,y_pred")
    parser.add_argument("--name", required=True, help="run name, keys results/metrics.json")
    parser.add_argument("--ranking-only", action="store_true",
                        help="output is a score, not kcal/mol — omit RMSE/MAE")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    report = evaluate_predictions(predictions, name=args.name,
                                  ranking_only=args.ranking_only)
    print(format_report(report))
    print(f"\n  metrics -> {save_report(report)}")

    if not args.no_plot:
        print(f"  figure  -> {plot_predictions(predictions, args.name)}")


if __name__ == "__main__":
    main()
