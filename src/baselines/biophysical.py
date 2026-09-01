"""Non-PLM baseline: ddG from substitution chemistry alone. No embeddings.

    python -m src.baselines.biophysical
    python -m src.baselines.biophysical --seed 43
    python -m src.baselines.biophysical --splits-file splits_seed43.csv

WHY THIS EXISTS
---------------
Every other method in this repo is ESM-2: zero-shot masked-marginal, a frozen
backbone with a ridge or MLP head, a full fine-tune, and LoRA. That makes the
obvious question unanswerable -- what does a model with NO language model in it
get? Without that number, "ESM-2 reaches 0.22 on an honest split" has nothing to
be measured against except zero, and a reader cannot tell whether the
representation is doing any work at all.

The features here are the ones a biochemist would write down in five minutes:
what the residue was, what it became, and how much that changes size, charge and
hydrophobicity. No structure, no evolution, no learned representation.

Both outcomes are informative, which is the point of running it:

  - If this MATCHES the frozen probe on an honest split, then on this dataset,
    at this size, ESM-2's representation adds nothing over substitution
    chemistry once the evaluation stops rewarding memorisation.
  - If it is clearly WORSE, the representation is carrying real interface
    information, and every claim made for the PLM methods gets firmer.

THREE FAMILIES, AND ONE OF THEM IS NOT SEQUENCE-ONLY
-----------------------------------------------------
  biophys_ridge    sequence-only  -- comparable to the PLM methods
  biophys_gbm      sequence-only  -- comparable to the PLM methods
  biophys_gbm_loc  + interface_location

`interface_location` is SKEMPI's own COR/RIM/SUP/SUR/INT annotation, derived
from the structure. A model using it is NOT sequence-only and must never be
reported as though it were. It is included because it answers a question the
README raises as a limitation and cannot otherwise answer without parsing PDBs:
how much does merely knowing WHERE the mutation sits buy you?

ONE MODEL PER SPLIT DEFINITION
------------------------------
Same rule as everywhere else in this repo, and for the same reason: the three
definitions disagree about which rows are held out, so a model fitted on one has
contaminated numbers on the other two. Each fit passes `trained_on=` and the
harness marks the others INVALID rather than printing them as if they counted.

ON THE LOSS, so the comparison is not overstated. Ridge and the GBMs minimise
squared error; the MLP probe and the fine-tuned model use Huber. The existing
ridge probe is squared-error too, so `biophys_ridge` against `linear_probe_ridge`
is like-for-like. Against the Huber-trained models it is not exactly, but
Spearman is a rank metric and the gap between these methods is far larger than
any plausible loss-function effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import MAX_TOKENS, apply_length_policy, load_frames
from src.utils import set_seed

RESULTS_DIR = Path("results")
SPLIT_COLUMNS = ["split_mutation", "split_pdb_id", "split_hold_out_proteins"]

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
INTERFACE_LOCATIONS = ["COR", "RIM", "SUP", "SUR", "INT"]

# Kyte-Doolittle hydropathy. Positive is hydrophobic.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Residue volume in cubic angstroms (Zamyatnin 1972).
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

# Formal charge at pH 7. Histidine is given a partial charge; its pKa sits close
# enough to 7 that calling it 0 or +1 is equally wrong.
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}

AROMATIC = set("FWYH")
POLAR = set("STNQYCHKRDE")


def build_features(df: pd.DataFrame, include_location: bool) -> tuple[np.ndarray, list[str]]:
    """Substitution chemistry as a dense matrix. Returns (features, column names).

    Deltas are always mutant minus wild-type, matching the sign convention of
    ddG itself, so a positive volume delta means the mutation made the residue
    bigger. Keeping that consistent means a coefficient's sign can be read
    directly rather than mentally inverted.
    """
    wt, mut = df["wt_aa"], df["mut_aa"]

    def prop(series: pd.Series, table: dict, default: float = 0.0) -> np.ndarray:
        return series.map(lambda a: table.get(a, default)).to_numpy(dtype=np.float64)

    columns: dict[str, np.ndarray] = {}

    from Bio.Align import substitution_matrices
    blosum = substitution_matrices.load("BLOSUM62")
    columns["blosum62"] = np.array(
        [float(blosum[w, m]) if (w, m) in blosum else 0.0 for w, m in zip(wt, mut)]
    )

    for name, table in (("hydropathy", HYDROPATHY), ("volume", VOLUME), ("charge", CHARGE)):
        wt_val, mut_val = prop(wt, table), prop(mut, table)
        columns[f"d_{name}"] = mut_val - wt_val
        columns[f"abs_d_{name}"] = np.abs(mut_val - wt_val)
        columns[f"wt_{name}"] = wt_val          # context, not just the change:
        # losing a buried leucine is not the same event as gaining one at a rim.

    for name, members in (("aromatic", AROMATIC), ("polar", POLAR)):
        columns[f"wt_{name}"] = wt.isin(members).to_numpy(dtype=np.float64)
        columns[f"mut_{name}"] = mut.isin(members).to_numpy(dtype=np.float64)

    # Alanine scanning is 58% of this dataset, so "to alanine" is worth its own
    # column rather than leaving the model to infer it from the one-hots.
    columns["to_alanine"] = (mut == "A").to_numpy(dtype=np.float64)
    columns["involves_proline"] = ((wt == "P") | (mut == "P")).to_numpy(dtype=np.float64)
    columns["involves_glycine"] = ((wt == "G") | (mut == "G")).to_numpy(dtype=np.float64)

    for aa in AMINO_ACIDS:
        columns[f"wt_is_{aa}"] = (wt == aa).to_numpy(dtype=np.float64)
        columns[f"mut_is_{aa}"] = (mut == aa).to_numpy(dtype=np.float64)

    if include_location:
        for loc in INTERFACE_LOCATIONS:
            columns[f"loc_{loc}"] = (df["interface_location"] == loc).to_numpy(dtype=np.float64)

    names = list(columns)
    return np.column_stack([columns[n] for n in names]).astype(np.float32), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-file", default="splits.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    # The SAME length policy as every other method, even though nothing here
    # tokenises anything. Dropping it would score these models on 15 rows the
    # PLM methods never saw, and the comparison would stop being like-for-like.
    df = apply_length_policy(load_frames(splits_file=args.splits_file),
                             max_tokens=args.max_tokens)
    targets = df["ddg"].to_numpy(dtype=np.float32)

    seq_features, seq_names = build_features(df, include_location=False)
    loc_features, loc_names = build_features(df, include_location=True)
    print(f"\n  sequence-only features : {seq_features.shape}")
    print(f"  + interface_location   : {loc_features.shape}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV

    from src.evaluate import evaluate_predictions, format_report, save_report

    split_seed = (args.splits_file.replace("splits_seed", "").replace(".csv", "")
                  if args.splits_file != "splits.csv" else "42")

    for split_column in SPLIT_COLUMNS:
        print(f"\n{'=' * 72}\n  training on {split_column}\n{'=' * 72}")
        train = (df[split_column] == "train").to_numpy()
        for part in ("train", "val", "test"):
            print(f"    {part:<6} {(df[split_column] == part).sum():>5} rows")

        fitted: dict[str, np.ndarray] = {}

        # RidgeCV picks alpha by leave-one-out on TRAIN only -- the val split is
        # never touched, exactly as in the existing ridge probe.
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 25))
        ridge.fit(seq_features[train], targets[train])
        fitted["ridge"] = ridge.predict(seq_features)
        print(f"      ridge alpha = {ridge.alpha_:.2f}")

        # The GBMs early-stop on a fraction carved out of TRAIN, so val stays
        # untouched here too. That differs from how the neural models select
        # (best val Spearman), but it keeps this baseline from getting a look at
        # data the comparison assumes is held out.
        for label, features in (("gbm", seq_features), ("gbm_loc", loc_features)):
            model = HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
                random_state=args.seed,
            )
            model.fit(features[train], targets[train])
            fitted[label] = model.predict(features)
            print(f"      {label:<8} stopped at {model.n_iter_} of 400 trees")

        for label, values in fitted.items():
            name = f"biophys_{label}_{split_column}_split{split_seed}_seed{args.seed}"
            predictions = pd.DataFrame({"uid": df["uid"], "y_pred": values})
            predictions.to_csv(args.results_dir / f"preds_{name}.csv", index=False)
            report = evaluate_predictions(
                predictions, name=name, trained_on=split_column,
                splits_file=args.splits_file,
            )
            print(format_report(report))
            save_report(report, args.results_dir)

    print(f"\n  metrics -> {args.results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
