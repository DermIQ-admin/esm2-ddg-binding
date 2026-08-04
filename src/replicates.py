"""Aggregate replicate runs into mean +/- std. PLAN.md section 9 (error bars).

TWO SOURCES OF VARIANCE, REPORTED SEPARATELY
--------------------------------------------
They are not interchangeable, and conflating them would overstate confidence:

  TRAINING seed   head init, dropout, batch order. The split is IDENTICAL across
                  these runs, so this measures only "is one training run
                  reproducible". Varied via `train.py --seed`.

  SPLIT seed      re-partitions the complexes via `splits.py --random-state`.
                  val and test hold DIFFERENT COMPLEXES, not different samples
                  of the same ones, so "which complexes landed where" is a real
                  and probably larger source of variance — and one that training
                  seeds cannot reach at all.

The session-4 single runs showed a val->test drop of 0.269 -> 0.148 on
split_pdb_id. If that gap is mostly split variance, training-seed error bars
alone would look reassuringly tight and mean very little.

Run names encode both: <method>_<split_column>_split<S>_seed<T>.

Usage:
    python -m src.replicates
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
METRICS = REPO_ROOT / "results" / "metrics.json"
NAME_RE = re.compile(r"^(?P<method>.+)_split(?P<split_seed>\d+)_seed(?P<train_seed>\d+)$")


def load_runs() -> list[dict]:
    """Every replicate-tagged run in metrics.json, with its test Spearman."""
    data = json.loads(METRICS.read_text())
    runs = []
    for name, report in data.items():
        match = NAME_RE.match(name)
        if not match:
            continue  # a pre-replicate run, or the zero-shot baselines
        column = report.get("trained_on")
        test = report["splits"].get(column, {}).get("test", {})
        if test.get("spearman") is None:
            continue
        runs.append({
            "method": match["method"],
            "split_seed": int(match["split_seed"]),
            "train_seed": int(match["train_seed"]),
            "spearman": test["spearman"],
            "rmse": test.get("rmse"),
        })
    return runs


def summarize(runs: list[dict]) -> None:
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)

    # Strip the shared backbone tag; it is constant and eats the column width.
    def short(name: str) -> str:
        return (name.replace("_esm2_t30_150M_UR50D", "")
                    .replace("_esm2_t12_35M_UR50D", "")
                    .replace("split_", ""))

    print(f"\n{'method':<40}{'varying':<16}{'n':>3}"
          f"{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 91)
    for method in sorted(by_method):
        group = by_method[method]
        # Training variance: split held fixed at the canonical 42.
        train_group = [r["spearman"] for r in group if r["split_seed"] == 42]
        # Split variance: training seed held fixed at 42.
        split_group = [r["spearman"] for r in group if r["train_seed"] == 42]

        for label, values in (("training seeds", train_group), ("split seeds", split_group)):
            if len(values) < 2:
                continue
            v = np.array(values)
            print(f"  {short(method):<38}{label:<16}{len(v):>3}"
                  f"{v.mean():>8.3f}{v.std(ddof=1):>8.3f}{v.min():>8.3f}{v.max():>8.3f}")


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No replicate-tagged runs yet. Run train.py / linear_probe.py with "
              "--seed and --splits-file.")
        return
    print(f"{len(runs)} replicate runs in {METRICS}")
    summarize(runs)
    print("\n  'training seeds' holds the partition fixed and varies only training "
          "stochasticity.\n  'split seeds' holds training fixed and re-partitions the "
          "complexes. The second\n  is the one that decides whether a difference "
          "between methods is real.")


if __name__ == "__main__":
    main()
