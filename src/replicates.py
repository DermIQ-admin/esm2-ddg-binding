"""Aggregate replicate runs. PLAN.md section 9 (error bars), section 13.1 (antibodies).

Prints four things:

  summarize()           mean +/- std per method, split by variance source
  paired_comparison()   finetune - probe on ALL test rows, paired, with CI and p
  summarize_antibody()  the AB/AG subset (section 13.1), pooled over replicates
  paired_comparison()   the same paired test restricted to AB/AG rows

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
from scipy import stats

# Single source of truth: if evaluate.py ever changes the minimum row count,
# the "not reported" explanation below must change with it, not drift.
from src.evaluate import MIN_ROWS_FOR_METRIC

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
        # evaluate.py writes error_analysis.antibody_antigen only when the test
        # set holds at least MIN_ROWS_FOR_METRIC (10) AB/AG rows. It is absent
        # for split_hold_out_proteins, whose test set lands only 6 of them --
        # summarize_antibody() says so out loud rather than dropping the split.
        antibody = report.get("error_analysis", {}).get("antibody_antigen") or {}
        runs.append({
            "method": match["method"],
            "split_seed": int(match["split_seed"]),
            "train_seed": int(match["train_seed"]),
            "spearman": test["spearman"],
            "rmse": test.get("rmse"),
            "antibody_spearman": antibody.get("spearman"),
            "antibody_n": antibody.get("n"),
        })
    return runs


def method_family(method: str) -> tuple[str, str]:
    """('finetune', 'pdb_id') from 'finetune_esm2_t30_150M_UR50D_split_pdb_id'.

    Splits on the backbone tag so the family can be compared EXACTLY. An earlier
    version matched with startswith(), which silently made 'finetune_lora' a
    'finetune' -- last write winning by dict order. On a mocked-up LoRA run that
    turned a true -0.02 difference into a reported +0.680 at p=0.000 with a
    zero-width CI: maximally confident and completely wrong. PLAN.md 7.4 puts a
    LoRA run on the roadmap, so that collision was scheduled, not theoretical.
    """
    head, _, column = method.partition("_split_")
    return head.partition("_esm2_")[0], column


def short(name: str) -> str:
    """Strip the shared backbone tag; it is constant and eats the column width."""
    return (name.replace("_esm2_t30_150M_UR50D", "")
                .replace("_esm2_t12_35M_UR50D", "")
                .replace("split_", ""))


def summarize(runs: list[dict]) -> None:
    by_method = defaultdict(list)
    for r in runs:
        by_method[r["method"]].append(r)

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


def summarize_antibody(runs: list[dict]) -> None:
    """PLAN.md section 13.1 -- the antibody/antigen subset.

    evaluate.py has been computing this all along into
    error_analysis.antibody_antigen; nothing read it. It is the subset that
    matters most for an antibody team, so it gets a table rather than staying
    buried in per-run JSON.

    Reported POOLED over all replicates, not split into the two variance
    sources the table above separates. The subset is small enough that
    3 training seeds and 3 split seeds give 5 unique runs, and splitting 5 into
    two groups of 3 would put error bars on error bars. The cost is that this
    std mixes both sources, so read it as "how much does this number move",
    not as a training-reproducibility claim.
    """
    with_ab = [r for r in runs if r["antibody_spearman"] is not None]
    if not with_ab:
        print("\nNo antibody-subset metrics in metrics.json.")
        return

    by_method = defaultdict(list)
    for r in with_ab:
        by_method[r["method"]].append(r)

    print(f"\n{'ANTIBODY / ANTIGEN SUBSET  (PLAN.md 13.1)':<91}")
    print(f"\n{'method':<40}{'AB/AG rows':<16}{'n':>3}"
          f"{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    print("-" * 91)
    for method in sorted(by_method):
        group = by_method[method]
        v = np.array([r["antibody_spearman"] for r in group])
        counts = {r["antibody_n"] for r in group}
        # Split seeds re-partition the complexes, so the number of AB/AG rows
        # in the test set is NOT constant across replicates. Show the range.
        rows = str(min(counts)) if len(counts) == 1 else f"{min(counts)}-{max(counts)}"
        print(f"  {short(method):<38}{rows:<16}{len(v):>3}"
              f"{v.mean():>8.3f}{v.std(ddof=1) if len(v) > 1 else 0:>8.3f}"
              f"{v.min():>8.3f}{v.max():>8.3f}")

    # Never let a split vanish silently -- say which are missing and why.
    missing = sorted({short(r["method"]) for r in runs} - {short(m) for m in by_method})
    if missing:
        print(f"\n  NOT REPORTED (test set holds fewer than {MIN_ROWS_FOR_METRIC} AB/AG rows,")
        print(f"  so evaluate.py correctly omits the metric):")
        for m in missing:
            print(f"    {m}")
        print("  hold_out_proteins lands only 6. Size-aware assignment sends the large,\n"
              "  well-studied antibody complexes to train, so the antibody claim can only\n"
              "  be made on pdb_id and on the leaky mutation split.")


def paired_comparison(
    runs: list[dict],
    metric: str,
    label: str,
    challenger: str = "finetune",
    baseline: str = "linear_probe_mlp",
) -> None:
    """Fine-tune minus frozen probe, PAIRED on identical partitions.

    Pairing is the whole point. Split variance exceeds training variance on
    most cells here, so comparing two unpaired means lets a lucky partition
    masquerade as a method difference -- which is exactly how the session-4
    headline survived a week before replicates killed it. Every difference
    below holds the partition AND the training seed fixed, so the only variable
    is the method.

    Five pairs is a small n and the t-interval is correspondingly wide. Read a
    significant result here as "worth reporting", not "settled".
    """
    indexed: dict[tuple, dict[str, float]] = defaultdict(dict)
    for r in runs:
        if r[metric] is None:
            continue
        family, column = method_family(r["method"])
        key = (column, r["split_seed"], r["train_seed"])
        # Exact match, NOT startswith -- see method_family's docstring.
        if family in (challenger, baseline):
            indexed[key][family] = r[metric]

    by_column: dict[str, list[float]] = defaultdict(list)
    for (column, _, _), pair in indexed.items():
        if challenger in pair and baseline in pair:
            by_column[column].append(pair[challenger] - pair[baseline])

    print(f"\n  {label}: {challenger} - {baseline}, paired")
    print(f"  {'split definition':<24}{'n':>3}{'mean diff':>11}"
          f"{'95% CI':>20}{'p':>8}")
    print("  " + "-" * 64)
    for column in sorted(by_column):
        d = np.array(by_column[column])
        if len(d) < 2:
            # Say so rather than dropping the row -- a split that quietly
            # vanishes from a results table reads as "not different".
            print(f"  {column:<24}{len(d):>3}   too few pairs to test")
            continue
        _, p = stats.ttest_1samp(d, 0.0)
        half = stats.t.ppf(0.975, len(d) - 1) * stats.sem(d)
        ci = f"[{d.mean() - half:+.3f}, {d.mean() + half:+.3f}]"
        print(f"  {column:<24}{len(d):>3}{d.mean():>+11.3f}{ci:>20}{p:>8.3f}")


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
    paired_comparison(runs, "spearman", "ALL TEST ROWS")
    summarize_antibody(runs)
    paired_comparison(runs, "antibody_spearman", "ANTIBODY / ANTIGEN ROWS ONLY")
    print("\n  The AB/AG subset was pre-specified in PLAN.md 13.1 before any results\n"
          "  existed, so it is a planned comparison rather than a subgroup found by\n"
          "  searching. It is still one subgroup out of several in error_analysis --\n"
          "  weigh it accordingly.")


if __name__ == "__main__":
    main()
