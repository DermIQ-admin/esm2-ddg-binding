"""Train/val/test splits — grouped (primary) and mutation-level (comparison).


THE CORE METHODOLOGICAL COMMITMENT OF THIS PROJECT:

    No protein complex may appear in more than one split.

Mutations of the same complex are highly correlated. Splitting at the mutation
level lets the model memorize a complex from its training mutations and then
"predict" held-out mutations of that same complex — an inflated number from a
leaky split. Splitting by complex prevents that.

We build THREE splits and report all three, which turns a two-point
contrast into a leakage gradient:

    mutation           naive row-level split. Leaks by construction. The
                       deliberate contrast case, never the headline number.
    pdb_id             group by PDB structure. 315 groups. The literal reading
                       of "complex-level".
    hold_out_proteins  SKEMPI's own curated grouping, 154 groups. Strictest:
                       it also catches the same protein pair appearing under
                       several different PDB ids, which pdb_id misses.

Expect the numbers to fall in that order (mutation best, hold_out_proteins
worst). The gap is the result, not an embarrassment.

NOTE ON PROPORTIONS: the split fractions apply to GROUPS, not rows. Groups vary
a lot in size — the ten largest hold_out_proteins groups hold about two-thirds
of all rows — so a 15% split of groups will not give exactly 15% of rows. That
is inherent to grouped splitting, and `summarize()` prints both so the actual
row balance is visible rather than assumed.

Usage:
    python -m src.data.splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Grouped keys are the honest splits; "mutation" is the leaky contrast case.
GROUP_KEYS = ["pdb_id", "hold_out_proteins"]
ALL_KEYS = ["mutation", *GROUP_KEYS]

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42


def split_by_group(
    groups: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Assign each row a split, splitting the UNIQUE GROUPS rather than rows.

    This is the whole trick, and it's why leakage is structurally impossible
    here rather than merely tested for: we partition the set of group names,
    then every row inherits its group's label. A group cannot land in two
    splits because it is assigned exactly once.

    WHY THIS ISN'T A PLAIN train_test_split OVER THE GROUP NAMES
    ------------------------------------------------------------
    The obvious approach splits the group names uniformly at random. That is
    correct in principle, but SKEMPI's groups are enormously uneven — the ten
    largest `hold_out_proteins` groups hold about two-thirds of all rows, and
    group sizes span three orders of magnitude. Sampling groups uniformly
    therefore gave a validation set of 227 rows (4.7%) where 15% was intended,
    which is far too thin to select a model on.

    So we keep groups atomic and honour the intended proportions in the
    dimension that actually matters for training — rows — by greedy
    bin-packing: shuffle the groups (seeded), take them largest-first, and give
    each to whichever split is furthest below its target row count. Placing the
    big groups first is what makes it work; leaving them to the end would strand
    them in whichever split had room.

    The invariant is untouched: every group is still assigned exactly once.
    """
    sizes = groups.dropna().value_counts()

    # Shuffle before the size sort so `random_state` still changes the result.
    # Many groups share a size, and pandas' sort is stable, so the shuffled
    # order survives as the tie-break among equal-sized groups.
    shuffled = sizes.sample(frac=1.0, random_state=random_state)
    ordered = shuffled.sort_values(ascending=False, kind="stable")

    total = int(sizes.sum())
    targets = {
        "train": (1.0 - test_size - val_size) * total,
        "val": val_size * total,
        "test": test_size * total,
    }
    assigned = {"train": 0, "val": 0, "test": 0}

    lookup: dict[object, str] = {}
    for name, size in ordered.items():
        # "Furthest below target" in absolute rows, so a big group lands where
        # it does least damage to the overall balance.
        split = max(targets, key=lambda s: targets[s] - assigned[s])
        lookup[name] = split
        assigned[split] += int(size)

    return groups.map(lookup)


def split_by_row(
    index: pd.Index,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Naive mutation-level split. Ignores complex identity entirely.

    This exists ONLY as the contrast case. It is expected to leak — mutations
    of the same complex will land in both train and test — and
    `tests/test_splits.py` asserts that it does, so the comparison in the
    README is a real difference rather than two labels on one computation.
    """
    train_idx, test_idx = train_test_split(
        index, test_size=test_size, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size / (1 - test_size), random_state=random_state
    )

    assignment = pd.Series("train", index=index)
    assignment[val_idx] = "val"
    assignment[test_idx] = "test"
    return assignment


def make_splits(mutations: pd.DataFrame,
                random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Build every split and return a uid-keyed table of assignments.

    `random_state` re-partitions the complexes. Varying it is how we measure the
    variance that training seeds cannot reach: val and test hold DIFFERENT
    COMPLEXES, not different samples of the same ones, so "which complexes
    landed where" is a distinct and probably larger source of variance than
    training stochasticity. See src/replicates.py.
    """
    out = pd.DataFrame({"uid": mutations["uid"]})
    out["split_mutation"] = split_by_row(mutations.index, random_state=random_state).values
    for key in GROUP_KEYS:
        out[f"split_{key}"] = split_by_group(mutations[key], random_state=random_state).values
    return out


def check_no_leakage(mutations: pd.DataFrame, splits: pd.DataFrame) -> None:
    """Assert the invariant for every grouped split. Raises on violation."""
    merged = mutations.merge(splits, on="uid", validate="one_to_one")
    for key in GROUP_KEYS:
        col = f"split_{key}"
        by_split = {s: set(g[key].dropna()) for s, g in merged.groupby(col)}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            overlap = by_split.get(a, set()) & by_split.get(b, set())
            assert not overlap, f"{key}: {len(overlap)} groups in both {a} and {b}: {sorted(overlap)[:5]}"


def summarize(mutations: pd.DataFrame, splits: pd.DataFrame) -> None:
    merged = mutations.merge(splits, on="uid", validate="one_to_one")
    for key in ALL_KEYS:
        col = f"split_{key}"
        group_col = key if key in GROUP_KEYS else "pdb_id"
        print(f"\n  {col}")
        for name in ("train", "val", "test"):
            sub = merged[merged[col] == name]
            pct = 100 * len(sub) / len(merged)
            extra = "" if key == "mutation" else f"  groups={sub[group_col].nunique()}"
            print(f"    {name:<6} rows={len(sub):>5} ({pct:>4.1f}%){extra}"
                  f"   ddG mean {sub['ddg'].mean():+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE,
                        help="re-partition the complexes; writes splits_seed{N}.csv "
                             "for anything other than the default")
    args = parser.parse_args()

    mutations = pd.read_csv(args.processed_dir / "mutations.csv")
    splits = make_splits(mutations, random_state=args.random_state)

    check_no_leakage(mutations, splits)
    print("Leakage check passed: no group appears in two splits.")

    # The default keeps the committed filename; alternates are suffixed so the
    # canonical splits.csv is never silently overwritten by a replicate run.
    out_path = args.processed_dir / (
        "splits.csv" if args.random_state == RANDOM_STATE
        else f"splits_seed{args.random_state}.csv")
    splits.to_csv(out_path, index=False)
    print(f"Wrote {len(splits)} assignments -> {out_path}")

    summarize(mutations, splits)


if __name__ == "__main__":
    main()
