"""Enforce the no-leakage invariant on the grouped splits.

Implements PLAN.md section 6.5, which asks for the inline asserts to become a
real pytest rather than a comment in a script.

    THE INVARIANT: no protein complex may appear in more than one split.

This is the most important test in the repo. The headline result is only
meaningful if it holds — a leaky split inflates the number and the project's
stated methodological point evaporates.

Most tests here run on SYNTHETIC data so they work on a fresh clone with no
downloads. The tests that need the real dataset are marked and skip cleanly
when `data/processed/` hasn't been built yet.

    python -m pytest tests/ -v
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.splits import (
    GROUP_KEYS,
    check_no_leakage,
    make_splits,
    split_by_group,
    split_by_row,
)

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """40 complexes with a deliberately uneven 1-to-12 mutations each.

    Uneven group sizes are the point: equal-sized groups would hide the fact
    that splitting groups does not split rows proportionally.
    """
    rows = []
    for c in range(40):
        for m in range((c % 12) + 1):
            rows.append(
                {
                    "uid": f"C{c}|M{m}|0",
                    "pdb_id": f"PDB{c}",
                    "hold_out_proteins": f"PROT{c // 2}",  # 2 structures per protein
                    "ddg": float(m),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# The invariant
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", GROUP_KEYS)
def test_no_group_crosses_splits(synthetic, key):
    """The three assertions from PLAN.md section 6.5, as a real test."""
    assignment = split_by_group(synthetic[key])
    members = {
        name: set(synthetic.loc[assignment == name, key])
        for name in ("train", "val", "test")
    }
    assert not members["train"] & members["val"]
    assert not members["train"] & members["test"]
    assert not members["val"] & members["test"]


def test_check_no_leakage_accepts_a_clean_split(synthetic):
    check_no_leakage(synthetic, make_splits(synthetic))


def test_check_no_leakage_catches_a_planted_leak(synthetic):
    """The guard must actually fire — a test that can't fail proves nothing.

    We deliberately move one row into `test` while leaving its group-mates in
    `train`, which is exactly what a mutation-level split does accidentally.
    """
    key = GROUP_KEYS[0]
    col = f"split_{key}"
    splits = make_splits(synthetic)

    # The group must have at least TWO rows in train, so that moving one of
    # them leaves siblings behind and genuinely straddles the split. Moving a
    # singleton group's only row just relocates the whole group — no leak, and
    # the test would pass while proving nothing.
    in_train = synthetic.loc[splits[col] == "train", key]
    multi_row = in_train.value_counts()
    group = multi_row[multi_row >= 2].index[0]

    victim = synthetic.index[(synthetic[key] == group) & (splits[col] == "train")][0]
    splits.loc[victim, col] = "test"

    with pytest.raises(AssertionError, match="in both"):
        check_no_leakage(synthetic, splits)


# --------------------------------------------------------------------------
# Partitioning and determinism
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", GROUP_KEYS)
def test_every_row_assigned_exactly_once(synthetic, key):
    assignment = split_by_group(synthetic[key])
    assert assignment.notna().all(), "some rows got no split"
    assert set(assignment.unique()) <= {"train", "val", "test"}
    assert len(assignment) == len(synthetic), "rows lost or duplicated"


def test_splits_table_covers_every_uid(synthetic):
    splits = make_splits(synthetic)
    assert len(splits) == len(synthetic)
    assert splits["uid"].is_unique
    assert set(splits["uid"]) == set(synthetic["uid"])


@pytest.mark.parametrize("key", GROUP_KEYS)
def test_split_is_deterministic(synthetic, key):
    """Same seed, same split — otherwise results aren't reproducible."""
    first = split_by_group(synthetic[key], random_state=42)
    second = split_by_group(synthetic[key], random_state=42)
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("key", GROUP_KEYS)
def test_different_seed_gives_a_different_split(synthetic, key):
    """Guards against a split that ignores random_state and only looks stable."""
    a = split_by_group(synthetic[key], random_state=0)
    b = split_by_group(synthetic[key], random_state=999)
    assert not a.equals(b)


# --------------------------------------------------------------------------
# The contrast case
# --------------------------------------------------------------------------

def test_mutation_level_split_does_leak(synthetic):
    """The naive split is EXPECTED to leak — assert that it genuinely does.

    Without this, `split_mutation` and `split_pdb_id` could quietly be
    computing the same thing, and the README's headline comparison would be
    two labels on one number rather than a real methodological contrast.
    """
    assignment = split_by_row(synthetic.index)
    train_groups = set(synthetic.loc[assignment == "train", "pdb_id"])
    test_groups = set(synthetic.loc[assignment == "test", "pdb_id"])
    assert train_groups & test_groups, (
        "the naive split shared no complexes between train and test — "
        "it is supposed to leak, so something is wrong"
    )


# --------------------------------------------------------------------------
# The real dataset
# --------------------------------------------------------------------------

@pytest.mark.skipif(
    not (PROCESSED / "mutations.csv").exists(),
    reason="run src.data.parse_skempi and src.data.splits first",
)
def test_real_dataset_has_no_leakage():
    mutations = pd.read_csv(PROCESSED / "mutations.csv")
    splits = pd.read_csv(PROCESSED / "splits.csv")
    check_no_leakage(mutations, splits)


@pytest.mark.skipif(
    not (PROCESSED / "mutations.csv").exists(),
    reason="run src.data.parse_skempi first",
)
def test_real_mutation_index_matches_wild_type_residue():
    """Every stored mutation_index must point at the recorded WT residue.

    This is the same check `apply_mutation` makes while parsing, re-run against
    the persisted files. It catches the case where complexes.csv and
    mutations.csv were regenerated out of step with each other.
    """
    mutations = pd.read_csv(PROCESSED / "mutations.csv")
    complexes = pd.read_csv(PROCESSED / "complexes.csv")
    sequences = dict(zip(complexes["pdb_field"], complexes["sequence"]))

    bad = [
        row.uid
        for row in mutations.itertuples()
        if sequences[row.pdb_field][row.mutation_index] != row.wt_aa
    ]
    assert not bad, f"{len(bad)} rows point at the wrong residue, e.g. {bad[:5]}"
