"""Train/val/test splits — complex-level (primary) and mutation-level (comparison).

Implements PLAN.md section 6.5.

THE CORE METHODOLOGICAL COMMITMENT OF THIS PROJECT:

    No protein complex may appear in more than one split.

Mutations of the same complex are highly correlated. Splitting at the mutation
level lets the model memorize a complex from its training mutations and then
"predict" held-out mutations of that same complex — an inflated number from a
leaky split. Splitting by complex prevents that.

Both splits get built and BOTH get reported. The gap between them is the
demonstration that we understand leakage, and per section 9 it is arguably
more valuable than either number on its own. The mutation-level split exists
purely as the contrast case; it is never the headline result.

TODO(session 2): implement.
  - split_by_complex(df, test_size=0.15, val_size=0.15/0.85, random_state=42)
      -> group by complex id, split the UNIQUE complexes, then select rows
  - split_by_mutation(df, ...) -> naive random row-level split, for comparison
  - Assert pairwise-empty complex intersections between all three splits
  - Persist to data/processed/ so results are reproducible without re-parsing

The inline asserts from section 6.5 are mirrored as real tests in
tests/test_splits.py — that file is the enforcement, this is the logic.
"""
