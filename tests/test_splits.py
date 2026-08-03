"""Enforce the no-leakage invariant on the complex-level split.

Implements PLAN.md section 6.5, which asks for the inline asserts to become a
real pytest rather than a comment in a script.

    THE INVARIANT: no protein complex may appear in more than one split.

This is the single most important test in the repo. The headline result of the
whole project is only meaningful if this holds — a leaky split inflates the
number and the project's stated methodological point evaporates.

TODO(session 2): implement, once src/data/splits.py exists.
  - test_no_complex_crosses_splits: pairwise-empty complex intersections
    between train/val/test (the three asserts from section 6.5)
  - test_splits_partition_the_data: no rows lost, no rows duplicated
  - test_split_is_deterministic: same random_state -> same split
  - test_mutation_level_split_does_leak: the naive split is EXPECTED to share
    complexes across train/test. Asserting that it does confirms the two split
    strategies genuinely differ, so the comparison in the README is real and
    not two labels on the same computation.
"""

import pytest


@pytest.mark.skip(reason="src/data/splits.py not implemented yet (session 2)")
def test_no_complex_crosses_splits():
    raise NotImplementedError
