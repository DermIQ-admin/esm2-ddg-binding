"""Frozen-embedding baseline: cached ESM-2 embeddings + a small trained head.

Implements PLAN.md section 7.2.

Build this BEFORE full fine-tuning. It is fast to iterate on, it validates the
entire data pipeline and evaluation harness before any GPU hours get spent on
the real thing, and it yields a second reportable baseline. Cheap experiment
before expensive experiment.

TODO(session 3): implement.
  - Extract ESM-2 embeddings ONCE for every WT and mutant sequence
  - Cache them to disk (see .gitignore: embeddings_cache/) — recomputing
    embeddings every epoch is the mistake this baseline exists to avoid
  - Train only a linear or small MLP head on [wt, mut, mut - wt], matching
    the section 7.3 feature construction so the comparison is apples-to-apples
  - Evaluate through the SAME harness as everything else (src/evaluate.py)

Equivalent to DdgRegressor(freeze_backbone=True), but with embeddings
precomputed rather than recomputed each pass — same model, far less compute.
"""
