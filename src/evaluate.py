"""Evaluation harness: metrics, both splits, error analysis, plots.

Implements PLAN.md section 9.

Every model — zero-shot, frozen-embedding, fine-tuned — reports through this
one harness, so the numbers in the README are directly comparable.

TODO(session 2, extended later): implement.

METRICS
  - PRIMARY: Spearman correlation. Standard in this literature because rank
    ordering matters more than absolute calibration for the real downstream
    use (ranking candidate mutations).
  - Report Pearson and RMSE/MAE alongside it. Not instead of it.

REPORT ON BOTH SPLITS
  Complex-level (the honest number) AND mutation-level (the leaky contrast).
  Per section 9 this side-by-side is one of the most interview-worthy things
  in the repo. Expect mutation-level to look BETTER — that gap is the point,
  not an embarrassment. Never report the mutation-level number alone.

ERROR ANALYSIS
  - Which complexes are worst? Any structure to the failures?
  - Patterns by mutation type — mutations to/from proline or glycine are
    structurally unusual and a plausible failure mode
  - If any multi-point entries were kept, do they underperform single-point?

PLOTS -> results/figures/
  - Predicted vs. true ddG scatter, train and test side by side
  - Optionally colored by complex, to make the clustering visible

Writes results/metrics.json (generated, not committed by hand).
"""
