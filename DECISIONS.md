# Decisions log

Why this project is built the way it is. Each entry states the decision, the evidence
behind it, and what it costs — including the ones that turned out to be wrong.

---

## Why split by complex rather than by mutation?

**Decision.** Partition *groups*, never rows, and report three definitions side by side:
`split_mutation` (naive, row-level), `split_pdb_id` (313 structures) and
`split_hold_out_proteins` (152 curated protein pairs). `tests/test_splits.py` enforces the
invariant as 14 tests.

**Evidence.** The same model and hyperparameters score Spearman 0.660 ± 0.021 on the naive
split and 0.178 ± 0.059 on `split_pdb_id` — a 3.7× gap produced by nothing but the split
definition. On the naive split, validation contains other mutations of complexes already in
training, so the best epoch is the last one in all five replicates; on the strictest split
three of five replicates are best at epoch 0.

**Cost.** The honest headline is ~0.17–0.27 rather than 0.66.

---

## Why Huber loss rather than MSE?

**Decision.** `nn.HuberLoss(delta=1.0)`, used by both the fine-tuned model and the frozen
probe so the two stay comparable.

**Evidence.** The ΔΔG distribution over 4,829 rows is mean +0.96, median +0.56, sd 1.71,
range ±12.2 kcal/mol, with **11.0% of rows beyond |ΔΔG| > 3**. Those are real measurements,
not noise, so a squared penalty would let a handful of them dominate the gradient.

**Cost.** Huber is less sensitive to exactly the large-effect mutations that matter most in
practice; `error_analysis` reports a `large_effect` breakdown for this reason.

---

## Why mean-pooling first, and position-specific pooling second?

**Decision.** Mask-aware mean-pooling over the whole complex for v1.
`dataset.py` already emits `mutation_token_index`, so the v2 is a head change, not a
pipeline change.

**Evidence.** A whole-sequence mean is a weak signal for a change of one residue in up to
2,048 — this is named in the README as plausibly the largest architectural limitation here,
and it is first on the next-steps list.

**Cost.** Almost certainly leaves signal on the table.

---

## Why a siamese (shared-weight) backbone rather than two encoders?

**Decision.** One backbone called twice, on the wild-type and mutant complex. The head sees
`concat[wt, mut, mut − wt]`.

**Evidence.** Shared weights put both embeddings in one space, which is what makes their
difference meaningful. The measured cost is real: calling the backbone twice per step is
why 650M needs 26.9 GB on a 24 GB card and runs 33× slower.

---

## Why 150M as the primary result?

**Decision.** `facebook/esm2_t30_150M_UR50D`, full fine-tune, 148.6M trainable parameters.

**Evidence.** Not a compute constraint in the way first assumed — 150M peaks at 10.9 GB and
fits comfortably. 650M does *not* fit: measured at 26.9 GB with gradient checkpointing on a
24 GB card, 10.7 s/step against 0.19, spilling silently into host memory rather than raising
`OutOfMemoryError`. LoRA is therefore a requirement for 650M, not a convenience.

**Cost.** No capacity comparison, so "would a bigger backbone close the honest gap?" is open.

---

## Why sequence-only, and why single-point mutations only for v1?

**Decision.** No structural features despite SKEMPI shipping structures; 1,973 multi-point
rows dropped, leaving 4,829 single-point rows across 316 complexes (4,814 / 314 after the
2,048-token length policy).

**Cost.** No epistasis, and no comparison against FoldX-style physics-based methods on their
own terms.

---

## Why exclude the "abolishes binding" entries?

**Decision.** 127 rows dropped because their affinity is a bound, not a measurement. The
filter runs against the **raw** `Affinity_* (M)` columns, not the `Affinity_*_parsed`
convenience columns.

**Evidence.** The parsed columns silently strip the leading `<` / `>` and return a bare
number, so filtering on missing values alone keeps "no detectable binding" entries as if
they were real K<sub>d</sub> values. A further 156 rows with unusable K<sub>d</sub> at
either end are also dropped.

**Cost.** Censored regression would use these rows properly; that is out of scope for v1.

---

## Decisions revisited

Things this project changed its mind about, and what changed them.

- **"Fine-tuning loses to the frozen probe on the honest splits."** Claimed from single runs
  (0.148 vs 0.198). Five replicates per method per split showed the difference is not
  distinguishable from noise on either honest split (p = 0.229 and p = 0.124). The
  supportable claim is narrower and sharper: fine-tuning is reliably better **only** on the
  leaky split (+0.038, p = 0.038).
- **"Fixing the seed makes a run reproducible."** The identical configuration produced 0.148
  and 0.225 on two different days from CUDA kernel nondeterminism alone. Every number in the
  README is now a 5-replicate mean.
- **"The training curves peak early on the honest splits."** Also a single-run claim. Across
  five replicates the best epoch on `split_pdb_id` is 3, 3, 7, 9, 9 — erratic rather than an
  early peak — and only one run ends with a negative validation correlation.
- **"The LoRA runs were truncated, so 0.211 is a lower bound."** Claimed on noticing that
  three of five runs hit the 10-epoch cap with validation still climbing. A 30-epoch probe on
  the identical partition disproved it: validation *did* keep improving (0.393 at epoch 9 to
  0.404 at epoch 25), but test Spearman **fell** from 0.137 to 0.054. More epochs make
  validation look better and test worse. The 10-epoch numbers are a fair comparison, not a
  floor — and the claim lasted about two hours, which is roughly how long it took to test it.
- **"650M should also fit."** Estimated before the siamese architecture existed, which calls
  the backbone twice per step and doubles activations. Measurement overturned it.
- **"LoRA is a speed lever, not a memory requirement."** True for 150M, false for 650M,
  which is the case that matters.
- **A fixed `batch_size` is fine.** Complexes span 60–2,048 tokens and batches pad to their
  longest member, so peak memory depended on which complexes a batch happened to draw.
  Replaced with a token budget (4,096 padded tokens), making the ceiling structural.
