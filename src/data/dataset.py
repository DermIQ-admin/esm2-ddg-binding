"""PyTorch Dataset and collate_fn for WT/mutant sequence pairs.

Implements the data-loading half of PLAN.md section 7.3.

TODO(session 2/3): implement.

The siamese model consumes FOUR tensors per example — the WT and mutant
sequences are tokenized independently and each needs its own attention mask:

    wt_ids, wt_mask, mut_ids, mut_mask   ->  ddg (float target)

  - Dataset yields one (wt_seq, mut_seq, ddg) record per row
  - collate_fn pads WT and mutant batches independently, because the two
    sequences are separate forward passes through the shared backbone
  - The attention mask is what makes mean-pooling correct: section 7.3 pools
    with `(hidden * mask).sum(1) / mask.sum(1)` so padding tokens contribute
    nothing to the sequence embedding. Getting the mask wrong silently
    corrupts every embedding rather than raising.
  - Section 6.4: plot the concatenated-complex length distribution during EDA
    and decide truncation empirically from what we see, not in advance
"""
