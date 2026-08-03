"""Siamese ESM-2 regressor for ddG of binding.

Implements PLAN.md section 7.3. The architecture below is a settled decision —
implement it as specified rather than substituting alternatives.

ARCHITECTURE (decided, not open):

    wt_seq  --.
              |--> ONE shared ESM-2 backbone --> mean-pool --> wt_emb
    mut_seq --'                                            --> mut_emb

    head( concat[wt_emb, mut_emb, mut_emb - wt_emb] ) -> scalar ddG

Two points to understand rather than just run:

  * ONE `self.esm` called twice, not two separate models. That is what makes
    it siamese: identical weights process both sequences, so the model learns
    a single consistent representation space instead of two that merely get
    compared. Two separate backbones would double the parameters and lose the
    guarantee that the two embeddings are commensurable.

  * The concatenation is [wt, mut, mut - wt]. The explicit difference vector
    is doing much of the work, because ddG is fundamentally about what
    CHANGED. The two absolute embeddings supply the context that tells the
    head how much a given change matters.

  * Mean-pooling is the SIMPLE choice, deliberately taken first — not the
    best one. The planned v2 is position-specific pooling: take the embedding
    at the mutated residue from both passes instead of averaging the whole
    sequence, since ddG is a local perturbation. Build the mean-pool version,
    confirm it works end to end, then iterate. Do not skip ahead.

TODO(session 3): implement.
  - DdgRegressor(backbone_name="facebook/esm2_t30_150M_UR50D", freeze_backbone=False)
  - head: Linear(hidden * 3, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 1)
  - embed(): mask-aware mean-pool, (h * mask).sum(1) / mask.sum(1)
  - forward(wt_ids, wt_mask, mut_ids, mut_mask) -> (batch,) predictions
  - freeze_backbone=True powers the section 7.2 frozen-embedding baseline
  - Verify gradients actually reach the backbone when unfrozen, and are
    None/zero when frozen. The plan verified this; re-verify in our code.

LoRA (section 7.4) is OPTIONAL on this hardware. The 24 GB 4090 fits full
fine-tuning of 150M comfortably and very likely 650M too, so LoRA is a
speed/throughput lever here, not a memory requirement.
"""
