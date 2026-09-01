"""The siamese ESM-2 regressor.

    wt_seq  --\
               >-- [shared ESM-2] -- mean-pool -- concat[wt, mut, mut-wt] -- head -- ddG
    mut_seq --/

WHY ONE BACKBONE CALLED TWICE, NOT TWO ENCODERS
-----------------------------------------------
This is what makes it "siamese". `self.esm` is a single set of weights, invoked
once on the wild-type sequence and once on the mutant. Both embeddings therefore
land in ONE representation space, and the difference between them means
something. Two separately-initialised encoders would drift into two spaces that
merely happen to be subtracted, and `mut_emb - wt_emb` would be noise.

It also halves the parameter count relative to two encoders, and — more
importantly — every gradient from both passes updates the same weights, so the
model sees twice the signal per step.

WHY concat[wt, mut, mut - wt]
-----------------------------
The difference vector does most of the work: ddG is fundamentally about what
CHANGED. Handing the head that difference explicitly saves it from having to
discover subtraction inside a linear layer. `wt` and `mut` are kept alongside it
because the same substitution means different things in different contexts — an
alanine at a hydrophobic core position is not the alanine at a solvent-exposed
rim position.

POOLING: MEAN FIRST, DELIBERATELY
---------------------------------
Mask-aware mean pooling over the whole sequence is the simple option, not the
best one. Position-specific pooling at the mutated residue is the planned v2
— ddG is a local perturbation, so averaging over 500 residues
dilutes the signal from the one that changed. `pooling="position"` is therefore
recognised and deliberately NOT implemented: building the simple version first
and iterating is part of the story, and `dataset.py` already emits
`mutation_token_index` so the upgrade is a small change, not a redesign.

Note the mean pools over <cls> and <eos> as well as the residues, because both
carry attention_mask == 1. The frozen-embedding baseline pools identically, and
src/baselines/linear_probe.py matches it too, so the frozen and fine-tuned
numbers stay comparable. Change it in one place and you must change it in both.

MEASURED ON THIS 4090 (session 4), one training step, bf16, batch 8 x 512:

    150M                        0.32 s/step   10.4 GB peak    ~2.2 min/epoch
    650M + checkpointing       10.7  s/step   26.9 GB peak    ~75  min/epoch

650M does not fit, even with checkpointing. It does not raise OutOfMemoryError
either — on Windows, CUDA spills into shared system memory, so the config runs
about 33x slower instead of failing loudly. That silence is the trap. Section
7.4's LoRA is the intended route to 650M: it removes the optimizer state
(5.2 GB) and the backbone gradients (2.6 GB), which is what makes it fit.

Usage (self-test: shapes, gradient flow, and a real VRAM/throughput measurement.
ONE config per run — see the note in main() about why sweeping in-process lies):
    python -m src.models.esm_regressor
    python -m src.models.esm_regressor --backbone facebook/esm2_t33_650M_UR50D         --checkpointing --batch-size 8 --seq-len 512
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_BACKBONE = "facebook/esm2_t30_150M_UR50D"


class DdgRegressor(nn.Module):
    """Siamese ESM-2 + MLP head. Section 7.3, implemented as specified.

    Args:
        backbone_name: any ESM-2 checkpoint. The head sizes itself from
            `config.hidden_size`, so 35M / 150M / 650M all work unchanged.
        freeze_backbone: True reproduces the frozen-embedding baseline
            (though linear_probe.py is far cheaper, since it caches).
        pooling: "mean" only. "position" is the planned v2 and raises.
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        freeze_backbone: bool = False,
        pooling: str = "mean",
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if pooling == "position":
            raise NotImplementedError(
                "position-specific pooling is the deliberate v2. "
                "dataset.py already provides mutation_token_index for it."
            )
        if pooling != "mean":
            raise ValueError(f"pooling must be 'mean', got {pooling!r}")
        self.pooling = pooling

        # AutoModel, not AutoModelForMaskedLM — we want the encoder's hidden
        # states, not vocabulary logits. transformers v5 prints a LOAD REPORT
        # calling lm_head.* UNEXPECTED and pooler.* MISSING. Both are correct
        # and harmless: we discard the LM head and never call the pooler.
        self.esm = AutoModel.from_pretrained(backbone_name)
        hidden = self.esm.config.hidden_size

        if freeze_backbone:
            # requires_grad=False stops autograd recording operations on these
            # tensors, so no gradients are computed OR stored for them.
            for parameter in self.esm.parameters():
                parameter.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(hidden * 3, head_hidden_dim),   # [wt, mut, mut - wt]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )

    def gradient_checkpointing_enable(self) -> None:
        """Trade compute for memory inside the backbone.

        Normally every intermediate activation from the forward pass is kept so
        the backward pass can reuse it. Checkpointing instead keeps only the
        inputs to each transformer layer and RECOMPUTES the interior during
        backward. Costs roughly 30% more time and saves most of the activation
        memory.

        This matters here more than usual: the siamese design runs the backbone
        TWICE per step, so activations — unlike weights and optimizer state —
        are doubled. See the measurements printed by this module's self-test.
        """
        self.esm.gradient_checkpointing_enable()

    def embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mask-aware mean pooling. (batch, seq) -> (batch, hidden).

        `attention_mask` is 1 on real tokens and 0 on padding. Unsqueezing it to
        (batch, seq, 1) lets it broadcast across the hidden dimension, so
        multiplying zeroes every padded position BEFORE the sum; dividing by
        `mask.sum(1)` then averages over real tokens only.

        Skip the mask and short sequences get their embeddings dragged toward
        whatever the model emits for <pad>, silently and without any error.
        """
        output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(1) / mask.sum(1)

    def forward(
        self,
        wt_ids: torch.Tensor,
        wt_mask: torch.Tensor,
        mut_ids: torch.Tensor,
        mut_mask: torch.Tensor,
    ) -> torch.Tensor:
        """-> (batch,) predicted ddG in kcal/mol.

        The two `self.embed` calls are the siamese part: same module, same
        weights, two inputs.
        """
        wt_emb = self.embed(wt_ids, wt_mask)
        mut_emb = self.embed(mut_ids, mut_mask)
        combined = torch.cat([wt_emb, mut_emb, mut_emb - wt_emb], dim=-1)
        # squeeze(-1) turns (batch, 1) into (batch,) to match the target shape.
        # Leaving it as (batch, 1) would broadcast against a (batch,) target and
        # silently compute a (batch, batch) loss matrix.
        return self.head(combined).squeeze(-1)


# --------------------------------------------------------------------------
# Self-test: shapes, gradient flow, and real VRAM
# --------------------------------------------------------------------------

def _measure_step(
    model: DdgRegressor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    steps: int = 4,
) -> tuple[float, float, bool]:
    """Run real training steps and report (seconds_per_step, peak_gb, ok).

    Two warm-up steps are discarded before timing: the first step pays for
    cuDNN autotuning and for AdamW allocating its exp_avg / exp_avg_sq state
    (2 x params x 4 bytes — 5.2 GB on the 650M), neither of which recurs.

    `ok` is False on out-of-memory, which is a legitimate measurement rather
    than a crash. Note it is NOT a reliable fit test on Windows: see the spill
    warning in main().
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    vocab = model.esm.config.vocab_size
    model.train()

    try:
        ids = torch.randint(4, vocab - 1, (batch_size, seq_len), device=device)
        mask = torch.ones_like(ids)
        target = torch.randn(batch_size, device=device)

        start = 0.0
        for i in range(steps + 2):
            if i == 2:
                torch.cuda.synchronize()
                start = time.time()
            # bf16 autocast: matmuls run in bfloat16, which has fp32's exponent
            # range, so gradients cannot underflow to zero and NO GradScaler is
            # needed. That is the only reason fp16 requires one.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(ids, mask, ids, mask)
                loss = nn.HuberLoss(delta=1.0)(prediction.float(), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        seconds = (time.time() - start) / steps
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        ok = True
    except torch.OutOfMemoryError:
        seconds, peak, ok = float("nan"), float("nan"), False

    del optimizer
    torch.cuda.empty_cache()
    return seconds, peak, ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Building DdgRegressor({args.backbone}) ...")
    model = DdgRegressor(args.backbone).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head = sum(p.numel() for p in model.head.parameters())
    print(f"  hidden size    : {model.esm.config.hidden_size}")
    print(f"  layers         : {model.esm.config.num_hidden_layers}")
    print(f"  parameters     : {total / 1e6:.1f}M total, {trainable / 1e6:.1f}M trainable")
    print(f"  head           : {head / 1e6:.2f}M ({100 * head / total:.2f}% of the model)")

    # --- shapes -----------------------------------------------------------
    batch, length = 4, 128
    ids = torch.randint(4, 30, (batch, length), device=device)
    mask = torch.ones_like(ids)
    mask[0, -20:] = 0  # a genuinely padded row, so pooling is actually exercised

    model.eval()
    with torch.no_grad():
        out = model(ids, mask, ids, mask)
    print(f"\n  forward: (4, 128) x4 -> {tuple(out.shape)}  {out.dtype}   OK")
    assert out.shape == (batch,), f"expected ({batch},), got {tuple(out.shape)}"

    # Identical WT and mutant must give a zero difference vector. Not a tautology
    # worth skipping: it proves embed() is deterministic and that the concat
    # order is what we think it is.
    with torch.no_grad():
        emb = model.embed(ids, mask)
    assert torch.equal(emb, model.embed(ids, mask)), "embed() is not deterministic in eval()"
    print("  embed() deterministic in eval(), identical inputs -> zero difference   OK")

    # --- gradient flow ----------------------------------------------------
    model.train()
    loss = nn.HuberLoss(delta=1.0)(model(ids, mask, ids, mask),
                                   torch.randn(batch, device=device))
    loss.backward()

    first_layer = next(model.esm.parameters())
    head_weight = model.head[0].weight
    print(f"\n  gradients reach the backbone : {first_layer.grad is not None} "
          f"(|g| = {first_layer.grad.abs().mean():.2e})")
    print(f"  gradients reach the head     : {head_weight.grad is not None} "
          f"(|g| = {head_weight.grad.abs().mean():.2e})")
    assert first_layer.grad is not None and head_weight.grad is not None
    model.zero_grad(set_to_none=True)

    # And the frozen case must be exactly the opposite.
    frozen = DdgRegressor(args.backbone, freeze_backbone=True).to(device)
    frozen.train()
    nn.HuberLoss()(frozen(ids, mask, ids, mask), torch.randn(batch, device=device)).backward()
    backbone_grads = [p.grad for p in frozen.esm.parameters() if p.grad is not None]
    print(f"  freeze_backbone=True -> backbone gradients: {len(backbone_grads)} "
          f"(must be 0), head still trains: {frozen.head[0].weight.grad is not None}")
    assert not backbone_grads, "freeze_backbone=True still produced backbone gradients"
    del frozen
    torch.cuda.empty_cache()

    if args.skip_memory or device.type != "cuda":
        print("\nAll checks passed.")
        return

    # --- real VRAM and throughput ----------------------------------------
    # ONE config per invocation, deliberately. Measuring several in a single
    # process gives numbers that are wrong in a way that looks plausible: the
    # caching allocator does not hand memory back between configs, so a later
    # config inherits an inflated baseline and can be pushed into host-memory
    # spill by its predecessors. An earlier version of this probe swept five
    # configs in one process and reported MORE memory for 8x512 than for 8x1024
    # — incoherent, and the tell that the sweep was unusable.
    #
    # Compare configs by running this module several times, not by looping.
    if args.batch_size * args.seq_len > 0:
        print(f"\n  peak VRAM and throughput, {args.batch_size} x {args.seq_len}, "
              f"bf16 autocast + AdamW, checkpointing={args.checkpointing}")
        if args.checkpointing:
            model.gradient_checkpointing_enable()
        seconds, peak, ok = _measure_step(
            model, args.batch_size, args.seq_len, device, steps=args.steps
        )
        free, total_vram = torch.cuda.mem_get_info()
        if not ok:
            print("    OUT OF MEMORY")
        else:
            print(f"    {seconds:.3f} s/step   peak {peak:.1f} GB   "
                  f"{args.batch_size / seconds:.1f} examples/s   "
                  f"~{3370 / (args.batch_size / seconds) / 60:.1f} min/epoch "
                  f"(3370 training rows)")
            # WINDOWS TRAP: exceeding VRAM does not raise OutOfMemoryError here.
            # CUDA silently spills into shared system memory, so an over-budget
            # config runs 30x slower instead of failing loudly. Measured on the
            # 650M: 10.7 s/step at 26.9 GB peak, against 0.32 s/step for the
            # 150M at 10.4 GB.
            if peak > total_vram / 1024**3 - 1.0:
                print(f"    WARNING: peak {peak:.1f} GB is at or over the card's "
                      f"{total_vram / 1024**3:.1f} GB. This did not raise — on "
                      f"Windows\n             CUDA spills to host memory instead. "
                      f"The step time above is the real cost.")

        print(f"\n  GPU: {torch.cuda.get_device_name(0)}  "
              f"{total_vram / 1024**3:.1f} GB total, {free / 1024**3:.1f} GB free")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
