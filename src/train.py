"""Training loop for the siamese ESM-2 ddG regressor.

Implements PLAN.md section 8, adapted for local RTX 4090 training.

TODO(session 3/4): implement.

LOSS: Huber (delta=1.0), a deliberate choice, not a default. ddG data has real
outliers — large destabilizing mutations, and any abolished-binding entries
that survive filtering. Huber is far less dominated by them than squared
error. This is settled; do not swap in MSE.

WORKFLOW (section 8, minus the cloud parts):
  1. Debug on a tiny subset (20-50 rows) first. The goal is purely mechanical:
     shapes line up, loss computes, one step runs. Cheap and fast.
  2. Then the full run on the 4090.
  3. Checkpoint every N steps, not just at the end.

MIXED PRECISION — READ BEFORE COPYING SECTION 8's SNIPPET:
  The plan's snippet uses `torch.cuda.amp.GradScaler()` with float16. Two
  things have moved on, and the plan itself anticipates this ("ask Claude Code
  to check whether the exact mixed-precision API has moved on"):
    - `torch.cuda.amp.GradScaler()` is deprecated; the current spelling is
      `torch.amp.GradScaler('cuda')`.
    - This GPU is Ada (sm_89) and supports bfloat16. bf16 has the same dynamic
      range as fp32, so it needs NO GradScaler at all — the scaler exists to
      stop fp16's narrow range from underflowing gradients to zero.
  Recommend bf16 autocast with no scaler. Confirm with Johannes before
  changing what the plan specifies.

VRAM: 24 GB total, ~2 GB held by desktop apps, so budget ~22 GB. Full
fine-tuning of 150M is comfortable; 650M should fit. LoRA (section 7.4) is a
speed lever here, not a memory requirement.

TRACKING: wandb is installed but NOT logged in. `wandb login` is Johannes's to
run. Section 8 rates the public experiment report as a strong portfolio signal.
"""
