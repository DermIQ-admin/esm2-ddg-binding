"""Fine-tune the siamese ESM-2 regressor. PLAN.md section 8.

    python -m src.train                          # split_pdb_id, the honest primary
    python -m src.train --split-column split_mutation
    python -m src.train --backbone facebook/esm2_t33_650M_UR50D --gradient-checkpointing

ONE MODEL PER SPLIT DEFINITION — NOT ONE MODEL SCORED THREE WAYS
----------------------------------------------------------------
The three split definitions disagree about which rows are held out: a row in
`split_pdb_id`'s test set is usually in `split_mutation`'s TRAINING set. So the
leakage gradient has to be assembled from three separately trained models. This
script trains ONE, and passes `trained_on=` to the evaluation harness, which
marks the other two tables INVALID rather than quietly reporting them. Run it
three times to build the full comparison.

WHAT DIFFERS FROM SECTION 8'S SNIPPET, AND WHY
----------------------------------------------
Section 8 invites this check explicitly, and two things have moved on:

1. NO GradScaler. The snippet uses fp16 with `torch.cuda.amp.GradScaler()`.
   A scaler exists solely to stop fp16 gradients underflowing to zero — fp16
   has only 5 exponent bits, so small gradients flush to zero and training
   silently stalls. This GPU (Ada, sm_89) supports bf16, which has fp32's
   8-bit exponent and therefore cannot underflow that way. bf16 trades mantissa
   precision instead, which gradient descent tolerates well. So: bf16 autocast,
   no scaler, and `torch.cuda.amp` -> `torch.amp` while we are here.

2. Sections 8's steps 1-4 are about Colab/Kaggle. We train locally on a 4090;
   see CLAUDE.md. Checkpoints still get written, but for resumability rather
   than to survive a session disconnect.

MODEL SELECTION IS ON VALIDATION SPEARMAN, NOT VALIDATION LOSS
--------------------------------------------------------------
Spearman is the primary metric (section 9), and the two genuinely diverge:
Huber loss keeps improving by tightening absolute calibration long after the
rank ordering has stopped changing. Selecting on loss would hand back a
checkpoint that is better calibrated and worse at the thing we actually report.

THE BAR TO CLEAR (session 3 baselines, split_pdb_id test):
    zero-shot masked-marginal      0.005
    frozen embeddings + ridge      0.092
    frozen embeddings + MLP        0.198   <- fine-tuning must beat this
If this lands near 0.60 on an honest split, suspect leakage before celebrating:
0.603 is what the deliberately-leaky split_mutation produced.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats

from src.data.dataset import MAX_TOKENS, SPLIT_COLUMNS, build_dataloaders
from src.models.esm_regressor import DEFAULT_BACKBONE, DdgRegressor
from src.utils import describe_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"
RESULTS_DIR = REPO_ROOT / "results"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"  # gitignored


def move(batch: dict, device: torch.device) -> dict:
    """Tensors to the GPU; everything else (uid strings) left alone."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader, device: torch.device,
                    loss_fn: nn.Module) -> tuple[float, float, pd.DataFrame]:
    """-> (mean loss, Spearman, per-row predictions).

    `model.eval()` switches Dropout off and is not optional: leaving it on would
    make validation noisy and the epoch selection partly random.
    """
    model.eval()
    losses, uids, predictions, targets = [], [], [], []

    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch["wt_ids"], batch["wt_mask"],
                           batch["mut_ids"], batch["mut_mask"])
        # .float() before the loss: bf16 has ~3 decimal digits of mantissa, too
        # few to accumulate a stable mean over hundreds of batches.
        output = output.float()
        losses.append(loss_fn(output, batch["ddg"]).item() * len(batch["ddg"]))
        uids.extend(batch["uid"])
        predictions.append(output.cpu().numpy())
        targets.append(batch["ddg"].cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    rho = stats.spearmanr(targets, predictions).statistic if len(targets) > 2 else float("nan")

    return (sum(losses) / len(targets), float(rho),
            pd.DataFrame({"uid": uids, "y_pred": predictions}))


def train(
    model: nn.Module,
    loaders: dict,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    huber_delta: float,
    accumulation_steps: int = 1,
    patience: int = 3,
    verbose: bool = True,
) -> tuple[nn.Module, dict]:
    """Fine-tune, keeping the epoch with the best validation Spearman."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.HuberLoss(delta=huber_delta)  # section 8, deliberate over MSE

    best = {"rho": -np.inf, "epoch": -1, "state": None}
    history = []

    for epoch in range(epochs):
        model.train()
        # Reshuffles the length buckets so batch composition differs per epoch
        # while staying length-homogeneous. See TokenBudgetBatchSampler.
        # getattr twice because --limit-batches replaces the loader with a plain
        # list of batches, which has neither attribute.
        sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        running, seen, started = 0.0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loaders["train"]):
            batch = move(batch, device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(batch["wt_ids"], batch["wt_mask"],
                               batch["mut_ids"], batch["mut_mask"])
                loss = loss_fn(output.float(), batch["ddg"])

            # Gradient accumulation: gradients ADD UP across backward() calls,
            # so dividing by the accumulation count keeps the effective step
            # size the same as a single batch that size would have given.
            (loss / accumulation_steps).backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                # set_to_none=True frees the gradient tensors rather than
                # zeroing them in place — slightly faster and less memory.
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * len(batch["ddg"])
            seen += len(batch["ddg"])

        # A trailing partial accumulation group would otherwise be discarded.
        if len(loaders["train"]) % accumulation_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running / seen
        val_loss, val_rho, _ = evaluate_loader(model, loaders["val"], device, loss_fn)
        elapsed = time.time() - started
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_spearman": val_rho,
                        "seconds": round(elapsed, 1)})

        marker = ""
        if val_rho > best["rho"]:
            best = {"rho": val_rho, "epoch": epoch,
                    # .cpu() so the snapshot does not pin a second copy of the
                    # model in VRAM for the rest of training.
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            marker = "  <- best"

        if verbose:
            print(f"    epoch {epoch:>2}  train {train_loss:.4f}  val {val_loss:.4f}"
                  f"  val rho {val_rho:+.4f}  {elapsed:>5.0f}s{marker}")

        if epoch - best["epoch"] >= patience:
            if verbose:
                print(f"    early stop: {patience} epochs without improvement")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "best_val_spearman": best["rho"],
                   "history": history}


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=config["model"]["backbone"])
    parser.add_argument("--split-column", default="split_pdb_id", choices=SPLIT_COLUMNS)
    parser.add_argument("--epochs", type=int, default=config["training"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=config["training"]["batch_size"])
    parser.add_argument("--lr", type=float, default=config["training"]["learning_rate"])
    # 4096, matching the config that was actually benchmarked (8 x 512 -> 10.9 GB
    # peak on the 150M). An earlier default of 8192 was set by eye rather than
    # from that measurement: it roughly doubles activation memory, and combined
    # with allocator fragmentation across variable batch shapes it drove the
    # reserved pool to 24.0 of the card's 24.0 GB and spilled into host memory —
    # 26.8 GB working set, 99% GPU utilisation, and no forward progress worth
    # the wall time. The median complex is 394 tokens, so at 4096 most batches
    # still reach the full batch_size of 8 and little throughput is lost.
    parser.add_argument("--max-tokens-per-batch", type=int, default=4096,
                        help="memory ceiling per step; see TokenBudgetBatchSampler")
    parser.add_argument("--accumulation-steps", type=int,
                        default=config["training"]["gradient_accumulation_steps"])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="needed for 650M; roughly free on 150M")
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="section 8 step 1: a tiny mechanical-correctness run")
    parser.add_argument("--seed", type=int, default=config["seed"],
                        help="training stochasticity: head init, dropout, batch order")
    parser.add_argument("--splits-file", default="splits.csv",
                        help="alternate partition from splits.py --random-state; "
                             "varies WHICH COMPLEXES are held out, which --seed cannot")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    info = describe_device()
    device = torch.device("cuda" if info["cuda_available"] else "cpu")
    print(f"Device: {device} ({info.get('device_name', 'cpu')})")
    print(f"Backbone: {args.backbone}   split: {args.split_column}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    print("\nData:")
    loaders = build_dataloaders(
        tokenizer,
        split_column=args.split_column,
        batch_size=args.batch_size,
        max_tokens=MAX_TOKENS,
        max_tokens_per_batch=args.max_tokens_per_batch,
        splits_file=args.splits_file,
    )
    for name, loader in loaders.items():
        print(f"    {name:<6} {len(loader.dataset):>5} rows, {len(loader):>4} batches")

    model = DdgRegressor(
        args.backbone,
        freeze_backbone=config["model"]["freeze_backbone"],
        pooling=config["model"]["pooling"],
        head_hidden_dim=config["model"]["head_hidden_dim"],
        dropout=config["model"]["dropout"],
    ).to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {trainable / 1e6:.1f}M trainable parameters, "
          f"loss=Huber(delta={config['training']['huber_delta']}), "
          f"lr={args.lr}, bf16 autocast, no GradScaler")

    if args.limit_batches:
        # Section 8 step 1: mechanical correctness on a handful of batches.
        import itertools
        for split in loaders:
            loaders[split] = list(itertools.islice(loaders[split], args.limit_batches))
        print(f"  --limit-batches {args.limit_batches}: a smoke run, not a result")

    print("\nTraining:")
    model, summary = train(
        model, loaders, device,
        epochs=args.epochs,
        learning_rate=args.lr,
        huber_delta=config["training"]["huber_delta"],
        accumulation_steps=args.accumulation_steps,
        patience=args.patience,
    )
    print(f"\n  best epoch {summary['best_epoch']} "
          f"(val Spearman {summary['best_val_spearman']:+.4f})")

    if args.limit_batches:
        print("\nSmoke run complete — no metrics written.")
        return

    # Predict every row, so the harness can slice by split itself.
    loss_fn = nn.HuberLoss(delta=config["training"]["huber_delta"])
    predictions = pd.concat(
        [evaluate_loader(model, loaders[s], device, loss_fn)[2] for s in loaders]
    ).drop_duplicates("uid")

    tag = args.backbone.split("/")[-1]
    # Both seeds go in the name: a replicate is only interpretable if you can
    # tell which partition and which training run produced it.
    split_seed = (args.splits_file.replace("splits_seed", "").replace(".csv", "")
                  if args.splits_file != "splits.csv" else "42")
    name = f"finetune_{tag}_{args.split_column}_split{split_seed}_seed{args.seed}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.results_dir / f"preds_{name}.csv", index=False)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args),
                "summary": summary}, CHECKPOINT_DIR / f"{name}.pt")

    from src.evaluate import evaluate_predictions, format_report, save_report
    report = evaluate_predictions(predictions, name=name, trained_on=args.split_column,
                                  splits_file=args.splits_file)
    report["training"] = summary
    print(format_report(report))
    print(f"\n  metrics    -> {save_report(report, args.results_dir)}")
    print(f"  checkpoint -> {CHECKPOINT_DIR / f'{name}.pt'}")


if __name__ == "__main__":
    main()
