"""Frozen-embedding baseline: cached ESM-2 embeddings + a small trained head.


Built BEFORE full fine-tuning. It is fast to iterate on, it validates the entire
data pipeline and evaluation harness before any GPU hours get spent on the real
thing, and it yields a second reportable baseline. Cheap experiment before
expensive experiment.

This is equivalent to `DdgRegressor(freeze_backbone=True)`, but
with the embeddings precomputed instead of recomputed every epoch — the same
model, a tiny fraction of the compute. Feature construction is deliberately
IDENTICAL to the fine-tuned model's, `concat[wt, mut, mut - wt]`, so the comparison
against the fine-tuned model is apples-to-apples and the only thing that changes
between them is whether the backbone learns.

WHY CACHING IS THE POINT
------------------------
A frozen backbone produces the same embedding for the same sequence every time.
Recomputing it each epoch is the mistake this baseline exists to avoid: it would
cost a full ESM-2 forward pass per example per epoch to obtain a number that
never changes. We pay it once.

Only 314 wild-type sequences exist (one per complex) against 4814 mutants, so
the WT embeddings are computed per COMPLEX and reused across every mutation of
that complex — 5128 forward passes rather than 9628.

ONE MODEL PER SPLIT DEFINITION
------------------------------
The three split definitions disagree about which rows are held out, so a single
trained head cannot be scored against all three (see the `trained_on` guard in
src/evaluate.py). We therefore train three heads, one per definition, and
assemble the leakage gradient from them. The cached embeddings are shared across
all three — the backbone is frozen, so they do not depend on the split.

Usage:
    python -m src.baselines.linear_probe
    python -m src.baselines.linear_probe --backbone facebook/esm2_t12_35M_UR50D
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import MAX_TOKENS, apply_length_policy, load_frames
from src.data.parse_skempi import apply_mutation
from src.utils import describe_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = REPO_ROOT / "embeddings_cache"  # gitignored

DEFAULT_BACKBONE = "facebook/esm2_t30_150M_UR50D"
SPLIT_COLUMNS = ["split_mutation", "split_pdb_id", "split_hold_out_proteins"]

# Sequences vary 60..2048 tokens, so batch by a TOKEN budget rather than a fixed
# count — otherwise a batch of long sequences costs 30x one of short ones.
TOKEN_BUDGET = 16_384

# Head hyperparameters. configs/config.yaml: model.head_hidden_dim / model.dropout
# match the fine-tuned model, deliberately. The learning rate is much higher than
# training.learning_rate (2e-5) because that value is for a pretrained backbone
# being nudged; a randomly-initialised head can and should move faster.
HEAD_LR = 1e-3
HEAD_EPOCHS = 300
HEAD_BATCH = 256
HUBER_DELTA = 1.0  # deliberate over MSE, ddG has real outliers
PATIENCE = 40


# --------------------------------------------------------------------------
# Embedding extraction
# --------------------------------------------------------------------------

@torch.inference_mode()
def embed_sequences(
    sequences: list[str], model, tokenizer, device: torch.device, verbose: bool = True
) -> np.ndarray:
    """Mean-pooled embeddings, one row per sequence. Shape (len(sequences), hidden).

    MASK-AWARE MEAN POOLING, exactly as the fine-tuned model does:

        (hidden_states * mask).sum(1) / mask.sum(1)

    `mask` is the attention mask broadcast over the hidden dimension. Multiplying
    by it zeroes every padding position BEFORE the sum, and dividing by
    `mask.sum(1)` averages over real tokens only. Without this, a short sequence
    padded out to the batch maximum would have its embedding dragged toward
    whatever the model emits for <pad> — and nothing would raise.

    Note this pools over <cls> and <eos> along with the residues, because they
    carry attention_mask == 1. That is what the fine-tuned model does, and we
    match it deliberately so the frozen and fine-tuned numbers stay comparable.

    Sorting by length before batching keeps similar lengths together, which
    minimises padding. We restore the original order before returning.
    """
    order = np.argsort([len(s) for s in sequences])
    out = np.zeros((len(sequences), model.config.hidden_size), dtype=np.float32)

    start, done, began = 0, 0, time.time()
    while start < len(order):
        longest = len(sequences[order[start]]) + 2  # <cls> ... <eos>
        size = max(1, TOKEN_BUDGET // longest)
        idx = order[start:start + size]

        batch = tokenizer([sequences[i] for i in idx], padding=True, return_tensors="pt")
        ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)

        hidden = model(input_ids=ids, attention_mask=attention).last_hidden_state
        mask = attention.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1)

        out[idx] = pooled.float().cpu().numpy()
        start += len(idx)
        done += len(idx)

        if verbose and (done % 1000 < len(idx) or start >= len(order)):
            print(f"    {done:>5}/{len(sequences)}  {time.time() - began:>5.1f}s")

    return out


def build_or_load_cache(
    df: pd.DataFrame, backbone: str, device: torch.device, cache_dir: Path = CACHE_DIR
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return ({pdb_field: wt_embedding}, {uid: mutant_embedding}), cached to disk.

    The cache key includes the backbone name, so switching models cannot
    silently reuse the wrong embeddings.
    """
    tag = backbone.split("/")[-1]
    path = cache_dir / f"{tag}.npz"

    wt_keys = sorted(df["pdb_field"].unique())
    mut_keys = list(df["uid"])

    if path.exists():
        cached = np.load(path, allow_pickle=False)
        # Guard against a cache built from a different filtered row set.
        if (list(cached["wt_keys"]) == wt_keys
                and list(cached["mut_keys"]) == mut_keys):
            print(f"  cache hit: {path}")
            return (dict(zip(wt_keys, cached["wt_emb"])),
                    dict(zip(mut_keys, cached["mut_emb"])))
        print(f"  cache at {path} does not match the current row set — rebuilding")

    from transformers import AutoModel, AutoTokenizer

    # AutoModel, NOT AutoModelForMaskedLM — here we want hidden states, not
    # vocabulary logits. This is the same class the siamese model wraps.
    print(f"  loading {backbone} ...")
    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(
        backbone,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    by_field = df.drop_duplicates("pdb_field").set_index("pdb_field")
    print(f"  embedding {len(wt_keys)} wild-type complexes:")
    wt_emb = embed_sequences(
        [by_field.loc[k, "sequence"] for k in wt_keys], model, tokenizer, device
    )

    print(f"  embedding {len(mut_keys)} mutant sequences:")
    mutant_sequences = [
        apply_mutation(r.sequence, int(r.mutation_index), r.wt_aa, r.mut_aa)
        for r in df.itertuples(index=False)
    ]
    mut_emb = embed_sequences(mutant_sequences, model, tokenizer, device)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, wt_keys=np.array(wt_keys), wt_emb=wt_emb,
                        mut_keys=np.array(mut_keys), mut_emb=mut_emb)
    print(f"  cached -> {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return dict(zip(wt_keys, wt_emb)), dict(zip(mut_keys, mut_emb))


def build_features(df: pd.DataFrame, wt: dict, mut: dict) -> np.ndarray:
    """concat[wt, mut, mut - wt] — the siamese model's feature construction, verbatim.

    The difference vector is doing most of the work: ddG is about what CHANGED,
    and giving the head that difference explicitly saves it from having to
    discover subtraction on its own.
    """
    wt_rows = np.stack([wt[f] for f in df["pdb_field"]])
    mut_rows = np.stack([mut[u] for u in df["uid"]])
    return np.concatenate([wt_rows, mut_rows, mut_rows - wt_rows], axis=1)


# --------------------------------------------------------------------------
# The head
# --------------------------------------------------------------------------

def make_head(input_dim: int, hidden: int = 256, dropout: float = 0.1) -> nn.Module:
    """Section 7.3's head exactly: Linear -> ReLU -> Dropout -> Linear."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


def train_head(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    device: torch.device, verbose: bool = True,
) -> nn.Module:
    """Train the head, selecting the epoch by VALIDATION SPEARMAN.

    Selecting on Spearman rather than on the loss is deliberate: Spearman is the
    primary metric, and the two do not always move together — Huber
    loss keeps improving by tightening calibration long after the rank ordering
    has stopped improving.

    `scaler` here is a feature standardiser, NOT a gradient scaler. There is no
    GradScaler anywhere in this project: this GPU supports bf16, which has
    fp32's dynamic range and therefore cannot underflow gradients to zero the
    way fp16 does.
    """
    from scipy import stats

    mean, std = x_train.mean(0, keepdims=True), x_train.std(0, keepdims=True) + 1e-6

    def to_tensor(x, y):
        return (torch.tensor((x - mean) / std, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.float32, device=device))

    xt, yt = to_tensor(x_train, y_train)
    xv, yv = to_tensor(x_val, y_val)

    head = make_head(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=HEAD_LR, weight_decay=0.01)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    best_rho, best_state, best_epoch = -np.inf, None, 0

    for epoch in range(HEAD_EPOCHS):
        head.train()
        permutation = torch.randperm(len(xt), device=device)
        for start in range(0, len(xt), HEAD_BATCH):
            batch = permutation[start:start + HEAD_BATCH]
            optimizer.zero_grad()          # gradients accumulate by default
            loss = loss_fn(head(xt[batch]).squeeze(-1), yt[batch])
            loss.backward()                # populate .grad
            optimizer.step()               # apply the update

        head.eval()  # switches Dropout off — leaving it on would add noise here
        with torch.no_grad():
            predictions = head(xv).squeeze(-1).cpu().numpy()
        rho = stats.spearmanr(y_val, predictions).statistic

        if rho > best_rho:
            best_rho, best_epoch = rho, epoch
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            if verbose:
                print(f"      early stop at epoch {epoch} "
                      f"(best val Spearman {best_rho:+.3f} @ epoch {best_epoch})")
            break

    head.load_state_dict(best_state)
    head.eval()
    head._standardisation = (mean, std)  # carried along so predict() matches
    if verbose and best_epoch == HEAD_EPOCHS - 1:
        print(f"      ran the full {HEAD_EPOCHS} epochs "
              f"(best val Spearman {best_rho:+.3f}) — consider raising HEAD_EPOCHS")
    return head


@torch.no_grad()
def predict(head: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    mean, std = head._standardisation
    tensor = torch.tensor((x - mean) / std, dtype=torch.float32, device=device)
    return head(tensor).squeeze(-1).cpu().numpy()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-file", default="splits.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device_info = describe_device()
    device = torch.device("cuda" if device_info["cuda_available"] else "cpu")
    print(f"Device: {device}  ({device_info.get('device_name', 'cpu')})\n")

    df = apply_length_policy(load_frames(splits_file=args.splits_file),
                             max_tokens=args.max_tokens)

    print("\nEmbeddings (frozen backbone — computed once, reused by all three heads):")
    wt, mut = build_or_load_cache(df, args.backbone, device, args.cache_dir)

    features = build_features(df, wt, mut)
    targets = df["ddg"].to_numpy(dtype=np.float32)
    print(f"\n  features {features.shape}  (hidden x 3 = [wt, mut, mut-wt])")

    from sklearn.linear_model import RidgeCV
    from src.evaluate import evaluate_predictions, format_report, save_report

    tag = args.backbone.split("/")[-1]
    split_seed = (args.splits_file.replace("splits_seed", "").replace(".csv", "")
                  if args.splits_file != "splits.csv" else "42")

    for split_column in SPLIT_COLUMNS:
        print(f"\n{'=' * 72}\n  training on {split_column}\n{'=' * 72}")
        masks = {s: (df[split_column] == s).to_numpy() for s in ("train", "val", "test")}
        for name, mask in masks.items():
            print(f"    {name:<6} {mask.sum():>5} rows")

        # --- the MLP head (the siamese head, frozen backbone) ----------------
        head = train_head(
            features[masks["train"]], targets[masks["train"]],
            features[masks["val"]], targets[masks["val"]], device,
        )
        mlp_predictions = predict(head, features, device)

        # --- a ridge regression, as a floor for the probe itself -------------
        # If a closed-form linear model matches the MLP, the nonlinearity is not
        # earning its place. RidgeCV picks alpha by leave-one-out on TRAIN only.
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 20))
        ridge.fit(features[masks["train"]], targets[masks["train"]])
        ridge_predictions = ridge.predict(features)
        print(f"      ridge alpha = {ridge.alpha_:.1f}")

        for label, values in (("mlp", mlp_predictions), ("ridge", ridge_predictions)):
            predictions = pd.DataFrame({"uid": df["uid"], "y_pred": values})
            predictions.to_csv(
                args.results_dir / f"preds_linear_probe_{label}_{tag}_{split_column}"
                f"_split{split_seed}_seed{args.seed}.csv",
                index=False,
            )
            report = evaluate_predictions(
                predictions,
                name=f"linear_probe_{label}_{tag}_{split_column}"
                     f"_split{split_seed}_seed{args.seed}",
                trained_on=split_column,
                splits_file=args.splits_file,
            )
            print(format_report(report))
            save_report(report, args.results_dir)

    print(f"\n  metrics -> {args.results_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
