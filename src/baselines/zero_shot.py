"""Zero-shot baseline: ESM-2 masked-marginal mutation scoring. No training.

The method is Meier et al. 2021 (ESM-1v), applied to ESM-2.

This is the cheapest baseline and is built FIRST: the build order is
each baseline is insurance against the next one silently failing. If the
fine-tuned model cannot beat masked-marginal scoring, something is wrong.

THE METHOD
----------
Mask the mutated position, then read off how much more (or less) the language
model likes the mutant residue than the wild-type one, given the surrounding
context:

    score = log P(mut_aa | context) - log P(wt_aa | context)

Both terms come from the SAME forward pass on the SAME masked input, so this is
one softmax over the vocabulary at one position — not two separate passes.

WHAT THE NUMBER MEANS, AND ITS SIGN
-----------------------------------
The score is a fitness-like quantity: high means "this substitution looks
natural here". ddG is the opposite convention — positive means destabilizing.
So we expect a NEGATIVE Spearman between score and ddG, and that negative
correlation is the baseline working, not a bug. This module reports the signed
value and states the direction rather than quietly taking an absolute value.

WHY THIS IS ONLY AN APPROXIMATION HERE
--------------------------------------
ESM-2 was pretrained on single chains and scores sequence plausibility, which
is closer to protein STABILITY than to BINDING. We are asking it about ddG of
binding. We therefore score the CONCATENATED interacting chains so
the masked prediction has at least some awareness of the partner, but a
language model has no explicit notion of an interface. Expect a modest
correlation. A modest, honestly-reported floor is the point.

MEASURED, NOT ASSUMED (session 3): `--context` runs that approximation as an
ablation. Scoring the mutated chain ALONE instead of the full complex, on the
150M backbone over all 4814 mutations:

    context     WT recovery   pooled Spearman   mean per-complex Spearman
    complex        24.9%          -0.062                 +0.014
    chain          25.0%          -0.091                 +0.002

    agreement between the two scorings: Spearman +0.888

Deleting the entire binding partner changes almost nothing. That is the real
explanation for the near-zero result, and it is worth more than the number
itself: ESM-2's masked-marginal score is effectively BLIND to the binding
partner, so it cannot express binding ddG even in principle. It measures
"does this residue belong in this chain", not "does this residue hold the
interface together". Section 7.3's siamese model exists to fix exactly this,
and this is the floor it has to clear.

TWO EFFICIENCIES WORTH KNOWING
------------------------------
1. The log-probabilities at a masked position do not depend on which mutant we
   are asking about — the mutant residue only picks a coordinate out of the
   resulting distribution. So we do ONE forward pass per unique
   (complex, position) pair and reuse it for every mutation at that site.
   SKEMPI is alanine-scanning-heavy, so many positions carry several mutations.
2. Every masked variant of a given complex has identical length, so they batch
   together with no padding waste at all.

Usage:
    python -m src.baselines.zero_shot                     # 35M, per config.yaml
    python -m src.baselines.zero_shot --backbone facebook/esm2_t30_150M_UR50D
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import MAX_TOKENS, apply_length_policy, load_frames
from src.utils import describe_device, set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# configs/config.yaml: baselines.zero_shot_model, the smallest ESM-2.
DEFAULT_BACKBONE = "facebook/esm2_t12_35M_UR50D"

# Masked variants of one complex all share a length, so we batch by a TOKEN
# budget rather than a fixed count: 16 variants of a 400-token complex cost the
# same as 3 variants of a 2048-token one. Keeps memory flat across complexes.
TOKEN_BUDGET = 16_384


@torch.inference_mode()
def score_complex(
    sequence: str,
    positions: list[int],
    model,
    tokenizer,
    device: torch.device,
    token_budget: int = TOKEN_BUDGET,
) -> dict[int, torch.Tensor]:
    """Return {sequence_position: log-probability vector over the vocabulary}.

    One entry per requested position, each from a forward pass in which THAT
    position (and only that position) was replaced by <mask>.

    `@torch.inference_mode()` is the modern, slightly stronger `no_grad()`: it
    switches off autograd bookkeeping entirely for everything inside, which is
    what we want since nothing here is trained.
    """
    encoded = tokenizer(sequence, return_tensors="pt")
    base_ids = encoded["input_ids"][0]
    attention = encoded["attention_mask"][0]

    # +1 for the leading <cls>. Verified against the real tokenizer (overhead is
    # exactly 2 tokens, <cls> ... <eos>) rather than trusted from the plan.
    token_indices = [p + 1 for p in positions]

    per_batch = max(1, token_budget // len(base_ids))
    log_probs: dict[int, torch.Tensor] = {}

    for start in range(0, len(positions), per_batch):
        chunk_pos = positions[start:start + per_batch]
        chunk_tok = token_indices[start:start + per_batch]

        # One row per position: the same sequence, each with a different single
        # token masked. `repeat` copies the row; we then overwrite one cell per row.
        ids = base_ids.unsqueeze(0).repeat(len(chunk_pos), 1).clone()
        rows = torch.arange(len(chunk_pos))
        ids[rows, chunk_tok] = tokenizer.mask_token_id

        mask = attention.unsqueeze(0).repeat(len(chunk_pos), 1)
        logits = model(input_ids=ids.to(device),
                       attention_mask=mask.to(device)).logits

        # Softmax over the vocabulary at the masked position of each row, in
        # log space — log_softmax is numerically stable where log(softmax(x))
        # is not. .float() because bf16 has too few mantissa bits for the small
        # differences between two amino-acid log-probabilities to survive.
        selected = logits[rows, chunk_tok].float()
        chunk_log_probs = torch.log_softmax(selected, dim=-1).cpu()

        for i, position in enumerate(chunk_pos):
            log_probs[position] = chunk_log_probs[i]

    return log_probs


def chain_span(pdb_field: str, chain: str, chain_offsets: dict, total_length: int) -> tuple[int, int]:
    """(start, end) of one chain inside the concatenated complex.

    `chain_offsets` records where each chain starts. A chain therefore runs
    until the next chain begins, or to the end of the complex if it is last.
    Dict order is the `#Pdb` order that built the concatenation, so we take the
    smallest start greater than this one rather than relying on that order.
    """
    start = chain_offsets[chain]
    later = [s for s in chain_offsets.values() if s > start]
    return start, (min(later) if later else total_length)


def masked_marginal_scores(
    df: pd.DataFrame,
    model,
    tokenizer,
    device: torch.device,
    context: str = "complex",
    verbose: bool = True,
) -> pd.DataFrame:
    """Score every mutation in `df`. Returns uid, y_pred.

    `df` must carry `sequence`, `mutation_index`, `wt_aa`, `mut_aa`.

    `context="complex"` feeds the full concatenation (the default method).
    `context="chain"` feeds only the chain carrying the mutation — the ablation
    described in the module docstring. Positions are translated into the chain's
    local coordinates and the wild-type assert re-run there, so an error in the
    span arithmetic crashes rather than scoring a shifted residue.
    """
    if context not in ("complex", "chain"):
        raise ValueError(f"context must be 'complex' or 'chain', got {context!r}")
    aa_to_id = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in set(df["wt_aa"]) | set(df["mut_aa"])}
    unknown = [aa for aa, i in aa_to_id.items() if i is None or i == tokenizer.unk_token_id]
    assert not unknown, f"amino acids missing from the ESM-2 vocabulary: {unknown}"

    records = []
    complexes = list(df.groupby("pdb_field"))
    started = time.time()

    for n, (pdb_field, group) in enumerate(complexes, 1):
        sequence = group["sequence"].iloc[0]

        # Both branches produce {GLOBAL mutation_index: log-probability vector},
        # so everything downstream is identical regardless of context.
        if context == "complex":
            positions = sorted(set(group["mutation_index"].astype(int)))
            log_probs = score_complex(sequence, positions, model, tokenizer, device)
            # Tokenize once per complex for the assert below, not once per row.
            tokens = {i: t for i, t in
                      enumerate(tokenizer.convert_ids_to_tokens(tokenizer(sequence)["input_ids"]))}
        else:
            offsets = json.loads(group["chain_offsets"].iloc[0])
            log_probs, tokens = {}, {}
            for chain, chain_group in group.groupby("chain"):
                start, end = chain_span(pdb_field, chain, offsets, len(sequence))
                subsequence = sequence[start:end]
                local = sorted({int(i) - start for i in chain_group["mutation_index"]})
                assert all(0 <= i < len(subsequence) for i in local), (
                    f"{pdb_field}/{chain}: a mutation index falls outside the "
                    f"chain span [{start}, {end})"
                )
                chain_log_probs = score_complex(
                    subsequence, local, model, tokenizer, device
                )
                chain_tokens = tokenizer.convert_ids_to_tokens(
                    tokenizer(subsequence)["input_ids"]
                )
                for i in local:
                    log_probs[i + start] = chain_log_probs[i]
                    # Key the assert lookup by the GLOBAL token index so the
                    # check below reads the same in both branches.
                    tokens[i + start + 1] = chain_tokens[i + 1]

        for row in group.itertuples(index=False):
            index = int(row.mutation_index)

            # THE LOAD-BEARING ASSERT. An off-by-one here would
            # silently score the wrong residue — the single most common bug in
            # this kind of code — and every downstream number would look
            # plausible and be wrong. We check the token the model actually saw,
            # not the string we think we passed it. Under context="chain" this
            # additionally proves the chain-span arithmetic is right.
            assert tokens[index + 1] == row.wt_aa, (
                f"{row.uid}: expected wild-type {row.wt_aa!r} at token index "
                f"{index + 1}, the tokenizer has {tokens[index + 1]!r}"
            )

            lp = log_probs[index]
            score = float(lp[aa_to_id[row.mut_aa]] - lp[aa_to_id[row.wt_aa]])
            records.append({"uid": row.uid, "y_pred": score})

        if verbose and (n % 25 == 0 or n == len(complexes)):
            elapsed = time.time() - started
            print(f"    {n:>3}/{len(complexes)} complexes  "
                  f"{len(records):>5} mutations  {elapsed:>5.1f}s "
                  f"({elapsed / n:.2f}s per complex)")

    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--context", default="complex", choices=("complex", "chain"),
                        help="score the full concatenated complex "
                             "or only the chain carrying the mutation (ablation)")
    parser.add_argument("--limit-complexes", type=int, default=None,
                        help="score only the first N complexes (a quick sanity run)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    set_seed(42)  # nothing here is stochastic, but pin it so the run is quotable
    device_info = describe_device()
    device = torch.device("cuda" if device_info["cuda_available"] else "cpu")
    print(f"Device: {device}  ({device_info.get('device_name', 'cpu')})")

    # AutoModelForMaskedLM, NOT AutoModel — we need the language-model head that
    # projects hidden states onto the vocabulary. AutoModel would return hidden
    # states only and there would be nothing to take a softmax over.
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    print(f"Loading {args.backbone} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForMaskedLM.from_pretrained(
        args.backbone,
        # bf16 on Ada halves the memory and the bandwidth for free. The scores
        # themselves are recovered in fp32 inside score_complex.
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()  # .eval() disables dropout — it is on by default

    print("\nLoading data (same length policy as the trained models, so the "
          "comparison is over identical rows):")
    df = apply_length_policy(load_frames(), max_tokens=args.max_tokens)
    if args.limit_complexes:
        keep = sorted(df["pdb_field"].unique())[:args.limit_complexes]
        df = df[df["pdb_field"].isin(keep)]
        print(f"  --limit-complexes: {len(df)} rows from {len(keep)} complexes")

    unique_sites = df.groupby("pdb_field")["mutation_index"].nunique().sum()
    print(f"\nScoring {len(df)} mutations across {df['pdb_field'].nunique()} "
          f"complexes ({unique_sites} unique masked positions — "
          f"{len(df) / unique_sites:.1f} mutations per forward pass):")

    predictions = masked_marginal_scores(df, model, tokenizer, device,
                                         context=args.context)

    tag = args.backbone.split("/")[-1]
    if args.context != "complex":
        tag = f"{tag}_context-{args.context}"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / f"preds_zero_shot_{tag}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\n  {len(predictions)} scores -> {out_path}")

    # Report through the shared harness. ranking_only=True because a
    # log-likelihood ratio is not kcal/mol and an RMSE against ddG would be a
    # number with no meaning.
    from src.evaluate import evaluate_predictions, format_report, save_report

    report = evaluate_predictions(predictions, name=f"zero_shot_{tag}", ranking_only=True)
    print(format_report(report))
    print(f"\n  metrics -> {save_report(report, args.results_dir)}")

    spearman = report["splits"]["split_pdb_id"]["test"]["spearman"]
    direction = "NEGATIVE, as expected" if spearman < 0 else "POSITIVE — investigate"
    print(f"\n  Sign check: Spearman on split_pdb_id/test is {spearman:+.3f} "
          f"({direction}).")
    print("  A fitness-like score should anti-correlate with ddG: destabilizing "
          "mutations\n  (positive ddG) are the ones the language model finds "
          "unlikely (negative score).")


if __name__ == "__main__":
    main()
