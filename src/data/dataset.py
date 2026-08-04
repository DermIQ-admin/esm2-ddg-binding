"""PyTorch Dataset and collate_fn for WT/mutant sequence pairs.

Implements the data-loading half of PLAN.md section 7.3.

WHAT THE MODEL CONSUMES
-----------------------
The siamese model runs ONE shared backbone twice — once on the wild-type
sequence, once on the mutant — so each example is four tensors plus a target:

    wt_ids, wt_mask, mut_ids, mut_mask   ->  ddg

`*_ids` are integer token ids; `*_mask` are 1 for real tokens and 0 for
padding. The mask is not optional bookkeeping: section 7.3 mean-pools with
`(hidden * mask).sum(1) / mask.sum(1)`, so a wrong mask silently averages
padding into the embedding rather than raising. That is the failure mode this
module is written to make impossible.

Only the WT sequence is stored on disk (see parse_skempi). The mutant is
derived here via `apply_mutation`, which re-verifies the wild-type residue on
every single access — the same load-bearing assert as at parse time.

SEQUENCE LENGTH POLICY (decided session 3, against the real tokenizer)
----------------------------------------------------------------------
ESM-2 uses ROTARY position embeddings (`config.position_embedding_type ==
"rotary"`) and `tokenizer.model_max_length` is unset. There is therefore no
hard 1024-token wall — the `max_position_embeddings: 1026` field in the config
is vestigial for this architecture. 1024 is a compute budget and a
pretraining-distribution boundary, not a limit.

Measured over all 316 complexes: tokenizer overhead is exactly 2 tokens
(`<cls>` ... `<eos>`), median 394 tokens, 95th percentile 987, max 3399.

    cap    complexes over   mutations over   MUTATION SITE TRUNCATED AWAY
    1024      15 / 316        189 (3.9%)                77
    1536       6              75  (1.6%)                13
    2048       2              15  (0.3%)                11

That last column is why we do not simply pass `truncation=True`. For those
rows the mutated residue itself falls outside the window, so the WT and mutant
token sequences come out IDENTICAL, `mut_emb - wt_emb` is exactly zero, and the
model trains on examples that carry a real ddG label and no signal at all.
Silent, not loud.

Johannes's call, session 3: run every complex at its FULL length and exclude
only what exceeds MAX_TOKENS = 2048. That drops 2 complexes (1KBH_A_B at 2122,
3VR6_ABCDEF_GH at 3399) and 15 of 4829 rows — 0.3% — while keeping worst-case
attention cost predictable at 2048 tokens. Everything else is fed whole, so
"the concatenated interacting chains" of section 7.1 stays literally true and
binding ddG stays binding ddG.

The exclusion happens HERE, at load time, not in parse_skempi. The processed
CSVs stay a faithful record of what SKEMPI contains; the length cut is a
modelling decision, applied in one visible place and reported every run.

TWO LIMITATIONS TO CARRY INTO THE README
  * rows between 1024 and 2048 tokens (1.6%) run past ESM-2's pretraining crop
    length. Rotary embeddings extrapolate, but quality there is not guaranteed.
  * the two excluded complexes are large multi-chain assemblies, and two of the
    three biggest complexes overall are antibody complexes — a small structured
    cost to the section 13.1 antibody subset, not a random one.

Usage (smoke test — prints shapes and verifies the WT/mutant token diff):
    python -m src.data.dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.parse_skempi import apply_mutation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Tokens, not residues — includes the 2 tokens the tokenizer adds. See the
# module docstring for how this number was chosen. Mirrored in configs/config.yaml.
MAX_TOKENS = 2048
TOKENIZER_OVERHEAD = 2  # <cls> ... <eos>, verified constant across all 316 complexes

SPLIT_COLUMNS = ["split_mutation", "split_pdb_id", "split_hold_out_proteins"]


def load_frames(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Join the three processed tables into one row-per-mutation frame.

    `complexes.csv` is keyed by `pdb_field` and holds the sequence; `splits.csv`
    is keyed by `uid` and holds the three assignments. `validate=` makes pandas
    raise if either join is not the many-to-one / one-to-one it should be —
    a duplicated key here would quietly multiply rows.
    """
    mutations = pd.read_csv(processed_dir / "mutations.csv")
    complexes = pd.read_csv(processed_dir / "complexes.csv")
    splits = pd.read_csv(processed_dir / "splits.csv")

    df = mutations.merge(
        # `chain_offsets` is unused by the Dataset itself but lets the zero-shot
        # baseline carve out a single chain without re-reading complexes.csv.
        complexes[["pdb_field", "sequence", "length", "chain_offsets"]],
        on="pdb_field",
        how="left",
        validate="many_to_one",
    )
    df = df.merge(splits, on="uid", validate="one_to_one")

    assert df["sequence"].notna().all(), "a mutation references a missing complex"
    return df


def apply_length_policy(
    df: pd.DataFrame, max_tokens: int = MAX_TOKENS, verbose: bool = True
) -> pd.DataFrame:
    """Drop rows whose complex exceeds `max_tokens` once tokenized.

    We compute the token count as `length + TOKENIZER_OVERHEAD` rather than
    running the tokenizer, which is exact for ESM-2: its vocabulary is
    character-level over amino acids, so one residue is always one token, and
    the overhead of 2 was verified constant across every complex. Keeping this
    arithmetic means the Dataset does not need a tokenizer to filter.
    """
    n_tokens = df["length"] + TOKENIZER_OVERHEAD
    keep = n_tokens <= max_tokens

    if verbose and not keep.all():
        dropped = df[~keep]
        print(f"  length policy (> {max_tokens} tokens): dropping "
              f"{len(dropped)} / {len(df)} rows "
              f"({100 * len(dropped) / len(df):.1f}%), "
              f"{dropped['pdb_field'].nunique()} complexes")
        for field, sub in dropped.groupby("pdb_field"):
            print(f"    {field:<16} {int(sub['length'].iloc[0]) + TOKENIZER_OVERHEAD:>5} tokens"
                  f"  {len(sub):>3} mutations")

    return df[keep].reset_index(drop=True)


class DdgDataset(Dataset):
    """One (wild-type sequence, mutant sequence, ddG) record per mutation.

    Deliberately returns RAW STRINGS, not tensors. Tokenization happens in
    `collate_fn` so each batch can be padded to its own longest member rather
    than to a global maximum — with lengths ranging 60..2048 tokens here, padding
    everything to 2048 would waste most of the compute on <pad>.

    A PyTorch Dataset is just two methods: `__len__` and `__getitem__(i)`.
    The DataLoader calls `__getitem__` for the indices it wants and hands the
    resulting list to `collate_fn`.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        split: str | None = None,
        split_column: str = "split_pdb_id",
        max_tokens: int = MAX_TOKENS,
        verbose: bool = True,
    ) -> None:
        if split_column not in SPLIT_COLUMNS:
            raise ValueError(f"unknown split column {split_column!r}, expected one of {SPLIT_COLUMNS}")

        df = apply_length_policy(frame, max_tokens=max_tokens, verbose=verbose)
        if split is not None:
            df = df[df[split_column] == split].reset_index(drop=True)
        self.df = df
        self.split = split
        self.split_column = split_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        row = self.df.iloc[i]
        wt_seq = row["sequence"]
        index = int(row["mutation_index"])

        # Re-verifies the wild-type residue on every access. Cheap (one string
        # comparison) and it means an indexing bug can never reach the GPU.
        mut_seq = apply_mutation(wt_seq, index, row["wt_aa"], row["mut_aa"])

        return {
            "uid": row["uid"],
            "wt_seq": wt_seq,
            "mut_seq": mut_seq,
            "ddg": float(row["ddg"]),
            # Index into the SEQUENCE. collate_fn converts it to a token index.
            # Needed by the zero-shot baseline now, and by the section 7.3 v2
            # position-specific pooling later.
            "mutation_index": index,
        }


def make_collate_fn(tokenizer, pad_to_multiple_of: int | None = 8):
    """Build the function that turns a list of records into padded batch tensors.

    `pad_to_multiple_of=8` rounds the padded length up so the matrix dimensions
    suit the GPU's tensor cores — a free speedup on bf16, not a correctness
    concern. Set it to None to see the true lengths.

    WT and mutant are tokenized as two separate batches because they are two
    separate forward passes through the shared backbone. A single-point
    substitution cannot change sequence length, so their masks must come out
    identical — we assert that rather than assume it, since a mismatch would
    mean `apply_mutation` had gone wrong upstream.
    """

    def collate(records: list[dict]) -> dict:
        wt = tokenizer(
            [r["wt_seq"] for r in records],
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=pad_to_multiple_of,
        )
        mut = tokenizer(
            [r["mut_seq"] for r in records],
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=pad_to_multiple_of,
        )

        assert torch.equal(wt["attention_mask"], mut["attention_mask"]), (
            "WT and mutant attention masks differ — a substitution changed the "
            "sequence length, which is impossible for a single-point mutation"
        )

        # +1 for the leading <cls>. Verified against the real tokenizer rather
        # than trusted from PLAN.md section 7.1's comment.
        mutation_token_index = torch.tensor(
            [r["mutation_index"] + 1 for r in records], dtype=torch.long
        )

        # A padded token can never be the mutation site; if it is, the length
        # policy or the index is wrong.
        rows = torch.arange(len(records))
        assert wt["attention_mask"][rows, mutation_token_index].all(), (
            "a mutation site landed on a padding token"
        )

        return {
            "uid": [r["uid"] for r in records],
            "wt_ids": wt["input_ids"],
            "wt_mask": wt["attention_mask"],
            "mut_ids": mut["input_ids"],
            "mut_mask": mut["attention_mask"],
            "mutation_token_index": mutation_token_index,
            # float32 targets: the loss is computed in fp32 even under bf16
            # autocast, so there is nothing to gain from a narrower dtype here.
            "ddg": torch.tensor([r["ddg"] for r in records], dtype=torch.float32),
        }

    return collate


def build_dataloaders(
    tokenizer,
    split_column: str = "split_pdb_id",
    batch_size: int = 8,
    max_tokens: int = MAX_TOKENS,
    processed_dir: Path = PROCESSED_DIR,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """train/val/test DataLoaders for one split definition.

    `num_workers=0` keeps loading in the main process. On Windows each worker
    re-imports the module and re-reads the CSVs, and our `__getitem__` is a
    string slice — the workers would cost more than they save.

    Only the training loader shuffles. Shuffling val/test would change nothing
    about the metrics but would make per-batch debugging output non-reproducible.
    """
    frame = load_frames(processed_dir)
    collate = make_collate_fn(tokenizer)

    loaders = {}
    for i, split in enumerate(("train", "val", "test")):
        dataset = DdgDataset(
            frame,
            split=split,
            split_column=split_column,
            max_tokens=max_tokens,
            verbose=(i == 0),  # the length policy report is identical each time
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate,
            num_workers=num_workers,
        )
    return loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--backbone", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--split-column", default="split_pdb_id", choices=SPLIT_COLUMNS)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    print(f"Building dataloaders for {args.split_column} "
          f"(max_tokens={MAX_TOKENS}):\n")
    loaders = build_dataloaders(
        tokenizer,
        split_column=args.split_column,
        batch_size=args.batch_size,
        processed_dir=args.processed_dir,
    )

    print("\n  split   rows   complexes")
    for name, loader in loaders.items():
        df = loader.dataset.df
        print(f"  {name:<6} {len(df):>5}   {df['pdb_field'].nunique():>5}"
              f"   ddG mean {df['ddg'].mean():+.2f}")

    batch = next(iter(loaders["train"]))
    print("\nOne training batch:")
    for key in ("wt_ids", "wt_mask", "mut_ids", "mut_mask", "ddg", "mutation_token_index"):
        print(f"  {key:<21} {tuple(batch[key].shape)}  {batch[key].dtype}")

    # The point of the whole module: WT and mutant must differ at EXACTLY one
    # token, and that token must be the mutation site.
    print("\nWT vs mutant token difference (must be exactly 1 position each):")
    differing = batch["wt_ids"] != batch["mut_ids"]
    for i, uid in enumerate(batch["uid"]):
        positions = differing[i].nonzero().flatten().tolist()
        site = int(batch["mutation_token_index"][i])
        wt_tok = tokenizer.convert_ids_to_tokens(int(batch["wt_ids"][i, site]))
        mut_tok = tokenizer.convert_ids_to_tokens(int(batch["mut_ids"][i, site]))
        ok = positions == [site]
        print(f"  {uid:<34} differs at {positions}  site={site}  "
              f"{wt_tok}->{mut_tok}  {'OK' if ok else 'FAIL'}")
        assert ok, f"{uid}: expected a difference at {site} only, found {positions}"

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
