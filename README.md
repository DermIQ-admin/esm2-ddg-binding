# Predicting ΔΔG of binding from sequence with fine-tuned ESM-2

> **Status: in progress.** Environment and scaffolding are done; the data pipeline is next.
>
> The prose sections below are Johannes's to write — PLAN.md §10 is explicit that the
> reasoning should be his own words, not drafted for him. The Setup section is filled in
> because it's factual and §15 requires it to actually work from a clean clone.
>
> Delete this blockquote before the repo goes public.

[One-line hook: I took the interface energetics I published on and built a model for it.]

## Motivation

<!-- 2-3 sentences connecting to the Nat Comms / Cell work on interface energetics.
     Don't overclaim — just state the genuine connection. -->

## Problem

<!-- ΔΔG(binding) prediction, SKEMPI 2.0, why sequence-only, why single-point for v1. -->

## Method

<!-- Siamese ESM-2: one shared backbone encodes the wild-type and mutant complexes,
     features are [wt, mut, mut-wt], an MLP head regresses ΔΔG. Mean-pooling in v1.
     Plus the two baselines and why each exists. -->

## The methodological point that matters most

<!-- Complex-level vs. mutation-level split, both numbers side by side, why the gap
     exists. Per PLAN.md §9 this is the most interview-worthy content in the repo —
     it belongs above the results table, not buried under it. -->

## Results

| Model                    | Split          | Spearman | Pearson | RMSE |
|--------------------------|----------------|----------|---------|------|
| Zero-shot ESM-2          | —              |          |         |      |
| Frozen embeddings + head | complex-level  |          |         |      |
| Fine-tuned ESM-2 (150M)  | complex-level  |          |         |      |
| Fine-tuned ESM-2 (150M)  | mutation-level |          |         |      |

## Limitations

<!-- Sequence-only vs. structure-aware; small dataset; single-point only; honest about
     what this doesn't do. PLAN.md §2: a modest, correctly-reported number on an honest
     split is worth more than an inflated one from a leaky split — say so explicitly. -->

## Decisions

See [DECISIONS.md](DECISIONS.md) for why the split, loss, pooling and backbone size are
what they are.

## Reproduce

Requires Python ≥3.10 and, for training, an NVIDIA GPU. Developed on Windows 11 with
Python 3.14 and an RTX 4090 (CUDA 13.0).

```powershell
git clone <repo-url>
cd esm2-ddg-binding

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` pins the CUDA build of PyTorch via `--extra-index-url`. For a
CPU-only install, drop that line and the `+cu130` suffix on `torch`.

Verify the GPU is visible before training:

```powershell
python -m src.utils
```

Then:

```powershell
python -m src.data.download        # fetch SKEMPI 2.0 into data/raw/ (not committed)
python -m src.data.parse_skempi    # parse mutations, compute ΔΔG, write data/processed/
python -m pytest tests/ -q         # enforce the no-leakage split invariant
python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
```

## Related work

<!-- MINT, ProBASS, AbTune, EJP Lab, Seq2Bind, 3D-ΔΔG — in your own words, with links.
     PLAN.md §2 and §16 have the list and the framing. -->

## License

MIT — see [LICENSE](LICENSE).
