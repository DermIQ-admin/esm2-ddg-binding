"""Zero-shot baseline: ESM-2 masked-marginal mutation scoring. No training.

Implements PLAN.md section 7.1 (the method from Meier et al. 2021, ESM-1v).

This is the cheapest baseline and should be built FIRST — section 7 is explicit
that each baseline is insurance against the next one silently failing. If the
fine-tuned model can't beat masked-marginal scoring, something is wrong.

METHOD: mask the mutated position, read the model's log-probability for the
mutant residue minus the wild-type residue.

    score = log P(mut_aa | context) - log P(wt_aa | context)

TODO(session 2): implement.
  - masked_marginal_score(sequence, position, wt_aa, mt_aa) -> float
  - Load with AutoModelForMaskedLM (NOT AutoModel — we need the LM head)
  - mask_idx = position + 1 to skip the leading <cls>. VERIFY this against the
    actual tokenizer rather than trusting the +1.
  - KEEP THE ASSERT from section 7.1:
        assert original == wt_aa
    It is not decoration. Off-by-one indexing that silently scores the wrong
    residue is the single most common bug in this kind of code, and the
    assert converts a silent wrong answer into a loud crash.
  - Score the CONCATENATED interacting chains, not a single chain. ESM-2 was
    pretrained on single chains so this is an approximation, but it at least
    gives the masked prediction some awareness of the binding partner — and
    binding ddG is the target here, not stability.
  - Sign: the raw score is a fitness-like quantity, so expect it to correlate
    NEGATIVELY with ddG (destabilizing = positive ddG = low LM likelihood).
    Report |Spearman| honestly and state the direction.
  - Multi-point mutations (out of scope for v1) would sum per-position scores.
"""
