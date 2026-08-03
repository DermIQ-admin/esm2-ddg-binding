"""Parse SKEMPI, build complex sequences, compute ddG, apply the v1 filters.

Implements PLAN.md sections 6.2-6.4.

WHY THIS MODULE IS MORE THAN A CSV PARSE
----------------------------------------
SKEMPI's CSV contains no amino-acid sequences. `#Pdb` is a PDB code plus chain
IDs (e.g. `1CSE_E_I`), so the sequences ESM-2 consumes have to be reconstructed
from the structures. SKEMPI ships a `.mapping` file per structure that does
exactly this, and it is the authoritative bridge between the CSV's residue
numbering and a sequence index:

    LEU I  45  38
    ^    ^   ^   ^
    |    |   |   +-- "cleaned" numbering: per chain, 1-based, CONTIGUOUS
    |    |   +------ PDB numbering: has gaps and insertion codes
    |    +---------- chain
    +--------------- residue

Verified against the full dataset before this module was written:
  * cleaned numbering is contiguous 1..N on all 918 chains, no exceptions,
    so `cleaned_position - 1` indexes directly into that chain's sequence
  * the wild-type residue in `Mutation(s)_cleaned` matches the structure for
    5112 / 5112 single-point rows (100%)

Output (two files, normalized rather than one wide table — the sequence for a
complex is ~500 characters and would otherwise repeat across the ~15 mutations
each complex has):

    data/processed/complexes.csv   one row per complex: the WT sequence
    data/processed/mutations.csv   one row per mutation: ddG + a mutation index

Only the WT sequence is stored. The mutant differs by a single character, so it
is derived on load via `apply_mutation` — smaller, and it removes a whole class
of "the two files disagree" bug.

Usage:
    python -m src.data.parse_skempi
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MAPPING_DIR = RAW_DIR / "PDBs" / "PDBs"

R_GAS = 1.987e-3  # kcal/(mol*K), PLAN.md section 6.3
DEFAULT_TEMP_K = 298.15

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
# 43 residues across the dataset are non-standard (UNK, SEP, PCA, MLY, ...).
# ESM-2's vocabulary has an <unk>-like "X" token, so mapping them to X is both
# faithful and safe. Silently dropping them would shift every downstream index.
UNKNOWN_AA = "X"

# `Mutation(s)_cleaned` format: wild-type AA, chain, position, mutant AA.
# Verified to match all 5112 single-point rows. (`Mutation(s)_PDB` does NOT —
# 35 entries carry insertion codes like `RD100bA` — which is why we use the
# cleaned column throughout.)
MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+)([A-Z])$")


# --------------------------------------------------------------------------
# Sequence reconstruction
# --------------------------------------------------------------------------

def read_mapping(pdb_id: str) -> dict[str, str]:
    """Return {chain: sequence} for one structure, from its .mapping file.

    The file is FIXED-WIDTH, not whitespace-delimited. This matters: when a
    residue number reaches four digits the chain and number run together —
    `'LYS A1000  674'` splits into three fields and yields chain "A1000".
    Insertion codes ('THR E 116A 113') need the number field to be 5 wide.
    These slices parse all 172,913 lines in the dataset without a failure.
    """
    path = MAPPING_DIR / f"{pdb_id.upper()}.mapping"
    by_chain: dict[str, dict[int, str]] = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        resname = line[0:3].strip()
        chain = line[4:5].strip()
        cleaned = int(line[10:].strip())
        by_chain.setdefault(chain, {})[cleaned] = THREE_TO_ONE.get(resname, UNKNOWN_AA)

    # Cleaned numbering is contiguous 1..N per chain (verified across the whole
    # dataset), so sorting by it reconstructs the sequence exactly.
    return {
        chain: "".join(residues[i] for i in sorted(residues))
        for chain, residues in by_chain.items()
    }


def build_complex(pdb_field: str) -> tuple[str, dict[str, int]]:
    """Concatenate the chains named in a `#Pdb` field into one sequence.

    `1CSE_E_I` -> PDB 1CSE, partner 1 is chain E, partner 2 is chain I.
    Partner groups can hold several chains (an antibody's `HL`, say), so every
    character in each group is a chain ID.

    Returns the concatenated sequence and {chain: offset}, where offset is where
    that chain starts in the concatenation. A mutation's index in the complex is
    then `offset[chain] + cleaned_position - 1`.

    The chains are concatenated with no separator token: ESM-2 has no
    chain-break token, and PLAN.md section 7.1 specifies feeding the
    concatenated interacting chains. Order follows `#Pdb` so it is reproducible.
    """
    parts = pdb_field.split("_")
    pdb_id, groups = parts[0], parts[1:]
    chains = read_mapping(pdb_id)

    sequence, offsets = [], {}
    cursor = 0
    for group in groups:
        for chain in group:
            seq = chains[chain]
            offsets[chain] = cursor
            sequence.append(seq)
            cursor += len(seq)

    return "".join(sequence), offsets


def apply_mutation(sequence: str, index: int, wt_aa: str, mut_aa: str) -> str:
    """Return `sequence` with position `index` substituted, verifying the WT.

    The assert is load-bearing, not decoration (PLAN.md section 7.1). An
    off-by-one here would silently mutate the wrong residue and quietly poison
    every label in the dataset — the single most common bug in this kind of
    code. Crashing loudly is strictly better than training on it.
    """
    actual = sequence[index]
    assert actual == wt_aa, (
        f"expected wild-type {wt_aa!r} at index {index}, found {actual!r}"
    )
    return sequence[:index] + mut_aa + sequence[index + 1:]


# --------------------------------------------------------------------------
# Thermodynamics
# --------------------------------------------------------------------------

def compute_ddg(kd_wt_m: float, kd_mut_m: float, temp_k: float = DEFAULT_TEMP_K) -> float:
    """ddG = dG(mutant) - dG(wild-type), in kcal/mol.

    Positive = destabilizing (weaker binding). Negative = stabilizing.

    Sanity check from PLAN.md section 6.3: a mutant Kd of 1 uM against a
    wild-type Kd of 1 nM (1000x weaker binding) gives ddG ~ +4.09 kcal/mol.
    """
    dg_wt = R_GAS * temp_k * np.log(kd_wt_m)
    dg_mut = R_GAS * temp_k * np.log(kd_mut_m)
    return dg_mut - dg_wt


def parse_temperature(value: object) -> float:
    """Pull a temperature in Kelvin out of SKEMPI's messy Temperature column.

    2351 rows record the literal string `298(assumed)`, so `astype(float)`
    raises and `pd.to_numeric(errors="coerce")` would throw the value away and
    fall back to the default. Extracting the leading digits keeps it.
    """
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else DEFAULT_TEMP_K


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

# SKEMPI's headers contain '#', '(', ')' and spaces, none of which survive
# attribute access on an itertuples row (`#Pdb` becomes `_0`). Renaming once,
# up front, keeps the rest of the module referring to columns by name — indexing
# by position would break silently if SKEMPI ever reorders its export.
COLUMN_RENAMES = {
    "#Pdb": "pdb_field",
    "Mutation(s)_PDB": "mutation_pdb",
    "Mutation(s)_cleaned": "mutation_cleaned",
    "iMutation_Location(s)": "interface_location",
    "Hold_out_type": "hold_out_type",
    "Hold_out_proteins": "hold_out_proteins",
    "Affinity_mut (M)": "affinity_mut_raw",
    "Affinity_mut_parsed": "affinity_mut",
    "Affinity_wt (M)": "affinity_wt_raw",
    "Affinity_wt_parsed": "affinity_wt",
    "Temperature": "temperature_raw",
    "Method": "method",
    "Protein 1": "protein_1",
    "Protein 2": "protein_2",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "skempi_v2.csv", sep=";")
    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise KeyError(f"SKEMPI schema changed — missing columns: {sorted(missing)}")
    return df.rename(columns=COLUMN_RENAMES)


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the PLAN.md section 6.4 filters, reporting each drop."""
    print(f"  all rows                                  : {len(df)}")

    n_mut = df["mutation_cleaned"].astype(str).str.count(",") + 1
    df = df[n_mut == 1].copy()
    print(f"  single-point only                         : {len(df)}")

    # Section 6.3: the "abolishes binding" entries are inequalities or bounds,
    # not measurements. SKEMPI's *_parsed columns strip the < / > and hand back
    # a bare number, so filtering on NaN alone would silently keep 127 rows
    # whose "Kd" is really a bound — fabricated precision. Filter on the RAW
    # columns, where the inequality marker survives.
    bounded = (
        df["affinity_mut_raw"].astype(str).str.contains(r"[<>~]")
        | df["affinity_wt_raw"].astype(str).str.contains(r"[<>~]")
    )
    df = df[~bounded].copy()
    print(f"  minus inequality / abolished-binding rows : {len(df)}  (dropped {int(bounded.sum())})")

    kd_wt = pd.to_numeric(df["affinity_wt"], errors="coerce")
    kd_mut = pd.to_numeric(df["affinity_mut"], errors="coerce")
    usable = kd_wt.notna() & kd_mut.notna() & (kd_wt > 0) & (kd_mut > 0)
    df = df[usable].copy()
    print(f"  minus unusable Kd                         : {len(df)}")

    return df


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn filtered SKEMPI rows into the complexes and mutations tables."""
    complexes: dict[str, dict] = {}
    records = []
    seen: collections.Counter = collections.Counter()

    for row in df.itertuples(index=False):
        if row.pdb_field not in complexes:
            sequence, offsets = build_complex(row.pdb_field)
            complexes[row.pdb_field] = {"sequence": sequence, "offsets": offsets}
        entry = complexes[row.pdb_field]

        mutation = str(row.mutation_cleaned)
        match = MUTATION_RE.match(mutation)
        if match is None:
            raise ValueError(f"unparseable mutation {mutation!r} for {row.pdb_field}")
        wt_aa, chain, position, mut_aa = match.groups()

        index = entry["offsets"][chain] + int(position) - 1
        # Raises if the WT residue disagrees with the structure.
        apply_mutation(entry["sequence"], index, wt_aa, mut_aa)

        temp_k = parse_temperature(row.temperature_raw)
        kd_wt, kd_mut = float(row.affinity_wt), float(row.affinity_mut)

        # Stable key so splits.csv can join on identity rather than row order.
        # The same mutation legitimately appears more than once (measured by
        # different methods or in different papers), so a running count
        # disambiguates rather than pretending the pair is unique.
        occurrence = seen[row.pdb_field, mutation]
        seen[row.pdb_field, mutation] += 1

        records.append({
            "uid": f"{row.pdb_field}|{mutation}|{occurrence}",
            "pdb_field": row.pdb_field,
            "pdb_id": row.pdb_field.split("_")[0].upper(),
            "hold_out_proteins": row.hold_out_proteins,
            "hold_out_type": row.hold_out_type,
            "mutation": mutation,
            "wt_aa": wt_aa,
            "chain": chain,
            "position": int(position),
            "mut_aa": mut_aa,
            "mutation_index": index,
            "interface_location": row.interface_location,
            "temperature_k": temp_k,
            "kd_wt_m": kd_wt,
            "kd_mut_m": kd_mut,
            "ddg": compute_ddg(kd_wt, kd_mut, temp_k),
            "method": row.method,
        })

    mutations = pd.DataFrame.from_records(records)
    complexes_df = pd.DataFrame.from_records([
        {
            "pdb_field": field,
            "pdb_id": field.split("_")[0].upper(),
            "sequence": entry["sequence"],
            "length": len(entry["sequence"]),
            "chain_offsets": json.dumps(entry["offsets"]),
        }
        for field, entry in complexes.items()
    ])
    return complexes_df, mutations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()

    # The sign convention is worth re-checking every run rather than trusting a
    # comment: 1 uM mutant against 1 nM wild-type is 1000x weaker binding.
    check = compute_ddg(1e-9, 1e-6)
    assert abs(check - 4.09) < 0.01, f"sign convention broken: got {check:+.3f}"
    print(f"ddG sign check: 1 uM vs 1 nM -> {check:+.3f} kcal/mol (expected +4.09)  OK\n")

    print("Filtering (PLAN.md section 6.4):")
    df = filter_rows(load_raw())

    print("\nBuilding sequences and computing ddG...")
    complexes, mutations = build_dataset(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    complexes.to_csv(args.out_dir / "complexes.csv", index=False)
    mutations.to_csv(args.out_dir / "mutations.csv", index=False)

    print(f"\n  complexes: {len(complexes):>5}  -> {args.out_dir / 'complexes.csv'}")
    print(f"  mutations: {len(mutations):>5}  -> {args.out_dir / 'mutations.csv'}")
    d = mutations["ddg"]
    print(f"\n  ddG: mean {d.mean():+.3f}  median {d.median():+.3f}  "
          f"std {d.std():.3f}  range [{d.min():+.2f}, {d.max():+.2f}]")
    print(f"  complex length: median {complexes['length'].median():.0f}  "
          f"max {complexes['length'].max()}")


if __name__ == "__main__":
    main()
