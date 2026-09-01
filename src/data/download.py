"""Download the raw SKEMPI 2.0 dataset.


The raw data is not committed to the repo (see .gitignore) — redistribution and
repo bloat both cut against it. This script is the reproducible substitute:
anyone cloning the repo runs it to recreate `data/raw/`.

Source: https://life.bsc.es/pid/skempi2/

TWO files are needed, not one. The CSV carries the mutations and affinities but
NO amino-acid sequences — `#Pdb` is just a PDB code plus chain IDs (e.g.
`1CSE_E_I`). The sequences ESM-2 consumes have to come from the structures, so
we fetch the PDB tarball too.

Usage:
    python -m src.data.download
    python -m src.data.download --force     # re-download even if present
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://life.bsc.es/pid/skempi2/database/download"
CSV_URL = f"{BASE_URL}/skempi_v2.csv"
PDB_TGZ_URL = f"{BASE_URL}/SKEMPI2_PDBs.tgz"

# Resolve paths relative to the repo root, not the current working directory, so
# the script behaves the same whether it's run from the repo root or elsewhere.
# __file__ is src/data/download.py, so three .parent hops land on the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Stream `url` to `dest`, skipping the work if it's already there.

    Streaming in chunks rather than reading the whole response into memory is
    the habit worth having — it costs nothing here (30 MB) but is what keeps
    the same code working when a file is 30 GB.
    """
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1024**2
        print(f"  [skip] {dest.name} already present ({size_mb:.2f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to a temporary name and rename only on success, so an interrupted
    # download can never leave a truncated file that looks complete to the
    # `dest.exists()` check above.
    tmp = dest.with_suffix(dest.suffix + ".part")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with open(tmp, "wb") as fh, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=f"  {dest.name}",
        ) as bar:
            while chunk := response.read(64 * 1024):
                fh.write(chunk)
                bar.update(len(chunk))

    tmp.replace(dest)
    return dest


def extract_pdbs(tgz_path: Path, dest_dir: Path, force: bool = False) -> Path:
    """Unpack the PDB tarball into `dest_dir`."""
    if dest_dir.exists() and any(dest_dir.iterdir()) and not force:
        n = sum(1 for _ in dest_dir.rglob("*") if _.is_file())
        print(f"  [skip] {dest_dir.name}/ already extracted ({n} files)")
        return dest_dir

    if force and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tgz_path, "r:gz") as tar:
        members = tar.getmembers()
        # filter="data" blocks the classic tar attacks — absolute paths and
        # ../ entries that would write outside dest_dir. It became the default
        # in Python 3.14, but passing it explicitly keeps the script correct on
        # older interpreters too, and documents the intent.
        for member in tqdm(members, desc=f"  extracting", unit="file"):
            tar.extract(member, path=dest_dir, filter="data")

    return dest_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download and re-extract"
    )
    args = parser.parse_args()

    print(f"Downloading SKEMPI 2.0 into {RAW_DIR}")

    csv_path = download_file(CSV_URL, RAW_DIR / "skempi_v2.csv", args.force)
    tgz_path = download_file(PDB_TGZ_URL, RAW_DIR / "SKEMPI2_PDBs.tgz", args.force)
    pdb_dir = extract_pdbs(tgz_path, RAW_DIR / "PDBs", args.force)

    print("\nDone.")
    print(f"  CSV       : {csv_path}  ({csv_path.stat().st_size / 1024**2:.2f} MB)")
    n_pdb = sum(1 for _ in pdb_dir.rglob("*") if _.is_file())
    print(f"  Structures: {pdb_dir}  ({n_pdb} files)")


if __name__ == "__main__":
    main()
