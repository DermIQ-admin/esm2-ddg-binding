"""Download the raw SKEMPI 2.0 dataset.

Implements PLAN.md section 6.1.

The raw CSV is not committed to the repo (see .gitignore) — redistribution and
repo bloat both cut against it. This script is the reproducible substitute:
anyone cloning the repo runs it to recreate `data/raw/`.

Source: https://life.bsc.es/pid/skempi2/

TODO(session 2): implement.
  - Fetch the SKEMPI 2.0 CSV into data/raw/
  - Do NOT assume the schema here; section 6.2 says inspect the real file first
  - Semicolon-delimited on export (`sep=";"`) per section 6.2 — verify, don't trust
  - Be idempotent: skip the download if the file already exists
"""
