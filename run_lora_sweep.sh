#!/usr/bin/env bash
# LoRA replicate sweep on split_pdb_id -- the primary honest split.
#
# Five replicates matching the partitions the existing finetune and probe runs
# used, so every result pairs against both: three training seeds at a fixed
# partition, and three split seeds at a fixed training seed, overlapping at
# split42/seed42.
#
# `python -u` is not optional. Piping buffers stdout, and short epoch lines
# never fill a 4KB buffer, which makes a long run completely unobservable.
set -u

PY=".venv/Scripts/python.exe"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run() {
  local label="$1"; shift
  echo "=== $label  $(date '+%H:%M:%S') ==="
  "$PY" -u -m src.train --split-column split_pdb_id --lora "$@" \
    > "$LOG_DIR/lora_${label}.log" 2>&1
  local status=$?
  if [ $status -ne 0 ]; then
    echo "!!! $label FAILED (exit $status) -- see $LOG_DIR/lora_${label}.log"
    tail -20 "$LOG_DIR/lora_${label}.log"
  else
    grep -E "best epoch|split_pdb_id .*test" "$LOG_DIR/lora_${label}.log" | head -3
  fi
  return $status
}

echo "LoRA sweep on split_pdb_id, 5 replicates. Started $(date)"
run split42_seed42
run split42_seed43 --seed 43
run split42_seed44 --seed 44
run split43_seed42 --splits-file splits_seed43.csv
run split44_seed42 --splits-file splits_seed44.csv
echo "=== sweep finished $(date) ==="
