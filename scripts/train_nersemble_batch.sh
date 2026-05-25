#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
INVENTORY="$ROOT/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/nersemble_subjects.tsv"
FAST_30K=0
LIMIT=""
SUBJECTS=""
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inventory) INVENTORY="$2"; shift 2 ;;
    --fast-30k) FAST_30K=1; shift ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --subjects) SUBJECTS="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -f "$INVENTORY" ]]; then
  echo "inventory not found: $INVENTORY" >&2
  exit 1
fi

python - "$INVENTORY" "$LIMIT" "$SUBJECTS" <<'PY' | while IFS=$'\t' read -r subject_id source_dir; do
from __future__ import annotations
import csv
import sys

inventory, limit, subjects = sys.argv[1:4]
limit_n = int(limit) if limit else None
subject_set = {s.strip() for s in subjects.split(",") if s.strip()} if subjects else None

rows = []
with open(inventory, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if subject_set and row["subject_id"] not in subject_set:
            continue
        rows.append((row["subject_id"], row["source_dir"]))
        if limit_n is not None and len(rows) >= limit_n:
            break

for subject_id, source_dir in rows:
    print(f"{subject_id}\t{source_dir}")
PY
  echo "[train_nersemble_batch] subject=$subject_id source=$source_dir"
  mkdir -p "$ROOT/data/subjects/$subject_id"
  CMD=(bash "$ROOT/scripts/train_gaussian_subject.sh" "$subject_id" --source "$source_dir")
  if [[ "$FAST_30K" == "1" ]]; then
    CMD+=(--fast-30k)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi
  printf '[train_nersemble_batch] %q ' "${CMD[@]}"; echo
  "${CMD[@]}"
done
