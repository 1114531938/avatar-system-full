#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
INVENTORY="$ROOT/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed/nersemble_subjects.tsv"
BASE_AVATAR_ID=2001
MODEL_SUFFIX="gaussian_train"
LIMIT=""
SUBJECTS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --inventory) INVENTORY="$2"; shift 2 ;;
    --base-avatar-id) BASE_AVATAR_ID="$2"; shift 2 ;;
    --model-suffix) MODEL_SUFFIX="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --subjects) SUBJECTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$INVENTORY" ]]; then
  echo "inventory not found: $INVENTORY" >&2
  exit 1
fi

python - "$INVENTORY" "$LIMIT" "$SUBJECTS" "$BASE_AVATAR_ID" <<'PY' | while IFS=$'\t' read -r avatar_id subject_id source_dir; do
from __future__ import annotations
import csv
import sys

inventory, limit, subjects, base_avatar_id = sys.argv[1:5]
limit_n = int(limit) if limit else None
subject_set = {s.strip() for s in subjects.split(",") if s.strip()} if subjects else None
avatar_id = int(base_avatar_id)

rows = []
with open(inventory, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if subject_set and row["subject_id"] not in subject_set:
            continue
        rows.append((avatar_id, row["subject_id"], row["source_dir"]))
        avatar_id += 1
        if limit_n is not None and len(rows) >= limit_n:
            break

for avatar_id, subject_id, source_dir in rows:
    print(f"{avatar_id}\t{subject_id}\t{source_dir}")
PY
  MODEL_PATH="$ROOT/data/subjects/$subject_id/$MODEL_SUFFIX"
  echo "[register_nersemble_batch] avatar_id=$avatar_id subject=$subject_id model=$MODEL_PATH"
  CMD=(
    bash "$ROOT/scripts/register_avatar_asset.sh" "$subject_id" "$avatar_id"
    --model "$MODEL_PATH"
    --canonical "$source_dir/canonical_flame_param.npz"
  )
  printf '[register_nersemble_batch] %q ' "${CMD[@]}"; echo
  "${CMD[@]}"
done
