#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
DATA_ROOT="$ROOT/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed"
RELEASE_DIR="${1:-$DATA_ROOT/release}"
OUT_TSV="${2:-$DATA_ROOT/nersemble_subjects.tsv}"

if [[ ! -d "$RELEASE_DIR" ]]; then
  echo "release dir not found: $RELEASE_DIR" >&2
  exit 1
fi

python - "$RELEASE_DIR" "$OUT_TSV" <<'PY'
from __future__ import annotations
import os
import sys
from pathlib import Path

release_dir = Path(sys.argv[1]).resolve()
out_tsv = Path(sys.argv[2]).resolve()

rows = []
seen = {}

for canonical in sorted(release_dir.rglob("canonical_flame_param.npz")):
    source_dir = canonical.parent
    base = f"nersemble_{source_dir.name}"
    subject_id = base
    idx = 2
    while subject_id in seen and seen[subject_id] != str(source_dir):
        subject_id = f"{base}_{idx}"
        idx += 1
    seen[subject_id] = str(source_dir)
    rows.append((subject_id, str(source_dir)))

out_tsv.parent.mkdir(parents=True, exist_ok=True)
with open(out_tsv, "w", encoding="utf-8") as f:
    f.write("subject_id\tsource_dir\n")
    for subject_id, source_dir in rows:
        f.write(f"{subject_id}\t{source_dir}\n")

print(f"wrote {out_tsv}")
print(f"subjects = {len(rows)}")
for subject_id, source_dir in rows:
    print(f"{subject_id}\t{source_dir}")
PY
