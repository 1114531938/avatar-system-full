#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
DATA_ROOT="$ROOT/GSavatar_runs/GaussianAvatars/datasets/nersemble_preprocessed"
ZIP_PATH="$DATA_ROOT/release.zip"
OUT_DIR="$DATA_ROOT/release"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "release.zip not found: $ZIP_PATH" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[unpack_nersemble_release] testing zip integrity..."
unzip -tq "$ZIP_PATH" >/dev/null

echo "[unpack_nersemble_release] extracting to $OUT_DIR"
unzip -q -o "$ZIP_PATH" -d "$OUT_DIR"

echo "[unpack_nersemble_release] done"
find "$OUT_DIR" -maxdepth 2 -mindepth 1 | sed -n '1,120p'
