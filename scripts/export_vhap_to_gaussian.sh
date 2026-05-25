#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
source "$ROOT/scripts/vhap_env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subject_id> [--src-folder PATH] [--tgt-folder PATH]"
  exit 2
fi

SUBJECT_ID="$1"
shift

SRC_FOLDER=""
TGT_FOLDER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src-folder) SRC_FOLDER="$2"; shift 2 ;;
    --tgt-folder) TGT_FOLDER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

SUBJECT_ROOT="$ROOT/data/subjects/$SUBJECT_ID"
SRC_FOLDER="${SRC_FOLDER:-$SUBJECT_ROOT/vhap/tracking}"
TGT_FOLDER="${TGT_FOLDER:-$SUBJECT_ROOT/gaussian_source}"

if [[ -d "$SRC_FOLDER" ]]; then
  mapfile -t TRACK_CANDIDATES < <(find "$SRC_FOLDER" -maxdepth 1 -mindepth 1 -type d | sort)
  if [[ ${#TRACK_CANDIDATES[@]} -eq 1 ]]; then
    SRC_FOLDER="${TRACK_CANDIDATES[0]}"
  fi
fi

mkdir -p "$TGT_FOLDER"
cd "$VHAP_REPO"
exec "$VHAP_PYTHON" vhap/export_as_nerf_dataset.py \
  --src_folder "$SRC_FOLDER" \
  --tgt_folder "$TGT_FOLDER" \
  --background-color white
