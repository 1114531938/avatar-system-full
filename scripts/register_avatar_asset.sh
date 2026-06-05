#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <subject_id> <avatar_id> [--model PATH] [--point-path PATH] [--canonical PATH] [--tracked PATH]"
  exit 2
fi

SUBJECT_ID="$1"
AVATAR_ID="$2"
shift 2

POINT_PATH=""
MODEL_PATH=""
CANONICAL_PATH=""
TRACKED_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2 ;;
    --point-path) POINT_PATH="$2"; shift 2 ;;
    --canonical) CANONICAL_PATH="$2"; shift 2 ;;
    --tracked) TRACKED_PATH="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

SUBJECT_ROOT="$ROOT/data/subjects/$SUBJECT_ID"
ASSET_ROOT="$ROOT/GSavatar_runs/GaussianAvatars/media/$AVATAR_ID"
MODEL_PATH="${MODEL_PATH:-$SUBJECT_ROOT/gaussian_train}"

if [[ -z "$POINT_PATH" ]]; then
  POINT_PATH="$(find "$MODEL_PATH/point_cloud" -type f -name 'point_cloud.ply' | sort | tail -n 1 || true)"
fi
CANONICAL_PATH="${CANONICAL_PATH:-$SUBJECT_ROOT/gaussian_source/canonical_flame_param.npz}"

if [[ -z "$TRACKED_PATH" && -d "$SUBJECT_ROOT/vhap/tracking" ]]; then
  TRACKED_PATH="$(find "$SUBJECT_ROOT/vhap/tracking" -type f -name 'tracked_flame_params*.npz' | sort | tail -n 1 || true)"
fi

if [[ -z "$POINT_PATH" || ! -f "$POINT_PATH" ]]; then
  echo "point_cloud.ply not found" >&2
  exit 1
fi
if [[ ! -f "$CANONICAL_PATH" ]]; then
  echo "canonical_flame_param.npz not found: $CANONICAL_PATH" >&2
  exit 1
fi

mkdir -p "$ASSET_ROOT"
cp "$POINT_PATH" "$ASSET_ROOT/point_cloud.ply"

POINT_DIR="$(dirname "$POINT_PATH")"
MICRO_WEIGHTS="$POINT_DIR/micro_expression.pth"
MICRO_CONFIG="$POINT_DIR/micro_expression_config.json"
if [[ -f "$MICRO_WEIGHTS" && -f "$MICRO_CONFIG" ]]; then
  cp "$MICRO_WEIGHTS" "$ASSET_ROOT/micro_expression.pth"
  cp "$MICRO_CONFIG" "$ASSET_ROOT/micro_expression_config.json"
else
  rm -f "$ASSET_ROOT/micro_expression.pth" "$ASSET_ROOT/micro_expression_config.json"
fi

CMD=(
  python "$ROOT/tools/avatar_agent/tools/build_template_from_vhap.py"
  --canonical "$CANONICAL_PATH"
  --out "$ASSET_ROOT/flame_param.npz"
)
if [[ -n "$TRACKED_PATH" ]]; then
  CMD+=(--tracked "$TRACKED_PATH")
fi
"${CMD[@]}"

echo "[register_avatar_asset] registered asset:"
echo "  avatar_id:    $AVATAR_ID"
echo "  point_cloud:  $ASSET_ROOT/point_cloud.ply"
echo "  flame_param:  $ASSET_ROOT/flame_param.npz"
if [[ -f "$ASSET_ROOT/micro_expression.pth" ]]; then
  echo "  micro model:  $ASSET_ROOT/micro_expression.pth"
  echo "  micro config: $ASSET_ROOT/micro_expression_config.json"
fi
