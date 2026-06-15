#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
GAUSS_ROOT="$ROOT/GSavatar_runs/GaussianAvatars"
GAUSS_PY="$GAUSS_ROOT/.GSavatar_glibc/bin/python"
EXPORTER="$ROOT/tools/avatar_agent/export_gaussian_video.py"
CONTAINER="$ROOT/containers/gaussianav_jammy"
FFMPEG="$ROOT/tools/ffmpeg-git-20240629-amd64-static/ffmpeg"

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <point_cloud.ply> <motion.npz> <audio.wav> <output_dir> [camera.json]"
  exit 2
fi

POINT_PATH="$(realpath "$1")"
MOTION_PATH="$(realpath "$2")"
AUDIO_PATH="$(realpath "$3")"
OUTPUT_DIR="$(realpath -m "$4")"
CAMERA_JSON="${5:-}"

for path in "$POINT_PATH" "$MOTION_PATH" "$AUDIO_PATH"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input file: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$(dirname "$POINT_PATH")/micro_expression.pth" ]]; then
  echo "Missing trained micro-expression weights next to point cloud:" >&2
  echo "  $(dirname "$POINT_PATH")/micro_expression.pth" >&2
  echo "Train or export an enhanced avatar before visual A/B comparison." >&2
  exit 1
fi
if [[ -n "$CAMERA_JSON" ]]; then
  CAMERA_JSON="$(realpath "$CAMERA_JSON")"
  if [[ ! -f "$CAMERA_JSON" ]]; then
    echo "Missing camera JSON: $CAMERA_JSON" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  --gaussian_root "$GAUSS_ROOT"
  --point_path "$POINT_PATH"
  --motion_path "$MOTION_PATH"
  --audio_path "$AUDIO_PATH"
  --audio_filter ""
  --render_mode gaussian
  --fps 25
  --width 550
  --height 802
  --ffmpeg "$FFMPEG"
)
if [[ -n "$CAMERA_JSON" ]]; then
  COMMON_ARGS+=(--camera_json "$CAMERA_JSON")
fi

printf -v BASE_CMD '%q ' "$GAUSS_PY" "$EXPORTER" "${COMMON_ARGS[@]}" \
  --disable_micro_expression \
  --out_video "$OUTPUT_DIR/baseline.mp4" \
  --frames_dir "$OUTPUT_DIR/baseline_frames"
printf -v ENHANCED_CMD '%q ' "$GAUSS_PY" "$EXPORTER" "${COMMON_ARGS[@]}" \
  --out_video "$OUTPUT_DIR/enhanced.mp4" \
  --frames_dir "$OUTPUT_DIR/enhanced_frames"

INNER_CMD="
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd '$GAUSS_ROOT'
$BASE_CMD
$ENHANCED_CMD
"

apptainer exec --fakeroot --writable --nv \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$CONTAINER" \
  bash -lc "$INNER_CMD"

if "$FFMPEG" -hide_banner -filters 2>&1 | grep -q '[[:space:]]drawtext[[:space:]]'; then
  STACK_FILTER="[0:v]drawtext=text='Baseline':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[left];[1:v]drawtext=text='Micro expression':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[right];[left][right]hstack=inputs=2[v]"
else
  echo "ffmpeg drawtext filter is unavailable; writing an unlabeled side-by-side video." >&2
  STACK_FILTER="[0:v][1:v]hstack=inputs=2[v]"
fi

"$FFMPEG" -y \
  -i "$OUTPUT_DIR/baseline.mp4" \
  -i "$OUTPUT_DIR/enhanced.mp4" \
  -filter_complex "$STACK_FILTER" \
  -map "[v]" -map 0:a? -c:v libx264 -pix_fmt yuv420p -c:a copy \
  "$OUTPUT_DIR/side_by_side.mp4"

echo "Baseline:     $OUTPUT_DIR/baseline.mp4"
echo "Enhanced:     $OUTPUT_DIR/enhanced.mp4"
echo "Side-by-side: $OUTPUT_DIR/side_by_side.mp4"
