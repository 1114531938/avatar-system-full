#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
source "$ROOT/scripts/vhap_env.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subject_id> [--mode monocular|nersemble] [--input PATH] [--sequence NAME] [--downsample 2 4] [--matting METHOD] [--track-only] [--export-only] [--preprocess-only]"
  exit 2
fi

SUBJECT_ID="$1"
shift

MODE="monocular"
INPUT_PATH=""
SEQUENCE="$SUBJECT_ID"
TRACK_ONLY=0
EXPORT_ONLY=0
PREPROCESS_ONLY=0
MATTING_METHOD=""
declare -a DOWNSAMPLE_SCALES=()
declare -a EXTRA_TRACK_ARGS=()
declare -a EXTRA_EXPORT_ARGS=()
LAST_DOWNSAMPLE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"; shift 2 ;;
    --input)
      INPUT_PATH="$2"; shift 2 ;;
    --sequence)
      SEQUENCE="$2"; shift 2 ;;
    --downsample)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DOWNSAMPLE_SCALES+=("$1")
        shift
      done
      ;;
    --matting)
      MATTING_METHOD="$2"; shift 2 ;;
    --track-only)
      TRACK_ONLY=1; shift ;;
    --export-only)
      EXPORT_ONLY=1; shift ;;
    --preprocess-only)
      PREPROCESS_ONLY=1; shift ;;
    --track-arg)
      EXTRA_TRACK_ARGS+=("$2"); shift 2 ;;
    --export-arg)
      EXTRA_EXPORT_ARGS+=("$2"); shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2 ;;
  esac
done

SUBJECT_ROOT="$ROOT/data/subjects/$SUBJECT_ID"
RAW_ROOT="$SUBJECT_ROOT/raw"
VHAP_ROOT="$SUBJECT_ROOT/vhap"
TRACK_ROOT="$VHAP_ROOT/tracking"
EXPORT_ROOT="$SUBJECT_ROOT/gaussian_source"

mkdir -p "$RAW_ROOT" "$TRACK_ROOT" "$EXPORT_ROOT"
cd "$VHAP_REPO"

if [[ "$MODE" == "monocular" ]]; then
  INPUT_PATH="${INPUT_PATH:-$RAW_ROOT/${SEQUENCE}.mp4}"
  [[ -n "$MATTING_METHOD" ]] || MATTING_METHOD="robust_video_matting"
  TRACK_OUTPUT="$TRACK_ROOT/${SEQUENCE}_whiteBg_staticOffset"

  PREPROCESS_CMD=("$VHAP_PYTHON" vhap/preprocess_video.py --input "$INPUT_PATH" --matting_method "$MATTING_METHOD")
  if [[ ${#DOWNSAMPLE_SCALES[@]} -gt 0 ]]; then
    PREPROCESS_CMD+=(--downsample_scales "${DOWNSAMPLE_SCALES[@]}")
  fi

  TRACK_CMD=(
    "$VHAP_PYTHON" vhap/track.py
    --data.root_folder "$RAW_ROOT"
    --exp.output_folder "$TRACK_OUTPUT"
    --data.sequence "$SEQUENCE"
  )
  if [[ ${#DOWNSAMPLE_SCALES[@]} -gt 0 ]]; then
    LAST_DOWNSAMPLE="${DOWNSAMPLE_SCALES[$((${#DOWNSAMPLE_SCALES[@]} - 1))]}"
    TRACK_CMD+=(--data.n_downsample_rgb "$LAST_DOWNSAMPLE")
  fi
  if [[ ${#EXTRA_TRACK_ARGS[@]} -gt 0 ]]; then
    TRACK_CMD+=("${EXTRA_TRACK_ARGS[@]}")
  fi
elif [[ "$MODE" == "nersemble" ]]; then
  INPUT_PATH="${INPUT_PATH:-$RAW_ROOT/$SEQUENCE}"
  [[ -n "$MATTING_METHOD" ]] || MATTING_METHOD="background_matting_v2"
  TRACK_OUTPUT="$TRACK_ROOT/${SUBJECT_ID}_${SEQUENCE}_v16_DS4_wBg_staticOffset"

  PREPROCESS_CMD=("$VHAP_PYTHON" vhap/preprocess_video.py --input "$INPUT_PATH"* --matting_method "$MATTING_METHOD")
  if [[ ${#DOWNSAMPLE_SCALES[@]} -eq 0 ]]; then
    DOWNSAMPLE_SCALES=(2 4)
  fi
  PREPROCESS_CMD+=(--downsample_scales "${DOWNSAMPLE_SCALES[@]}")
  LAST_DOWNSAMPLE="${DOWNSAMPLE_SCALES[$((${#DOWNSAMPLE_SCALES[@]} - 1))]}"

  TRACK_CMD=(
    "$VHAP_PYTHON" vhap/track_nersemble.py
    --data.root_folder "$RAW_ROOT"
    --exp.output_folder "$TRACK_OUTPUT"
    --data.subject "$SUBJECT_ID"
    --data.sequence "$SEQUENCE"
    --data.n_downsample_rgb "$LAST_DOWNSAMPLE"
  )
  if [[ ${#EXTRA_TRACK_ARGS[@]} -gt 0 ]]; then
    TRACK_CMD+=("${EXTRA_TRACK_ARGS[@]}")
  fi
else
  echo "Unsupported mode: $MODE" >&2
  exit 2
fi

EXPORT_CMD=(
  "$VHAP_PYTHON" vhap/export_as_nerf_dataset.py
  --src_folder "$TRACK_OUTPUT"
  --tgt_folder "$EXPORT_ROOT"
  --background-color white
)
if [[ ${#EXTRA_EXPORT_ARGS[@]} -gt 0 ]]; then
  EXPORT_CMD+=("${EXTRA_EXPORT_ARGS[@]}")
fi

echo "[run_vhap_subject] subject=$SUBJECT_ID mode=$MODE"
echo "[run_vhap_subject] repo=$VHAP_REPO"
echo "[run_vhap_subject] python=$VHAP_PYTHON"

if [[ "$EXPORT_ONLY" == "1" ]]; then
  printf '[run_vhap_subject] export: %q ' "${EXPORT_CMD[@]}"; echo
  exec "${EXPORT_CMD[@]}"
fi
if [[ "$TRACK_ONLY" == "1" ]]; then
  printf '[run_vhap_subject] track: %q ' "${TRACK_CMD[@]}"; echo
  exec "${TRACK_CMD[@]}"
fi
if [[ "$PREPROCESS_ONLY" == "1" ]]; then
  printf '[run_vhap_subject] preprocess: %q ' "${PREPROCESS_CMD[@]}"; echo
  exec "${PREPROCESS_CMD[@]}"
fi

printf '[run_vhap_subject] preprocess: %q ' "${PREPROCESS_CMD[@]}"; echo
"${PREPROCESS_CMD[@]}"
printf '[run_vhap_subject] track: %q ' "${TRACK_CMD[@]}"; echo
"${TRACK_CMD[@]}"
printf '[run_vhap_subject] export: %q ' "${EXPORT_CMD[@]}"; echo
"${EXPORT_CMD[@]}"
