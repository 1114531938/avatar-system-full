#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <subject_id> [--source PATH] [--model PATH] [--fast-30k|--quality-200k] [extra GaussianAvatars args...]"
  exit 2
fi

SUBJECT_ID="$1"
shift

SOURCE_PATH=""
MODEL_PATH=""
FAST_30K=0
QUALITY_200K=0
declare -a EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE_PATH="$2"; shift 2 ;;
    --model) MODEL_PATH="$2"; shift 2 ;;
    --fast-30k) FAST_30K=1; shift ;;
    --quality-200k) QUALITY_200K=1; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$FAST_30K" == "1" && "$QUALITY_200K" == "1" ]]; then
  echo "--fast-30k and --quality-200k cannot be used together" >&2
  exit 2
fi

SUBJECT_ROOT="$ROOT/data/subjects/$SUBJECT_ID"
SOURCE_PATH="${SOURCE_PATH:-$SUBJECT_ROOT/gaussian_source}"
if [[ -z "$MODEL_PATH" ]]; then
  if [[ "$FAST_30K" == "1" ]]; then
    MODEL_PATH="$SUBJECT_ROOT/gaussian_train_30k"
  elif [[ "$QUALITY_200K" == "1" ]]; then
    MODEL_PATH="$SUBJECT_ROOT/gaussian_train_200k"
  else
    MODEL_PATH="$SUBJECT_ROOT/gaussian_train"
  fi
fi
GAUSS_ROOT="$ROOT/GSavatar_runs/GaussianAvatars"
GAUSS_PY="$GAUSS_ROOT/.GSavatar_glibc/bin/python"

if [[ ! -x "$GAUSS_PY" ]]; then
  echo "GaussianAvatars python not found: $GAUSS_PY" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_PATH/canonical_flame_param.npz" ]]; then
  echo "Expected Gaussian source with canonical_flame_param.npz: $SOURCE_PATH" >&2
  exit 1
fi

mkdir -p "$MODEL_PATH"
cd "$GAUSS_ROOT"

CMD=(
  "$GAUSS_PY" train.py
  -s "$SOURCE_PATH"
  -m "$MODEL_PATH"
  --eval
  --bind_to_mesh
  --white_background
)
if [[ "$FAST_30K" == "1" ]]; then
  CMD+=(
    --iterations 30000
    --interval 5000
    --test_iterations 5000 10000 15000 20000 25000 30000
    --save_iterations 5000 10000 15000 20000 25000 30000
    --checkpoint_iterations 10000 20000 30000
    --densify_until_iter 30000
    --opacity_reset_interval 3000
  )
elif [[ "$QUALITY_200K" == "1" ]]; then
  CMD+=(
    --iterations 200000
    --interval 20000
    --test_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000
    --save_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000
    --checkpoint_iterations 20000 40000 60000 80000 100000 120000 140000 160000 180000 200000
    --position_lr_max_steps 200000
    --densify_until_iter 200000
    --opacity_reset_interval 20000
  )
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf '[train_gaussian_subject] %q ' "${CMD[@]}"; echo
exec "${CMD[@]}"
