#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
HOST="${DEEPTALK_WORKER_HOST:-127.0.0.1}"
PORT="${DEEPTALK_WORKER_PORT:-8790}"

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PATH="$ROOT/tools/ffmpeg-git-20240629-amd64-static:$PATH"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"

cd "$ROOT/wav_to_flame/DEEPTalk_runs/repos/DEEPTalk/DEEPTalk"
exec "$ROOT/wav_to_flame/DEEPTalk_runs/.deeptalk39/bin/python" deeptalk_worker.py \
  --host "$HOST" \
  --port "$PORT"
