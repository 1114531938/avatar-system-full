#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
HOST="${PERCEPTION_WORKER_HOST:-127.0.0.1}"
PORT="${PERCEPTION_WORKER_PORT:-8791}"
MODEL="${PERCEPTION_WORKER_MODEL:-small}"
SER_MODEL="${PERCEPTION_WORKER_SER_MODEL:-iic/emotion2vec_plus_seed}"

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"

cmd="
cd '$ROOT/perception_layer'
'$ROOT/perception_layer/.perception/bin/python' scripts/perception_worker.py \
  --host '$HOST' \
  --port '$PORT' \
  --model '$MODEL' \
  --ser_model '$SER_MODEL'
"

exec apptainer exec --fakeroot --writable --nv \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$ROOT/containers/gaussianav_jammy" \
  bash -lc "$cmd"
