#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
HOST="${AVAMERG_WORKER_HOST:-127.0.0.1}"
PORT="${AVAMERG_WORKER_PORT:-8789}"

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
export PYTHONPATH='$ROOT/AvaMERG_runs/AvaMERG-Pipeline/merg_code:$ROOT/AvaMERG_runs/AvaMERG-Pipeline':\$PYTHONPATH
cd '$ROOT/AvaMERG_runs/AvaMERG-Pipeline'
'$ROOT/AvaMERG_runs/AvaMERG-Pipeline/.avamerg38/bin/python' avamerg_worker.py \
  --host '$HOST' \
  --port '$PORT'
"

exec apptainer exec --fakeroot --writable --nv \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$ROOT/containers/gaussianav_jammy" \
  bash -lc "$cmd"
