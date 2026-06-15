#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
HOST="${GAUSSIAN_RENDER_WORKER_HOST:-127.0.0.1}"
PORT="${GAUSSIAN_RENDER_WORKER_PORT:-8792}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cmd="
export PYTHONPATH='$ROOT/GSavatar_runs/GaussianAvatars:$ROOT/tools/avatar_agent':\$PYTHONPATH
cd '$ROOT/GSavatar_runs/GaussianAvatars'
'$ROOT/GSavatar_runs/GaussianAvatars/.GSavatar_glibc/bin/python' gaussian_render_worker.py \
  --host '$HOST' \
  --port '$PORT'
"

APPTAINER_FLAGS="${APPTAINER_FLAGS:---nv}"

exec apptainer exec $APPTAINER_FLAGS \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$ROOT/containers/gaussianav_jammy" \
  bash -lc "$cmd"
