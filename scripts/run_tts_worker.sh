#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
HOST="${TTS_WORKER_HOST:-127.0.0.1}"
PORT="${TTS_WORKER_PORT:-8788}"

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
cd '$ROOT/EmotiVoice_runs/repo'
'$ROOT/EmotiVoice_runs/repo/.EmotiVoice/bin/python' tts_worker.py \
  --host '$HOST' \
  --port '$PORT' \
  --logdir prompt_tts_open_source_joint \
  --config_folder config/joint \
  --checkpoint g_00140000
"

APPTAINER_FLAGS="${APPTAINER_FLAGS:---nv}"

exec apptainer exec $APPTAINER_FLAGS \
  -B /scratch:/scratch,/home/svu:/home/svu \
  "$ROOT/containers/gaussianav_jammy" \
  bash -lc "$cmd"
