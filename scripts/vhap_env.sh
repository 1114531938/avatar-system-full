#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
VHAP_RUNS_ROOT="${VHAP_RUNS_ROOT:-$ROOT/VHAP_runs}"
VHAP_REPO="${VHAP_REPO:-$VHAP_RUNS_ROOT/repo}"
VHAP_ENV_NAME="${VHAP_ENV_NAME:-.vhap121}"
VHAP_ENV_ROOT="${VHAP_ENV_ROOT:-$VHAP_RUNS_ROOT/$VHAP_ENV_NAME}"
VHAP_PYTHON="${VHAP_PYTHON:-$VHAP_ENV_ROOT/bin/python}"

detect_cuda_home() {
  if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME:-}" ]]; then
    printf '%s\n' "$CUDA_HOME"
    return 0
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path
    nvcc_path="$(command -v nvcc)"
    printf '%s\n' "$(cd "$(dirname "$nvcc_path")/.." && pwd)"
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda \
    /usr/local/cuda-12.9 \
    /usr/local/cuda-12.8 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-12.3 \
    /usr/local/cuda-12.2 \
    /usr/local/cuda-12.1 \
    /usr/local/cuda-12.0 \
    /opt/cuda \
    /opt/cuda-12.1
  do
    if [[ -f "$candidate/include/cuda_runtime_api.h" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export NLTK_DATA="${NLTK_DATA:-$ROOT/cache/nltk_data}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/cache/pip}"
export PYTHONPATH="$VHAP_REPO:${PYTHONPATH:-}"
export PATH="$VHAP_ENV_ROOT/bin:$ROOT/tools/ffmpeg-git-20240629-amd64-static:${PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if CUDA_HOME_DETECTED="$(detect_cuda_home)"; then
  export CUDA_HOME="$CUDA_HOME_DETECTED"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  export CPATH="$CUDA_HOME/include:${CPATH:-}"
fi

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE" "$NLTK_DATA" "$PIP_CACHE_DIR"

if [[ ! -x "$VHAP_PYTHON" ]]; then
  echo "[vhap_env] VHAP python not found: $VHAP_PYTHON" >&2
  exit 1
fi
if [[ ! -d "$VHAP_REPO" ]]; then
  echo "[vhap_env] VHAP repo not found: $VHAP_REPO" >&2
  exit 1
fi
if [[ -z "${CUDA_HOME:-}" || ! -f "${CUDA_HOME}/include/cuda_runtime_api.h" ]]; then
  echo "[vhap_env] CUDA toolkit not found. Set CUDA_HOME before running VHAP tracking." >&2
  exit 1
fi
