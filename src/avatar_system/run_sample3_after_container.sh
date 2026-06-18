#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x /usr/bin/python3.10 ]]; then
  echo "python3.10 was not found. Enter the gaussianav_jammy container first, then rerun this script." >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before running the agent." >&2
  exit 1
fi

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"

export HF_HOME="$ROOT/runtime/cache/hf"
export XDG_CACHE_HOME="$ROOT/runtime/cache/xdg"
export MODELSCOPE_CACHE="$ROOT/runtime/cache/modelscope"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b:free}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

cd "$ROOT"
/usr/bin/python3.10 -m avatar_system.pipeline.cli \
  --input_wav "$ROOT/integrations/perception/data/demo_wavs/sample3.wav" \
  --avatar_id 306 \
  --config "$ROOT/src/avatar_system/pipeline_config.yaml" \
  "$@"
