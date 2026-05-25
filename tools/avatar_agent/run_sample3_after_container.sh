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

export HF_HOME=/scratch/e1554543/avatar_system_full/cache/hf
export XDG_CACHE_HOME=/scratch/e1554543/avatar_system_full/cache/xdg
export MODELSCOPE_CACHE=/scratch/e1554543/avatar_system_full/cache/modelscope
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://openrouter.ai/api/v1}"
export LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b:free}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE"

cd /scratch/e1554543/avatar_system_full/tools/avatar_agent
/usr/bin/python3.10 run_avatar_agent.py \
  --input_wav /scratch/e1554543/avatar_system_full/perception_layer/data/demo_wavs/sample3.wav \
  --avatar_id 306 \
  --config /scratch/e1554543/avatar_system_full/tools/avatar_agent/pipeline_config.yaml \
  "$@"
