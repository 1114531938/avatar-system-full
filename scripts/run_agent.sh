#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
INPUT_WAV="$ROOT/perception_layer/data/demo_wavs/sample_dialog_02.wav"
AVATAR_ID="306"

if [[ $# -gt 0 && "$1" != --* ]]; then
  INPUT_WAV="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != --* ]]; then
  AVATAR_ID="$1"
  shift
fi

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export NLTK_DATA="${NLTK_DATA:-$ROOT/cache/nltk_data}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE" "$NLTK_DATA"

cd "$ROOT/tools/avatar_agent"
"${AGENT_PYTHON:-python}" run_avatar_agent.py \
  --input_wav "$INPUT_WAV" \
  --avatar_id "$AVATAR_ID" \
  --config "$ROOT/tools/avatar_agent/pipeline_config.yaml" \
  "$@"
