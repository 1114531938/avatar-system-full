#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
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

AGENT_PYTHON="${AGENT_PYTHON:-$ROOT/wav_to_flame/DEEPTalk_runs/.deeptalk39/bin/python}"
if [[ ! -x "$AGENT_PYTHON" ]]; then
  echo "[run_agent] AGENT_PYTHON is not executable: $AGENT_PYTHON" >&2
  echo "[run_agent] Set AGENT_PYTHON=/path/to/python with PyYAML installed." >&2
  exit 1
fi

"$AGENT_PYTHON" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("yaml",) if importlib.util.find_spec(name) is None]
if missing:
    print(
        "[run_agent] missing Python modules in {}: {}".format(
            sys.executable, ", ".join(missing)
        ),
        file=sys.stderr,
    )
    print("[run_agent] Install PyYAML or set AGENT_PYTHON to a compatible project Python.", file=sys.stderr)
    raise SystemExit(1)
PY

cd "$ROOT/tools/avatar_agent"
"$AGENT_PYTHON" run_avatar_agent.py \
  --input_wav "$INPUT_WAV" \
  --avatar_id "$AVATAR_ID" \
  --config "$ROOT/tools/avatar_agent/pipeline_config.yaml" \
  "$@"
