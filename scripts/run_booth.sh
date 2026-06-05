#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7862}"

export HF_HOME="${HF_HOME:-$ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/cache/xdg}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/cache/modelscope}"
export NLTK_DATA="${NLTK_DATA:-$ROOT/cache/nltk_data}"
export BOOTH_DEFAULT_ROUTE="${BOOTH_DEFAULT_ROUTE:-1}"

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE" "$NLTK_DATA" "$ROOT/outputs"

cd "$ROOT"
exec "$ROOT/web_app/.web_venv/bin/python" -m uvicorn web_app.server:app --host "$HOST" --port "$PORT"
