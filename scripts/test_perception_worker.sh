#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
URL="${PERCEPTION_WORKER_URL:-http://127.0.0.1:8791}"
WAV="${1:-$ROOT/perception_layer/data/demo_wavs/sample_dialog_02.wav}"
OUT_DIR="${2:-$ROOT/outputs/perception_worker_smoke}"

mkdir -p "$OUT_DIR/perception" "$OUT_DIR/task1"

curl -fsS "$URL/health"
echo

payload=$(printf '{"wav":"%s","perception_out":"%s","task1_out":"%s","model":"small","language":"Chinese","speaker_id":"user","ser_model":"iic/emotion2vec_plus_seed","no_llm":true}' \
  "$WAV" "$OUT_DIR/perception" "$OUT_DIR/task1")

curl -fsS \
  -H 'Content-Type: application/json' \
  -d "$payload" \
  "$URL/run"
echo
