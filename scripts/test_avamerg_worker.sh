#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
URL="${AVAMERG_WORKER_URL:-http://127.0.0.1:8789}"
INPUT_JSON="${1:-$ROOT/perception_layer/results/task1/web_recording_1776930628367_20260423_155032_task1_input.json}"
OUT_DIR="$ROOT/outputs/avamerg_worker_smoke"
OUT_JSON="$OUT_DIR/task1_reply.json"

mkdir -p "$OUT_DIR"

if ! curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; then
  echo "[smoke] AvaMERG worker is not reachable at $URL"
  echo "[smoke] Start it first on GN-A40-043:"
  echo "        bash $ROOT/scripts/avatar_service.sh start-avamerg"
  exit 1
fi

echo "[smoke] health:"
curl -fsS "$URL/health"
echo

echo "[smoke] infer input: $INPUT_JSON"
rm -f "$OUT_JSON"
curl -fsS -X POST "$URL/infer" \
  -H 'Content-Type: application/json' \
  -d "{\"input_json\":\"$INPUT_JSON\",\"out_json\":\"$OUT_JSON\"}" \
  > "$OUT_DIR/response.json"

echo "[smoke] response:"
python -m json.tool "$OUT_DIR/response.json" | sed -n '1,80p'

if [[ ! -s "$OUT_JSON" ]]; then
  echo "[smoke] missing output: $OUT_JSON"
  exit 1
fi

echo "[smoke] output:"
python -m json.tool "$OUT_JSON" | sed -n '1,120p'
echo "[smoke] ok: $OUT_JSON"
