#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
URL="${DEEPTALK_WORKER_URL:-http://127.0.0.1:8790}"
AUDIO_PATH="${1:-$ROOT/outputs/web_recording_1776933053788_20260423_163059/outputs/reply.wav}"
OUT_DIR="$ROOT/outputs/deeptalk_worker_smoke"
OUT_NPY="$OUT_DIR/deeptalk.npy"

mkdir -p "$OUT_DIR"

if ! curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; then
  echo "[smoke] DEEPTalk worker is not reachable at $URL"
  echo "[smoke] Start it first on GN-A40-043:"
  echo "        bash $ROOT/scripts/avatar_service.sh start-deeptalk"
  exit 1
fi

echo "[smoke] health:"
curl -fsS "$URL/health"
echo

echo "[smoke] infer audio: $AUDIO_PATH"
rm -f "$OUT_NPY"
curl -fsS -X POST "$URL/infer" \
  -H 'Content-Type: application/json' \
  -d "{\"audio_path\":\"$AUDIO_PATH\",\"output_npy\":\"$OUT_NPY\"}" \
  > "$OUT_DIR/response.json"

echo "[smoke] response:"
python -m json.tool "$OUT_DIR/response.json" | sed -n '1,120p'

if [[ ! -s "$OUT_NPY" ]]; then
  echo "[smoke] missing output: $OUT_NPY"
  exit 1
fi

python - <<PY
import numpy as np
path = "$OUT_NPY"
arr = np.load(path)
print("[smoke] npy:", path, "shape=", arr.shape, "dtype=", arr.dtype)
PY
echo "[smoke] ok: $OUT_NPY"
