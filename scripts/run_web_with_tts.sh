#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
TTS_WORKER_HOST="${TTS_WORKER_HOST:-127.0.0.1}"
TTS_WORKER_PORT="${TTS_WORKER_PORT:-8788}"
TTS_WORKER_URL="http://${TTS_WORKER_HOST}:${TTS_WORKER_PORT}"
LOG_DIR="$ROOT/outputs/service_logs"
TTS_LOG="$LOG_DIR/tts_worker.log"

mkdir -p "$LOG_DIR"

tts_pid=""

cleanup() {
  if [[ -n "${tts_pid}" ]] && kill -0 "$tts_pid" 2>/dev/null; then
    echo "[service] stopping TTS worker pid=$tts_pid"
    kill "$tts_pid" 2>/dev/null || true
    wait "$tts_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -fsS --max-time 2 "$TTS_WORKER_URL/health" >/dev/null 2>&1; then
  echo "[service] TTS worker already running at $TTS_WORKER_URL"
else
  echo "[service] starting TTS worker at $TTS_WORKER_URL"
  TTS_WORKER_HOST="$TTS_WORKER_HOST" TTS_WORKER_PORT="$TTS_WORKER_PORT" \
    bash "$ROOT/scripts/run_tts_worker.sh" >"$TTS_LOG" 2>&1 &
  tts_pid="$!"

  for _ in $(seq 1 90); do
    if curl -fsS --max-time 2 "$TTS_WORKER_URL/health" >/dev/null 2>&1; then
      echo "[service] TTS worker ready pid=$tts_pid"
      break
    fi
    if ! kill -0 "$tts_pid" 2>/dev/null; then
      echo "[service] TTS worker exited while starting. Log:"
      tail -80 "$TTS_LOG" || true
      exit 1
    fi
    sleep 1
  done

  if ! curl -fsS --max-time 2 "$TTS_WORKER_URL/health" >/dev/null 2>&1; then
    echo "[service] TTS worker did not become ready in time. Log:"
    tail -80 "$TTS_LOG" || true
    exit 1
  fi
fi

echo "[service] starting web server on ${HOST:-0.0.0.0}:${PORT:-7861}"
bash "$ROOT/scripts/run_web.sh"
