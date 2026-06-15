#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7861}"
RUN_DIR="$ROOT/outputs/service_logs"
PID_FILE="$RUN_DIR/video_input_web.pid"
LOG_FILE="$RUN_DIR/video_input_web.log"
URL="http://127.0.0.1:${PORT}/booth"

mkdir -p "$RUN_DIR"

is_running() {
  [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

wait_for_url() {
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "$URL" >/dev/null 2>&1; then
      return 0
    fi
    if ! is_running; then
      echo "[video-input-test] web exited while starting. Log:"
      tail -80 "$LOG_FILE" || true
      return 1
    fi
    sleep 1
  done
  echo "[video-input-test] web did not become ready. Log:"
  tail -80 "$LOG_FILE" || true
  return 1
}

start() {
  if is_running; then
    echo "[video-input-test] already running pid=$(cat "$PID_FILE") url=$URL"
    return 0
  fi
  echo "[video-input-test] starting web-only booth input test on ${HOST}:${PORT}"
  HOST="$HOST" PORT="$PORT" WEB_HOME=booth BOOTH_INPUT_TEST_ONLY=1 \
    nohup bash "$ROOT/scripts/run_web.sh" >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  wait_for_url
  echo "[video-input-test] ready pid=$(cat "$PID_FILE")"
  echo "[video-input-test] open http://localhost:${PORT}/booth"
}

stop() {
  if is_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    echo "[video-input-test] stopping web pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[video-input-test] force stopping web pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$PID_FILE"
}

status() {
  if is_running; then
    echo "[video-input-test] running pid=$(cat "$PID_FILE") url=$URL"
  else
    echo "[video-input-test] stopped"
  fi
}

case "${1:-start}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  logs)
    tail -120 "$LOG_FILE" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 2
    ;;
esac
