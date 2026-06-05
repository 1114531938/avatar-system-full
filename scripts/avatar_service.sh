#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7861}"
TTS_WORKER_HOST="${TTS_WORKER_HOST:-127.0.0.1}"
TTS_WORKER_PORT="${TTS_WORKER_PORT:-8788}"
TTS_WORKER_URL="http://${TTS_WORKER_HOST}:${TTS_WORKER_PORT}"
TTS_WORKER_START_TIMEOUT="${TTS_WORKER_START_TIMEOUT:-240}"
AVAMERG_WORKER_HOST="${AVAMERG_WORKER_HOST:-127.0.0.1}"
AVAMERG_WORKER_PORT="${AVAMERG_WORKER_PORT:-8789}"
AVAMERG_WORKER_URL="http://${AVAMERG_WORKER_HOST}:${AVAMERG_WORKER_PORT}"
START_AVAMERG_WORKER="${START_AVAMERG_WORKER:-1}"
DEEPTALK_WORKER_HOST="${DEEPTALK_WORKER_HOST:-127.0.0.1}"
DEEPTALK_WORKER_PORT="${DEEPTALK_WORKER_PORT:-8790}"
DEEPTALK_WORKER_URL="http://${DEEPTALK_WORKER_HOST}:${DEEPTALK_WORKER_PORT}"
START_DEEPTALK_WORKER="${START_DEEPTALK_WORKER:-1}"
PERCEPTION_WORKER_HOST="${PERCEPTION_WORKER_HOST:-127.0.0.1}"
PERCEPTION_WORKER_PORT="${PERCEPTION_WORKER_PORT:-8791}"
PERCEPTION_WORKER_URL="http://${PERCEPTION_WORKER_HOST}:${PERCEPTION_WORKER_PORT}"
START_PERCEPTION_WORKER="${START_PERCEPTION_WORKER:-1}"
GAUSSIAN_RENDER_WORKER_HOST="${GAUSSIAN_RENDER_WORKER_HOST:-127.0.0.1}"
GAUSSIAN_RENDER_WORKER_PORT="${GAUSSIAN_RENDER_WORKER_PORT:-8792}"
GAUSSIAN_RENDER_WORKER_URL="http://${GAUSSIAN_RENDER_WORKER_HOST}:${GAUSSIAN_RENDER_WORKER_PORT}"
START_GAUSSIAN_RENDER_WORKER="${START_GAUSSIAN_RENDER_WORKER:-1}"
RUN_DIR="$ROOT/outputs/service_logs"
WEB_SCRIPT="${WEB_SCRIPT:-$ROOT/scripts/run_web.sh}"
WEB_LOG="${WEB_LOG:-$RUN_DIR/web.log}"
TTS_LOG="$RUN_DIR/tts_worker.log"
AVAMERG_LOG="$RUN_DIR/avamerg_worker.log"
DEEPTALK_LOG="$RUN_DIR/deeptalk_worker.log"
PERCEPTION_LOG="$RUN_DIR/perception_worker.log"
GAUSSIAN_RENDER_LOG="$RUN_DIR/gaussian_render_worker.log"
WEB_PID_FILE="${WEB_PID_FILE:-$RUN_DIR/web.pid}"
TTS_PID_FILE="$RUN_DIR/tts_worker.pid"
AVAMERG_PID_FILE="$RUN_DIR/avamerg_worker.pid"
DEEPTALK_PID_FILE="$RUN_DIR/deeptalk_worker.pid"
PERCEPTION_PID_FILE="$RUN_DIR/perception_worker.pid"
GAUSSIAN_RENDER_PID_FILE="$RUN_DIR/gaussian_render_worker.pid"

mkdir -p "$RUN_DIR"

is_running() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

port_listen_pid() {
  local port="$1"
  ss -ltnpH "sport = :$port" 2>/dev/null \
    | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
    | head -n 1
}

port_has_listener() {
  local port="$1"
  [[ -n "$(port_listen_pid "$port")" ]]
}

ensure_managed_listener() {
  local name="$1"
  local probe_url="$2"
  local display_url="$3"
  local port="$4"
  local pid_file="$5"
  local expected_pid=""
  local listening_pid=""

  if is_running "$pid_file"; then
    expected_pid="$(cat "$pid_file")"
  fi
  listening_pid="$(port_listen_pid "$port")"

  if curl -fsS --max-time 2 "$probe_url" >/dev/null 2>&1; then
    if [[ -n "$expected_pid" && "$expected_pid" == "$listening_pid" ]]; then
      echo "[service] $name already reachable at $display_url (managed pid=$expected_pid)"
      return 0
    fi
    if [[ -n "$listening_pid" ]]; then
      echo "[service] $name port $port is occupied by unmanaged/stale pid=$listening_pid; replacing it"
      kill "$listening_pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        if ! kill -0 "$listening_pid" 2>/dev/null; then
          break
        fi
        sleep 0.5
      done
      if kill -0 "$listening_pid" 2>/dev/null; then
        echo "[service] force stopping stale $name pid=$listening_pid"
        kill -9 "$listening_pid" 2>/dev/null || true
      fi
    fi
    rm -f "$pid_file"
    return 1
  fi

  if [[ -n "$listening_pid" && ( -z "$expected_pid" || "$expected_pid" != "$listening_pid" ) ]]; then
    echo "[service] $name port $port has unmanaged listener pid=$listening_pid without healthy endpoint; replacing it"
    kill "$listening_pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$listening_pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$listening_pid" 2>/dev/null; then
      echo "[service] force stopping stale $name pid=$listening_pid"
      kill -9 "$listening_pid" 2>/dev/null || true
    fi
  fi

  if [[ -n "$expected_pid" && -n "$listening_pid" && "$expected_pid" != "$listening_pid" ]]; then
    rm -f "$pid_file"
  fi
  return 1
}

stop_pid() {
  local name="$1"
  local pid_file="$2"
  if is_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    echo "[service] stopping $name pid=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[service] force stopping $name pid=$pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pid_file"
}

wait_for_url() {
  local url="$1"
  local pid_file="$2"
  local log_file="$3"
  local name="$4"
  local seconds="$5"
  for _ in $(seq 1 "$seconds"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! is_running "$pid_file"; then
      echo "[service] $name exited while starting. Log:"
      tail -80 "$log_file" || true
      return 1
    fi
    sleep 1
  done
  echo "[service] $name did not become ready. Log:"
  tail -80 "$log_file" || true
  return 1
}

start_tts() {
  if ensure_managed_listener "TTS worker" "$TTS_WORKER_URL/health" "$TTS_WORKER_URL" "$TTS_WORKER_PORT" "$TTS_PID_FILE"; then
    return 0
  fi
  if is_running "$TTS_PID_FILE"; then
    echo "[service] stale TTS worker pid exists; stopping it first"
    stop_pid "TTS worker" "$TTS_PID_FILE"
  fi
  echo "[service] starting TTS worker at $TTS_WORKER_URL"
  TTS_WORKER_HOST="$TTS_WORKER_HOST" TTS_WORKER_PORT="$TTS_WORKER_PORT" \
    nohup bash "$ROOT/scripts/run_tts_worker.sh" >"$TTS_LOG" 2>&1 &
  echo "$!" >"$TTS_PID_FILE"
  wait_for_url "$TTS_WORKER_URL/health" "$TTS_PID_FILE" "$TTS_LOG" "TTS worker" "$TTS_WORKER_START_TIMEOUT"
  echo "[service] TTS worker ready pid=$(cat "$TTS_PID_FILE")"
}

start_avamerg() {
  if ensure_managed_listener "AvaMERG worker" "$AVAMERG_WORKER_URL/health" "$AVAMERG_WORKER_URL" "$AVAMERG_WORKER_PORT" "$AVAMERG_PID_FILE"; then
    return 0
  fi
  if is_running "$AVAMERG_PID_FILE"; then
    echo "[service] stale AvaMERG worker pid exists; stopping it first"
    stop_pid "AvaMERG worker" "$AVAMERG_PID_FILE"
  fi
  echo "[service] starting AvaMERG worker at $AVAMERG_WORKER_URL"
  AVAMERG_WORKER_HOST="$AVAMERG_WORKER_HOST" AVAMERG_WORKER_PORT="$AVAMERG_WORKER_PORT" \
    nohup bash "$ROOT/scripts/run_avamerg_worker.sh" >"$AVAMERG_LOG" 2>&1 &
  echo "$!" >"$AVAMERG_PID_FILE"
  wait_for_url "$AVAMERG_WORKER_URL/health" "$AVAMERG_PID_FILE" "$AVAMERG_LOG" "AvaMERG worker" 240
  echo "[service] AvaMERG worker ready pid=$(cat "$AVAMERG_PID_FILE")"
}

start_deeptalk() {
  if ensure_managed_listener "DEEPTalk worker" "$DEEPTALK_WORKER_URL/health" "$DEEPTALK_WORKER_URL" "$DEEPTALK_WORKER_PORT" "$DEEPTALK_PID_FILE"; then
    return 0
  fi
  if is_running "$DEEPTALK_PID_FILE"; then
    echo "[service] stale DEEPTalk worker pid exists; stopping it first"
    stop_pid "DEEPTalk worker" "$DEEPTALK_PID_FILE"
  fi
  echo "[service] starting DEEPTalk worker at $DEEPTALK_WORKER_URL"
  DEEPTALK_WORKER_HOST="$DEEPTALK_WORKER_HOST" DEEPTALK_WORKER_PORT="$DEEPTALK_WORKER_PORT" \
    nohup bash "$ROOT/scripts/run_deeptalk_worker.sh" >"$DEEPTALK_LOG" 2>&1 &
  echo "$!" >"$DEEPTALK_PID_FILE"
  wait_for_url "$DEEPTALK_WORKER_URL/health" "$DEEPTALK_PID_FILE" "$DEEPTALK_LOG" "DEEPTalk worker" 180
  echo "[service] DEEPTalk worker ready pid=$(cat "$DEEPTALK_PID_FILE")"
}

start_perception() {
  if ensure_managed_listener "perception worker" "$PERCEPTION_WORKER_URL/health" "$PERCEPTION_WORKER_URL" "$PERCEPTION_WORKER_PORT" "$PERCEPTION_PID_FILE"; then
    return 0
  fi
  if is_running "$PERCEPTION_PID_FILE"; then
    echo "[service] stale perception worker pid exists; stopping it first"
    stop_pid "perception worker" "$PERCEPTION_PID_FILE"
  fi
  echo "[service] starting perception worker at $PERCEPTION_WORKER_URL"
  PERCEPTION_WORKER_HOST="$PERCEPTION_WORKER_HOST" PERCEPTION_WORKER_PORT="$PERCEPTION_WORKER_PORT" \
    nohup bash "$ROOT/scripts/run_perception_worker.sh" >"$PERCEPTION_LOG" 2>&1 &
  echo "$!" >"$PERCEPTION_PID_FILE"
  wait_for_url "$PERCEPTION_WORKER_URL/health" "$PERCEPTION_PID_FILE" "$PERCEPTION_LOG" "perception worker" 240
  echo "[service] perception worker ready pid=$(cat "$PERCEPTION_PID_FILE")"
}

start_gaussian_render() {
  if ensure_managed_listener "Gaussian render worker" "$GAUSSIAN_RENDER_WORKER_URL/health" "$GAUSSIAN_RENDER_WORKER_URL" "$GAUSSIAN_RENDER_WORKER_PORT" "$GAUSSIAN_RENDER_PID_FILE"; then
    return 0
  fi
  if is_running "$GAUSSIAN_RENDER_PID_FILE"; then
    echo "[service] stale Gaussian render worker pid exists; stopping it first"
    stop_pid "Gaussian render worker" "$GAUSSIAN_RENDER_PID_FILE"
  fi
  echo "[service] starting Gaussian render worker at $GAUSSIAN_RENDER_WORKER_URL"
  GAUSSIAN_RENDER_WORKER_HOST="$GAUSSIAN_RENDER_WORKER_HOST" GAUSSIAN_RENDER_WORKER_PORT="$GAUSSIAN_RENDER_WORKER_PORT" \
    nohup bash "$ROOT/scripts/run_gaussian_render_worker.sh" >"$GAUSSIAN_RENDER_LOG" 2>&1 &
  echo "$!" >"$GAUSSIAN_RENDER_PID_FILE"
  wait_for_url "$GAUSSIAN_RENDER_WORKER_URL/health" "$GAUSSIAN_RENDER_PID_FILE" "$GAUSSIAN_RENDER_LOG" "Gaussian render worker" 180
  echo "[service] Gaussian render worker ready pid=$(cat "$GAUSSIAN_RENDER_PID_FILE")"
}

start_web() {
  local web_url="http://127.0.0.1:${PORT}/"
  if ensure_managed_listener "web server" "$web_url" "$web_url" "$PORT" "$WEB_PID_FILE"; then
    return 0
  fi
  if is_running "$WEB_PID_FILE"; then
    echo "[service] stale web pid exists; stopping it first"
    stop_pid "web" "$WEB_PID_FILE"
  fi
  echo "[service] starting web server on ${HOST}:${PORT}"
  HOST="$HOST" PORT="$PORT" GAUSSIAN_RENDER_WORKER_URL="$GAUSSIAN_RENDER_WORKER_URL" \
    nohup bash "$WEB_SCRIPT" >"$WEB_LOG" 2>&1 &
  echo "$!" >"$WEB_PID_FILE"
  wait_for_url "$web_url" "$WEB_PID_FILE" "$WEB_LOG" "web server" 30
  echo "[service] web ready pid=$(cat "$WEB_PID_FILE")"
}

status() {
  if is_running "$TTS_PID_FILE"; then
    echo "TTS worker: running pid=$(cat "$TTS_PID_FILE") url=$TTS_WORKER_URL"
  else
    echo "TTS worker: stopped"
  fi
  if is_running "$AVAMERG_PID_FILE"; then
    echo "AvaMERG worker: running pid=$(cat "$AVAMERG_PID_FILE") url=$AVAMERG_WORKER_URL"
  else
    echo "AvaMERG worker: stopped"
  fi
  if is_running "$DEEPTALK_PID_FILE"; then
    echo "DEEPTalk worker: running pid=$(cat "$DEEPTALK_PID_FILE") url=$DEEPTALK_WORKER_URL"
  else
    echo "DEEPTalk worker: stopped"
  fi
  if is_running "$PERCEPTION_PID_FILE"; then
    echo "perception worker: running pid=$(cat "$PERCEPTION_PID_FILE") url=$PERCEPTION_WORKER_URL"
  else
    echo "perception worker: stopped"
  fi
  if is_running "$GAUSSIAN_RENDER_PID_FILE"; then
    echo "Gaussian render worker: running pid=$(cat "$GAUSSIAN_RENDER_PID_FILE") url=$GAUSSIAN_RENDER_WORKER_URL"
  else
    echo "Gaussian render worker: stopped"
  fi
  if is_running "$WEB_PID_FILE"; then
    echo "web server: running pid=$(cat "$WEB_PID_FILE") url=http://127.0.0.1:${PORT}"
  else
    echo "web server: stopped"
  fi
  ss -ltnp | grep -E ":(${PORT}|${TTS_WORKER_PORT}|${AVAMERG_WORKER_PORT}|${DEEPTALK_WORKER_PORT}|${PERCEPTION_WORKER_PORT}|${GAUSSIAN_RENDER_WORKER_PORT})\\b" || true
}

case "${1:-start}" in
  start)
    start_tts
    if [[ "$START_PERCEPTION_WORKER" == "1" ]]; then
      start_perception
    fi
    if [[ "$START_AVAMERG_WORKER" == "1" ]]; then
      start_avamerg
    fi
    if [[ "$START_DEEPTALK_WORKER" == "1" ]]; then
      start_deeptalk
    fi
    if [[ "$START_GAUSSIAN_RENDER_WORKER" == "1" ]]; then
      start_gaussian_render
    fi
    start_web
    echo "[service] ready. Open http://localhost:${PORT}"
    ;;
  stop)
    stop_pid "web" "$WEB_PID_FILE"
    stop_pid "Gaussian render worker" "$GAUSSIAN_RENDER_PID_FILE"
    stop_pid "perception worker" "$PERCEPTION_PID_FILE"
    stop_pid "DEEPTalk worker" "$DEEPTALK_PID_FILE"
    stop_pid "AvaMERG worker" "$AVAMERG_PID_FILE"
    stop_pid "TTS worker" "$TTS_PID_FILE"
    ;;
  start-avamerg)
    start_avamerg
    ;;
  start-tts)
    start_tts
    ;;
  stop-tts)
    stop_pid "TTS worker" "$TTS_PID_FILE"
    ;;
  stop-avamerg)
    stop_pid "AvaMERG worker" "$AVAMERG_PID_FILE"
    ;;
  start-deeptalk)
    start_deeptalk
    ;;
  stop-deeptalk)
    stop_pid "DEEPTalk worker" "$DEEPTALK_PID_FILE"
    ;;
  start-perception)
    start_perception
    ;;
  stop-perception)
    stop_pid "perception worker" "$PERCEPTION_PID_FILE"
    ;;
  start-gaussian-render)
    start_gaussian_render
    ;;
  stop-gaussian-render)
    stop_pid "Gaussian render worker" "$GAUSSIAN_RENDER_PID_FILE"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status
    ;;
  logs)
    echo "==> $TTS_LOG <=="
    tail -80 "$TTS_LOG" 2>/dev/null || true
    echo
    echo "==> $WEB_LOG <=="
    tail -80 "$WEB_LOG" 2>/dev/null || true
    echo
    echo "==> $AVAMERG_LOG <=="
    tail -80 "$AVAMERG_LOG" 2>/dev/null || true
    echo
    echo "==> $DEEPTALK_LOG <=="
    tail -80 "$DEEPTALK_LOG" 2>/dev/null || true
    echo
    echo "==> $PERCEPTION_LOG <=="
    tail -80 "$PERCEPTION_LOG" 2>/dev/null || true
    echo
    echo "==> $GAUSSIAN_RENDER_LOG <=="
    tail -80 "$GAUSSIAN_RENDER_LOG" 2>/dev/null || true
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|start-tts|stop-tts|start-avamerg|stop-avamerg|start-deeptalk|stop-deeptalk|start-perception|stop-perception|start-gaussian-render|stop-gaussian-render}"
    exit 2
    ;;
esac
