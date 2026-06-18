#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
PYTHON="${AVATAR_PYTHON:-$ROOT/runtime/cache/venvs/web/bin/python}"
CONTAINER="${AVATAR_CONTAINER:-$ROOT/runtime/containers/gaussianav_jammy}"

usage() {
  cat <<'EOF'
Usage:
  scripts/avatar.sh web [uvicorn args...]
  scripts/avatar.sh booth [uvicorn args...]
  scripts/avatar.sh agent [input.wav] [avatar_id] [pipeline args...]
  scripts/avatar.sh 3depb
  scripts/avatar.sh worker tts|avamerg|deeptalk|perception|gaussian
  scripts/avatar.sh service web|booth|3depb start|stop|restart|status|logs
  scripts/avatar.sh booth-service start|stop|restart|status|logs
EOF
}

common_env() {
  export HF_HOME="${HF_HOME:-$ROOT/runtime/cache/hf}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT/runtime/cache/xdg}"
  export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$ROOT/runtime/cache/modelscope}"
  export NLTK_DATA="${NLTK_DATA:-$ROOT/runtime/cache/nltk_data}"
  export AVATAR_FFMPEG="${AVATAR_FFMPEG:-$ROOT/runtime/cache/bin/ffmpeg}"
  export AVATAR_FFPROBE="${AVATAR_FFPROBE:-$ROOT/runtime/cache/bin/ffprobe}"
  export DEPB_FFMPEG="${DEPB_FFMPEG:-$AVATAR_FFMPEG}"
  export PATH="$ROOT/runtime/cache/bin:$PATH"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
  mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$MODELSCOPE_CACHE" "$NLTK_DATA" "$ROOT/runtime/outputs"
}

run_web() {
  local host="${HOST:-0.0.0.0}"
  local port="${PORT:-7861}"
  common_env
  cd "$ROOT"
  exec "$PYTHON" -m uvicorn apps.web.server:app --host "$host" --port "$port" "$@"
}

run_booth() {
  export BOOTH_DEFAULT_ROUTE="${BOOTH_DEFAULT_ROUTE:-1}"
  PORT="${PORT:-7862}" run_web "$@"
}

run_agent() {
  local input_wav="$ROOT/integrations/perception/data/demo_wavs/sample_dialog_02.wav"
  local avatar_id="306"
  if [[ $# -gt 0 && "$1" != --* ]]; then input_wav="$1"; shift; fi
  if [[ $# -gt 0 && "$1" != --* ]]; then avatar_id="$1"; shift; fi
  common_env
  local agent_python="${AGENT_PYTHON:-$ROOT/runtime/cache/venvs/deeptalk/bin/python}"
  export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
  cd "$ROOT"
  exec "$agent_python" -m avatar_system.pipeline.cli \
    --input_wav "$input_wav" \
    --avatar_id "$avatar_id" \
    --config "$ROOT/src/avatar_system/pipeline_config.yaml" \
    "$@"
}

run_3depb() {
  local port="${PORT:-7862}"
  export PORT="$port"
  cd "$ROOT/apps/booth"
  exec /usr/bin/python3 server.py
}

container_exec() {
  local cmd="$1"
  local flags="${APPTAINER_FLAGS:---nv}"
  exec apptainer exec $flags \
    -B /scratch:/scratch,/home/svu:/home/svu \
    "$CONTAINER" \
    bash -lc "$cmd"
}

run_worker() {
  local kind="${1:-}"
  shift || true
  common_env
  case "$kind" in
    tts)
      container_exec "cd '$ROOT/integrations/emotivoice' && '$ROOT/integrations/emotivoice/.EmotiVoice/bin/python' tts_worker.py --host '${TTS_WORKER_HOST:-127.0.0.1}' --port '${TTS_WORKER_PORT:-8788}' --logdir prompt_tts_open_source_joint --config_folder config/joint --checkpoint g_00140000 $*"
      ;;
    avamerg)
      container_exec "export PYTHONPATH='$ROOT/integrations/avamerg/merg_code:$ROOT/integrations/avamerg':\$PYTHONPATH; cd '$ROOT/integrations/avamerg' && '$ROOT/integrations/avamerg/.avamerg38/bin/python' avamerg_worker.py --host '${AVAMERG_WORKER_HOST:-127.0.0.1}' --port '${AVAMERG_WORKER_PORT:-8789}' $*"
      ;;
    deeptalk)
      cd "$ROOT/integrations/deeptalk/DEEPTalk"
      exec "$ROOT/runtime/cache/venvs/deeptalk/bin/python" deeptalk_worker.py --host "${DEEPTALK_WORKER_HOST:-127.0.0.1}" --port "${DEEPTALK_WORKER_PORT:-8790}" "$@"
      ;;
    perception)
      container_exec "cd '$ROOT/integrations/perception' && '$ROOT/runtime/cache/venvs/perception/bin/python' scripts/perception_worker.py --host '${PERCEPTION_WORKER_HOST:-127.0.0.1}' --port '${PERCEPTION_WORKER_PORT:-8791}' --model '${PERCEPTION_WORKER_MODEL:-small}' --ser_model '${PERCEPTION_WORKER_SER_MODEL:-iic/emotion2vec_plus_seed}' $*"
      ;;
    gaussian)
      container_exec "export PYTHONPATH='$ROOT/integrations/gaussian_avatar:$ROOT/src':\$PYTHONPATH; cd '$ROOT/integrations/gaussian_avatar' && '$ROOT/integrations/gaussian_avatar/.GSavatar_glibc/bin/python' gaussian_render_worker.py --host '${GAUSSIAN_RENDER_WORKER_HOST:-127.0.0.1}' --port '${GAUSSIAN_RENDER_WORKER_PORT:-8792}' $*"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

service_log_dir() {
  printf '%s\n' "$ROOT/runtime/outputs/service_logs"
}

service_pid_file() {
  local name="$1"
  printf '%s/%s.pid\n' "$(service_log_dir)" "$name"
}

service_log_file() {
  local name="$1"
  printf '%s/%s.log\n' "$(service_log_dir)" "$name"
}

service_is_running() {
  local pid_file="$1"
  [[ -s "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

service_stop_pid() {
  local name="$1"
  local pid_file
  pid_file="$(service_pid_file "$name")"
  if service_is_running "$pid_file"; then
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

service_wait_for_url() {
  local name="$1"
  local url="$2"
  local seconds="$3"
  local pid_file log_file
  pid_file="$(service_pid_file "$name")"
  log_file="$(service_log_file "$name")"
  for _ in $(seq 1 "$seconds"); do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    if ! service_is_running "$pid_file"; then
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

service_start_bg() {
  local name="$1"
  local url="$2"
  local timeout="$3"
  shift 3
  local pid_file log_file
  pid_file="$(service_pid_file "$name")"
  log_file="$(service_log_file "$name")"
  mkdir -p "$(service_log_dir)"

  if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
    echo "[service] $name already reachable at $url"
    return 0
  fi
  if service_is_running "$pid_file"; then
    echo "[service] stale $name pid exists; stopping it first"
    service_stop_pid "$name"
  fi

  echo "[service] starting $name"
  nohup "$@" >"$log_file" 2>&1 &
  echo "$!" >"$pid_file"
  service_wait_for_url "$name" "$url" "$timeout"
  echo "[service] $name ready pid=$(cat "$pid_file")"
}

service_start_worker() {
  local kind="$1"
  local name="$2"
  local url="$3"
  local timeout="$4"
  service_start_bg "$name" "$url/health" "$timeout" bash "$ROOT/scripts/avatar.sh" worker "$kind"
}

service_start_frontend() {
  local mode="$1"
  local name="$2"
  local port="$3"
  local url="http://127.0.0.1:${port}/"
  service_start_bg "$name" "$url" 30 env PORT="$port" bash "$ROOT/scripts/avatar.sh" "$mode"
}

service_start() {
  local mode="$1"
  common_env
  mkdir -p "$(service_log_dir)"
  if [[ "${START_TTS_WORKER:-1}" == "1" ]]; then
    service_start_worker tts tts_worker "http://${TTS_WORKER_HOST:-127.0.0.1}:${TTS_WORKER_PORT:-8788}" "${TTS_WORKER_START_TIMEOUT:-240}"
  fi
  if [[ "${START_AVAMERG_WORKER:-1}" == "1" ]]; then
    service_start_worker avamerg avamerg_worker "http://${AVAMERG_WORKER_HOST:-127.0.0.1}:${AVAMERG_WORKER_PORT:-8789}" 240
  fi
  if [[ "${START_DEEPTALK_WORKER:-1}" == "1" ]]; then
    service_start_worker deeptalk deeptalk_worker "http://${DEEPTALK_WORKER_HOST:-127.0.0.1}:${DEEPTALK_WORKER_PORT:-8790}" 180
  fi
  if [[ "${START_PERCEPTION_WORKER:-1}" == "1" ]]; then
    service_start_worker perception perception_worker "http://${PERCEPTION_WORKER_HOST:-127.0.0.1}:${PERCEPTION_WORKER_PORT:-8791}" 240
  fi
  if [[ "${START_GAUSSIAN_RENDER_WORKER:-1}" == "1" ]]; then
    service_start_worker gaussian gaussian_render_worker "http://${GAUSSIAN_RENDER_WORKER_HOST:-127.0.0.1}:${GAUSSIAN_RENDER_WORKER_PORT:-8792}" 180
  fi

  case "$mode" in
    web) service_start_frontend web web 7861 ;;
    booth) service_start_frontend booth booth_web 7862 ;;
    3depb) service_start_frontend 3depb booth_web 7862 ;;
    *) usage; exit 2 ;;
  esac
}

service_stop() {
  service_stop_pid booth_web
  service_stop_pid web
  service_stop_pid gaussian_render_worker
  service_stop_pid perception_worker
  service_stop_pid deeptalk_worker
  service_stop_pid avamerg_worker
  service_stop_pid tts_worker
}

service_status() {
  local name pid_file
  for name in web booth_web tts_worker avamerg_worker deeptalk_worker perception_worker gaussian_render_worker; do
    pid_file="$(service_pid_file "$name")"
    if service_is_running "$pid_file"; then
      echo "[service] $name running pid=$(cat "$pid_file")"
    else
      echo "[service] $name stopped"
    fi
  done
}

service_logs() {
  mkdir -p "$(service_log_dir)"
  tail -n 120 -f "$(service_log_dir)"/*.log
}

run_service() {
  local mode="${1:-web}"
  local action="${2:-start}"
  case "$action" in
    start) service_start "$mode" ;;
    stop) service_stop ;;
    restart) service_stop; service_start "$mode" ;;
    status) service_status ;;
    logs) service_logs ;;
    *) usage; exit 2 ;;
  esac
}

case "${1:-}" in
  web) shift; run_web "$@" ;;
  booth) shift; run_booth "$@" ;;
  agent) shift; run_agent "$@" ;;
  3depb) shift; run_3depb "$@" ;;
  worker) shift; run_worker "$@" ;;
  service) shift; run_service "$@" ;;
  booth-service) shift; run_service 3depb "${1:-start}" ;;
  -h|--help|help|"") usage ;;
  *) usage; exit 2 ;;
esac
