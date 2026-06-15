#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
export PORT="${PORT:-7862}"
export DEPB_ROOT="${DEPB_ROOT:-$ROOT/3DEPB_runs/3DEPB}"
export WEB_SCRIPT="${WEB_SCRIPT:-$ROOT/scripts/run_3depb.sh}"
export WEB_LOG="${WEB_LOG:-$ROOT/outputs/service_logs/booth_web.log}"
export WEB_PID_FILE="${WEB_PID_FILE:-$ROOT/outputs/service_logs/booth_web.pid}"
export START_TTS_WORKER="${START_TTS_WORKER:-1}"
export START_AVAMERG_WORKER="${START_AVAMERG_WORKER:-1}"
export START_DEEPTALK_WORKER="${START_DEEPTALK_WORKER:-1}"
export START_PERCEPTION_WORKER="${START_PERCEPTION_WORKER:-1}"
export START_GAUSSIAN_RENDER_WORKER="${START_GAUSSIAN_RENDER_WORKER:-1}"

exec bash "$ROOT/scripts/avatar_service.sh" "${1:-start}"
