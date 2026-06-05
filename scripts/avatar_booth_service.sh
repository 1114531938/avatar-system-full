#!/usr/bin/env bash
set -euo pipefail

ROOT="/scratch/e1554543/avatar_system_full"
export PORT="${PORT:-7862}"
export WEB_SCRIPT="${WEB_SCRIPT:-$ROOT/scripts/run_booth.sh}"
export WEB_LOG="${WEB_LOG:-$ROOT/outputs/service_logs/booth_web.log}"
export WEB_PID_FILE="${WEB_PID_FILE:-$ROOT/outputs/service_logs/booth_web.pid}"
export BOOTH_DEFAULT_ROUTE="${BOOTH_DEFAULT_ROUTE:-1}"

exec bash "$ROOT/scripts/avatar_service.sh" "${1:-start}"
