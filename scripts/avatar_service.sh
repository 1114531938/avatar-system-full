#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
exec bash "$ROOT/scripts/avatar.sh" service web "${1:-start}"
