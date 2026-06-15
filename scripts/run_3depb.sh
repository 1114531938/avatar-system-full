#!/usr/bin/env bash
set -euo pipefail

ROOT="${AVATAR_SYSTEM_ROOT:-/scratch/e1554543/avatar_system_full}"
PORT="${PORT:-7862}"
DEPB_ROOT="${DEPB_ROOT:-$ROOT/3DEPB_runs/3DEPB}"
DEPB_START_CMD="${DEPB_START_CMD:-}"
export PORT

if [[ ! -d "$DEPB_ROOT" ]]; then
  cat >&2 <<EOF
[run_3depb] 3DEPB project not found: $DEPB_ROOT
[run_3depb] Clone it with:
[run_3depb]   git clone https://github.com/sh-lin24/3DEPB.git $DEPB_ROOT
EOF
  exit 1
fi

cd "$DEPB_ROOT"

if [[ -n "$DEPB_START_CMD" ]]; then
  exec bash -lc "$DEPB_START_CMD"
fi

exec setsid /usr/bin/python3 server.py
