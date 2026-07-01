#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${AVATAR_SYSTEM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
FORCE="${AVATAR_REBUILD_VENVS_FORCE:-0}"

venv_exists() {
  [[ -x "$1/bin/python" ]]
}

make_venv() {
  local name="$1"
  local python_bin="$2"
  local venv_dir="$3"
  local req_file="$4"

  if venv_exists "$venv_dir" && [[ "$FORCE" != "1" && "$FORCE" != "true" && "$FORCE" != "yes" ]]; then
    echo "Using existing $name venv: $venv_dir"
    return
  fi

  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Missing Python interpreter for $name: $python_bin" >&2
    exit 1
  fi
  if [[ ! -f "$ROOT/$req_file" ]]; then
    echo "Missing requirements file for $name: $req_file" >&2
    exit 1
  fi

  echo "Creating $name venv: $venv_dir"
  rm -rf "$venv_dir"
  mkdir -p "$(dirname "$venv_dir")"
  "$python_bin" -m venv "$venv_dir"
  "$venv_dir/bin/python" -m pip install -U pip setuptools wheel
  "$venv_dir/bin/python" -m pip install -r "$ROOT/$req_file"
}

PYTHON3="${AVATAR_PYTHON3:-python3}"
PYTHON38="${AVATAR_PYTHON38:-python3.8}"

make_venv "web" "$PYTHON3" \
  "$ROOT/runtime/cache/venvs/web" \
  "apps/web/requirements.txt"

make_venv "perception" "$PYTHON3" \
  "$ROOT/runtime/cache/venvs/perception" \
  "integrations/perception/env/requirements_whisper_stage1.txt"

make_venv "deeptalk" "$PYTHON3" \
  "$ROOT/runtime/cache/venvs/deeptalk" \
  "integrations/deeptalk/requirements.txt"

make_venv "avamerg" "$PYTHON38" \
  "$ROOT/integrations/avamerg/.avamerg38" \
  "integrations/avamerg/requirements.txt"

make_venv "emotivoice" "$PYTHON3" \
  "$ROOT/integrations/emotivoice/.EmotiVoice" \
  "integrations/emotivoice/requirements.txt"

make_venv "gaussian_avatar" "$PYTHON3" \
  "$ROOT/integrations/gaussian_avatar/.GSavatar_glibc" \
  "integrations/gaussian_avatar/requirements.txt"

echo "Runtime Python environments are ready."
