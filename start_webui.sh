#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_SPACE_ROOT_FILE="$ROOT/.project_space_root_default"
if [[ -f "$DEFAULT_PROJECT_SPACE_ROOT_FILE" ]]; then
  DEFAULT_PROJECT_SPACE_ROOT="$(<"$DEFAULT_PROJECT_SPACE_ROOT_FILE")"
else
  DEFAULT_PROJECT_SPACE_ROOT="$ROOT/project_space"
fi
PROJECT_SPACE_ROOT="${CATMASTER_PROJECT_SPACE_ROOT:-$DEFAULT_PROJECT_SPACE_ROOT}"
CONDA_ENV_NAME="${CATMASTER_CONDA_ENV:-catmaster}"
HOST="${CATMASTER_HOST:-127.0.0.1}"
PORT="${CATMASTER_PORT:-7860}"

has_flag() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f "$ROOT/scripts/install_jsmol_assets.py" ]]; then
  python3 "$ROOT/scripts/install_jsmol_assets.py" --quiet
fi

CMD=(python -m catmaster.webui)

if ! has_flag "--project-space-root" "$@"; then
  CMD+=(--project-space-root "$PROJECT_SPACE_ROOT")
fi

if ! has_flag "--host" "$@"; then
  CMD+=(--host "$HOST")
fi

if ! has_flag "--port" "$@"; then
  CMD+=(--port "$PORT")
fi

CMD+=("$@")

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
  exec "${CMD[@]}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available in PATH." >&2
  echo "Either activate your conda setup first, or run inside env '$CONDA_ENV_NAME'." >&2
  exit 1
fi

if conda run --help 2>/dev/null | grep -q -- "--no-capture-output"; then
  exec conda run --no-capture-output -n "$CONDA_ENV_NAME" "${CMD[@]}"
fi

exec conda run -n "$CONDA_ENV_NAME" "${CMD[@]}"
