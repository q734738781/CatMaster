#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local startup defaults.
# Set these if you want convenient persistent values when launching via this script.
LOCAL_PROJECT_SPACE_ROOT=""
LOCAL_CONDA_ENV_NAME="catmaster"
LOCAL_HOST="127.0.0.1"
LOCAL_PORT="7990"

PROJECT_SPACE_ROOT="${CATMASTER_PROJECT_SPACE_ROOT:-${LOCAL_PROJECT_SPACE_ROOT:-$ROOT/project_space}}"
CONDA_ENV_NAME="${CATMASTER_CONDA_ENV:-${LOCAL_CONDA_ENV_NAME:-catmaster}}"
HOST="${CATMASTER_HOST:-${LOCAL_HOST:-127.0.0.1}}"
PORT="${CATMASTER_PORT:-${LOCAL_PORT:-7860}}"
RUNTIME_DIR="${CATMASTER_RUNTIME_DIR:-$ROOT/.runtime}"
LOG_FILE="${CATMASTER_WEBUI_LOG:-$RUNTIME_DIR/webui.log}"
PID_FILE="${CATMASTER_WEBUI_PID:-$RUNTIME_DIR/webui.pid}"

RUN_MODE="background"
declare -a FORWARD_ARGS=()

usage() {
  cat <<EOF
Usage: ./start_webui.sh [script-options] [webui-options]

Script options:
  --start        Start WebUI in the background (default).
  --foreground   Run WebUI in the current terminal.
  --status       Show whether the background WebUI is running.
  --stop         Stop the background WebUI recorded in $PID_FILE.
  --help         Show this help.

All other arguments are passed through to \`python -m catmaster.webui\`.
Example: ./start_webui.sh --foreground --no-login
EOF
}

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

pid_command() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" -o args= 2>/dev/null || true
}

is_webui_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  is_running_pid "$pid" || return 1
  local cmd
  cmd="$(pid_command "$pid")"
  [[ "$cmd" == *" -m catmaster.webui"* ]] || return 1
  [[ "$cmd" != *"conda run"* ]] || return 1
}

pid_field() {
  local pid="$1"
  local field="$2"
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" -o "${field}=" 2>/dev/null | tr -d '[:space:]'
}

pid_related_to_launcher() {
  local pid="$1"
  local launcher_pid="$2"
  [[ -n "$pid" && -n "$launcher_pid" ]] || return 1
  [[ "$pid" == "$launcher_pid" ]] && return 0

  local pid_ppid pid_pgid pid_sid launcher_pgid launcher_sid
  pid_ppid="$(pid_field "$pid" ppid || true)"
  pid_pgid="$(pid_field "$pid" pgid || true)"
  pid_sid="$(pid_field "$pid" sid || true)"
  launcher_pgid="$(pid_field "$launcher_pid" pgid || true)"
  launcher_sid="$(pid_field "$launcher_pid" sid || true)"

  [[ -n "$pid_ppid" && "$pid_ppid" == "$launcher_pid" ]] && return 0
  [[ -n "$pid_pgid" && "$pid_pgid" == "$launcher_pid" ]] && return 0
  [[ -n "$pid_sid" && "$pid_sid" == "$launcher_pid" ]] && return 0
  [[ -n "$pid_pgid" && -n "$launcher_pgid" && "$pid_pgid" == "$launcher_pgid" ]] && return 0
  [[ -n "$pid_sid" && -n "$launcher_sid" && "$pid_sid" == "$launcher_sid" ]] && return 0
  return 1
}

log_size() {
  local path="$1"
  if [[ -f "$path" ]]; then
    wc -c < "$path" | tr -d '[:space:]'
  else
    echo "0"
  fi
}

server_pid_from_log_since() {
  local offset="$1"
  [[ -f "$LOG_FILE" ]] || return 1
  tail -c "+$((offset + 1))" "$LOG_FILE" 2>/dev/null \
    | sed -n 's/.*Started server process \[\([0-9][0-9]*\)\].*/\1/p' \
    | tail -n 1
}

server_pid_from_port() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "( sport = :$port )" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | head -n 1
    return 0
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -n 1
    return 0
  fi
  return 1
}

resolve_started_webui_pid() {
  local launcher_pid="$1"
  local log_start_size="$2"
  local candidate

  candidate="$(server_pid_from_port "$PORT" || true)"
  if [[ -n "$candidate" ]] && is_webui_pid "$candidate" && pid_related_to_launcher "$candidate" "$launcher_pid"; then
    echo "$candidate"
    return 0
  fi

  candidate="$(server_pid_from_log_since "$log_start_size" || true)"
  if [[ -n "$candidate" ]] && is_webui_pid "$candidate" && pid_related_to_launcher "$candidate" "$launcher_pid"; then
    echo "$candidate"
    return 0
  fi

  if is_webui_pid "$launcher_pid"; then
    echo "$launcher_pid"
    return 0
  fi

  return 1
}

read_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  tr -d '[:space:]' < "$PID_FILE"
}

remove_stale_pid_file() {
  if [[ -f "$PID_FILE" ]]; then
    local stale_pid
    stale_pid="$(read_pid || true)"
    if [[ -z "$stale_pid" ]] || ! is_webui_pid "$stale_pid"; then
      rm -f "$PID_FILE"
    fi
  fi
}

print_status() {
  remove_stale_pid_file

  local pid
  pid="$(read_pid || true)"
  if [[ -n "$pid" ]] && is_webui_pid "$pid"; then
    echo "CatMaster WebUI is running in background."
    echo "PID: $pid"
    echo "Command: $(pid_command "$pid")"
    echo "Log: $LOG_FILE"
    return 0
  fi

  echo "CatMaster WebUI is not running."
  echo "Expected PID file: $PID_FILE"
  echo "Log: $LOG_FILE"
  return 1
}

stop_webui() {
  remove_stale_pid_file

  local pid
  pid="$(read_pid || true)"
  if [[ -z "$pid" ]]; then
    echo "CatMaster WebUI is not running."
    return 0
  fi

  echo "Stopping CatMaster WebUI (PID $pid)..."
  kill "$pid" >/dev/null 2>&1 || true

  local _i
  for _i in {1..30}; do
    if ! is_running_pid "$pid"; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 1
  done

  echo "Process did not exit within 30s; sending SIGKILL." >&2
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  echo "Stopped."
}

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      RUN_MODE="background"
      shift
      ;;
    --foreground)
      RUN_MODE="foreground"
      shift
      ;;
    --status)
      RUN_MODE="status"
      shift
      ;;
    --stop)
      RUN_MODE="stop"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUNTIME_DIR"
remove_stale_pid_file

case "$RUN_MODE" in
  status)
    print_status
    exit $?
    ;;
  stop)
    stop_webui
    exit 0
    ;;
esac

if [[ -f "$ROOT/scripts/install_jsmol_assets.py" ]]; then
  python3 "$ROOT/scripts/install_jsmol_assets.py" --quiet
fi

CMD=(python -m catmaster.webui)

if ! has_flag "--project-space-root" "${FORWARD_ARGS[@]}"; then
  CMD+=(--project-space-root "$PROJECT_SPACE_ROOT")
fi

if ! has_flag "--host" "${FORWARD_ARGS[@]}"; then
  CMD+=(--host "$HOST")
fi

if ! has_flag "--port" "${FORWARD_ARGS[@]}"; then
  CMD+=(--port "$PORT")
fi

CMD+=("${FORWARD_ARGS[@]}")

if [[ -f "$PID_FILE" ]]; then
  existing_pid="$(read_pid || true)"
  if [[ -n "${existing_pid:-}" ]] && is_running_pid "$existing_pid"; then
    echo "CatMaster WebUI is already running in background (PID $existing_pid)." >&2
    echo "Log: $LOG_FILE" >&2
    exit 0
  fi
  rm -f "$PID_FILE"
fi

launch_cmd() {
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
}

if [[ "$RUN_MODE" == "foreground" ]]; then
  launch_cmd
fi

echo "Starting CatMaster WebUI in background..."
echo "Log: $LOG_FILE"
log_start_size="$(log_size "$LOG_FILE")"

if command -v setsid >/dev/null 2>&1; then
  nohup setsid bash -lc '
  set -euo pipefail
  cd "$1"
  export PYTHONPATH="$2"
  export CONDA_ENV_NAME="$3"
  shift 3
  CMD=("$@")

  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    exec "${CMD[@]}"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH." >&2
    echo "Either activate your conda setup first, or run inside env '\''$CONDA_ENV_NAME'\''." >&2
    exit 1
  fi

  if conda run --help 2>/dev/null | grep -q -- "--no-capture-output"; then
    exec conda run --no-capture-output -n "$CONDA_ENV_NAME" "${CMD[@]}"
  fi

  exec conda run -n "$CONDA_ENV_NAME" "${CMD[@]}"
' _ "$ROOT" "$PYTHONPATH" "$CONDA_ENV_NAME" "${CMD[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
else
  nohup bash -lc '
  set -euo pipefail
  cd "$1"
  export PYTHONPATH="$2"
  export CONDA_ENV_NAME="$3"
  shift 3
  CMD=("$@")

  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV_NAME" ]]; then
    exec "${CMD[@]}"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH." >&2
    echo "Either activate your conda setup first, or run inside env '\''$CONDA_ENV_NAME'\''." >&2
    exit 1
  fi

  if conda run --help 2>/dev/null | grep -q -- "--no-capture-output"; then
    exec conda run --no-capture-output -n "$CONDA_ENV_NAME" "${CMD[@]}"
  fi

  exec conda run -n "$CONDA_ENV_NAME" "${CMD[@]}"
' _ "$ROOT" "$PYTHONPATH" "$CONDA_ENV_NAME" "${CMD[@]}" >>"$LOG_FILE" 2>&1 < /dev/null &
fi

bg_pid=$!

actual_pid=""
for _i in {1..30}; do
  actual_pid="$(resolve_started_webui_pid "$bg_pid" "$log_start_size" || true)"
  if [[ -n "$actual_pid" ]] && is_webui_pid "$actual_pid"; then
    echo "$actual_pid" > "$PID_FILE"
    echo "Started. PID: $actual_pid"
    if [[ "$actual_pid" != "$bg_pid" ]]; then
      echo "Launcher PID: $bg_pid"
    fi
    echo "Use './start_webui.sh --status' to check and './start_webui.sh --stop' to stop."
    exit 0
  fi
  sleep 0.2
done

if is_webui_pid "$bg_pid"; then
  echo "$bg_pid" > "$PID_FILE"
  echo "Started. PID: $bg_pid"
  echo "Use './start_webui.sh --status' to check and './start_webui.sh --stop' to stop."
  exit 0
fi

rm -f "$PID_FILE"
echo "WebUI failed to stay up. Check log: $LOG_FILE" >&2
tail -n 40 "$LOG_FILE" >&2 || true
exit 1
