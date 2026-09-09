#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNTIME_ENV="${OMNIVOICE_RUNTIME_ENV:-$REPO_DIR/.env}"
LOG_DIR="${OMNIVOICE_LOG_DIR:-/workspace/logs/omnivoice-triton}"
PID_FILE="$LOG_DIR/server.pid"
LOG_FILE="$LOG_DIR/server.log"
SESSION_NAME="${OMNIVOICE_TMUX_SESSION:-omnivoice-triton}"

if [[ -f "$RUNTIME_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  set +a
fi

PORT="${PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/healthz"

read_pid1_env() {
  local key="$1"
  if [[ -r /proc/1/environ ]]; then
    tr '\0' '\n' < /proc/1/environ 2>/dev/null |
      awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}'
  fi
}

export PUBLIC_IPADDR="${PUBLIC_IPADDR:-$(read_pid1_env PUBLIC_IPADDR)}"
mapped_port_key="VAST_TCP_PORT_${PORT}"
mapped_port="${!mapped_port_key:-$(read_pid1_env "$mapped_port_key")}" 
if [[ -n "$mapped_port" ]]; then
  export LINGUA_WORKER_PUBLIC_PORT="${LINGUA_WORKER_PUBLIC_PORT:-$mapped_port}"
fi
export LINGUA_WORKER_PUBLIC_IP="${LINGUA_WORKER_PUBLIC_IP:-$PUBLIC_IPADDR}"

running_pid() {
  [[ -s "$PID_FILE" ]] || return 1
  local pid
  pid=$(tr -d '[:space:]' < "$PID_FILE")
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

case "${1:-status}" in
  start)
    mkdir -p "$LOG_DIR"
    if pid=$(running_pid); then
      echo "OmniVoice is already running (pid $pid)."
      exit 0
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
    fi
    if [[ -f "$PID_FILE" ]]; then
      rm -f "$PID_FILE"
    fi
    : > "$LOG_FILE"
    tmux new-session -d -s "$SESSION_NAME" \
      "exec '$SCRIPT_DIR/start_worker_process.sh' >>'$LOG_FILE' 2>&1"
    pid=$(tmux display-message -p -t "$SESSION_NAME" '#{pane_pid}')
    printf '%s\n' "$pid" > "$PID_FILE"
    echo "Started OmniVoice (pid $pid); log: $LOG_FILE"
    ;;
  status)
    if pid=$(running_pid); then
      if curl -fsS --max-time 3 "$HEALTH_URL"; then
        echo
      else
        echo "OmniVoice is starting or unhealthy (pid $pid)."
      fi
    else
      echo "OmniVoice is stopped."
      exit 1
    fi
    ;;
  stop)
    if pid=$(running_pid); then
      kill -TERM "$pid"
      echo "Stopping OmniVoice (pid $pid)."
    else
      echo "OmniVoice is already stopped."
    fi
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      tmux kill-session -t "$SESSION_NAME"
    fi
    ;;
  logs)
    touch "$LOG_FILE"
    tail -n "${LOG_LINES:-100}" "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|status|stop|logs}" >&2
    exit 2
    ;;
esac
