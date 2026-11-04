#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
RUNTIME_ENV="${OMNIVOICE_RUNTIME_ENV:-$REPO_DIR/.env}"

if [[ -f "$RUNTIME_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  set +a
fi

read_pid1_env() {
  local key="$1"
  if [[ -r /proc/1/environ ]]; then
    tr '\0' '\n' < /proc/1/environ 2>/dev/null |
      awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}'
  fi
}

PORT="${PORT:-8000}"
export PUBLIC_IPADDR="${PUBLIC_IPADDR:-$(read_pid1_env PUBLIC_IPADDR)}"
mapped_port_key="VAST_TCP_PORT_${PORT}"
mapped_port="${!mapped_port_key:-$(read_pid1_env "$mapped_port_key")}" 
if [[ -n "$mapped_port" ]]; then
  export LINGUA_WORKER_PUBLIC_PORT="${LINGUA_WORKER_PUBLIC_PORT:-$mapped_port}"
fi
export LINGUA_WORKER_PUBLIC_IP="${LINGUA_WORKER_PUBLIC_IP:-$PUBLIC_IPADDR}"

exec "$SCRIPT_DIR/start_api.sh" "$@"
