#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Stop the engines and proxy started by run_1p1d.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_FILE="$LOG_DIR/pids"

if [[ -f "$PID_FILE" ]]; then
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop_1p1d.sh] SIGTERM $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done <"$PID_FILE"
  sleep 20
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "[stop_1p1d.sh] SIGKILL $pid"
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <"$PID_FILE"
  rm -f "$PID_FILE"
else
  echo "[stop_1p1d.sh] no pid file at $PID_FILE"
fi

# Engine workers and the proxy can outlive their parent; clean up leftovers.
pkill -f "load_balance_proxy_layerwise_server_example.py" 2>/dev/null || true
echo "[stop_1p1d.sh] remaining vllm processes:"
pgrep -af "vllm serve" || echo "  none"
echo "[stop_1p1d.sh] if any remain, stop them with: pkill -f 'vllm serve'"
