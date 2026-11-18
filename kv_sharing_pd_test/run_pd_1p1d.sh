#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Single-node 1P1D (MooncakeConnectorV2) for a model with cross-layer KV sharing,
# e.g. google/gemma-4-E2B-it. Prefill on one NPU, decode on another, plus the
# load-balance proxy. Wraps examples/disaggregated_prefill_v1/launch_single_node_pd.py.
#
#   bash run_pd_1p1d.sh /models/gemma-4-E2B-it
#   MODEL=/models/gemma-4-E2B-it PREFILL_NPUS=0 DECODE_NPUS=1 bash run_pd_1p1d.sh
#
# Ctrl-C stops every engine. Logs land in $LOG_DIR.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHER="$REPO_ROOT/examples/disaggregated_prefill_v1/launch_single_node_pd.py"

# A leading non-flag argument is the model; anything else is passed through to
# the launcher (e.g. --additional-config '{...}').
MODEL="${MODEL:-google/gemma-4-E2B-it}"
if [[ $# -gt 0 && $1 != --* ]]; then
  MODEL="$1"
  shift
fi

# --- topology -----------------------------------------------------------------
PREFILL_NPUS="${PREFILL_NPUS:-0}"
DECODE_NPUS="${DECODE_NPUS:-1}"
TP_SIZE="${TP_SIZE:-1}"

# --- ports --------------------------------------------------------------------
PREFILL_PORT="${PREFILL_PORT:-13700}"
DECODE_PORT="${DECODE_PORT:-13701}"
PROXY_PORT="${PROXY_PORT:-8080}"
PREFILL_KV_PORT="${PREFILL_KV_PORT:-30000}"
DECODE_KV_PORT="${DECODE_KV_PORT:-30100}"

# --- serving ------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
LOG_DIR="${LOG_DIR:-/tmp/gemma4_kv_sharing_pd}"

# --- networking (single node: one NIC, one IP) --------------------------------
# HCCL_IF_IP must be this node's host IP, not 127.0.0.1: the connector
# advertises it to the peer engine in the handshake.
NIC="${NIC:-$(ip -o -4 route show to default | awk '{print $5; exit}')}"
export HCCL_IF_IP="${HCCL_IF_IP:-$(ip -o -4 addr show dev "$NIC" | awk '{print $4}' | cut -d/ -f1)}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NIC}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$NIC}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$NIC}"

# Keep KV-sharing fast prefill OFF. It swaps Gemma4 onto the self/cross
# decoder-layer split, which is a different code path from the one the
# connector's borrower handling is written against.
export VLLM_SERVER_DEV_MODE=1          # /reset_prefix_cache for verify_pd.py
export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"

echo "model        : $MODEL"
echo "prefill NPUs : $PREFILL_NPUS   decode NPUs: $DECODE_NPUS   TP: $TP_SIZE"
echo "node IP      : $HCCL_IF_IP ($NIC)"
echo "proxy        : http://127.0.0.1:$PROXY_PORT"
echo "logs         : $LOG_DIR"
echo

exec python "$LAUNCHER" \
  --model "$MODEL" \
  --prefill-npus "$PREFILL_NPUS" \
  --decode-npus "$DECODE_NPUS" \
  --tp-size "$TP_SIZE" \
  --kv-connector MooncakeConnectorV2 \
  --prefill-port "$PREFILL_PORT" \
  --decode-port "$DECODE_PORT" \
  --proxy-port "$PROXY_PORT" \
  --prefill-kv-port "$PREFILL_KV_PORT" \
  --decode-kv-port "$DECODE_KV_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --prefill-max-num-batched-tokens "$MAX_MODEL_LEN" \
  --decode-max-num-batched-tokens 256 \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --log-dir "$LOG_DIR" \
  "$@"
