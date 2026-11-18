#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Start ONE engine of a single-node 1P1D pair with MooncakeConnectorV2.
# Run it twice, in two shells -- no proxy, which is all verify_pd.py needs.
#
#   # shell 1
#   MODEL=/models/gemma-4-E2B-it bash serve_one.sh prefill
#   # shell 2
#   MODEL=/models/gemma-4-E2B-it bash serve_one.sh decode
#
# Unlike run_pd_1p1d.sh this respects VLLM_LOGGING_LEVEL, so
#   VLLM_LOGGING_LEVEL=DEBUG bash serve_one.sh prefill
# prints the connector's full borrower -> owner map at registration time.
#
# Env knobs: MODEL, TP_SIZE, MAX_MODEL_LEN, MAX_NUM_SEQS, GPU_MEM_UTIL,
# PREFILL_NPUS, DECODE_NPUS, PREFILL_PORT, DECODE_PORT, *_KV_PORT, USE_LAYERWISE.
set -euo pipefail

ROLE="${1:?usage: serve_one.sh prefill|decode}"

MODEL="${MODEL:-google/gemma-4-E2B-it}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"

case "$ROLE" in
  prefill)
    export ASCEND_RT_VISIBLE_DEVICES="${PREFILL_NPUS:-0}"
    PORT="${PREFILL_PORT:-13700}"
    KV_PORT="${PREFILL_KV_PORT:-30000}"
    KV_ROLE="kv_producer"
    ENGINE_ID="prefill-0"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}"
    ;;
  decode)
    export ASCEND_RT_VISIBLE_DEVICES="${DECODE_NPUS:-1}"
    PORT="${DECODE_PORT:-13701}"
    KV_PORT="${DECODE_KV_PORT:-30100}"
    KV_ROLE="kv_consumer"
    ENGINE_ID="decode-0"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-256}"
    ;;
  *)
    echo "usage: serve_one.sh prefill|decode" >&2
    exit 2
    ;;
esac

# --- networking: one node, one NIC, the host IP (not 127.0.0.1) --------------
NIC="${NIC:-$(ip -o -4 route show to default | awk '{print $5; exit}')}"
export HCCL_IF_IP="${HCCL_IF_IP:-$(ip -o -4 addr show dev "$NIC" | awk '{print $4}' | cut -d/ -f1)}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$NIC}"
export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$NIC}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-$NIC}"

export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT="${VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT:-1800}"
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-3600000}"
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# libmooncake*.so must be loadable; Mooncake installs beside its python package.
if [[ -z "${LD_LIBRARY_PATH:-}" || "$LD_LIBRARY_PATH" != *mooncake* ]]; then
  MOONCAKE_LIB="$(python -c 'import mooncake,pathlib;print(pathlib.Path(mooncake.__file__).parent)' 2>/dev/null || true)"
  if [[ -n "$MOONCAKE_LIB" ]]; then
    export LD_LIBRARY_PATH="$MOONCAKE_LIB:${LD_LIBRARY_PATH:-}"
  fi
fi

# use_layerwise=true selects MooncakePushConnector instead of the default pull.
EXTRA_CONFIG='{"use_ascend_direct": true'
if [[ "${USE_LAYERWISE:-0}" == "1" ]]; then
  EXTRA_CONFIG="$EXTRA_CONFIG, \"use_layerwise\": true"
fi
EXTRA_CONFIG="$EXTRA_CONFIG, \"prefill\": {\"dp_size\": 1, \"tp_size\": $TP_SIZE}"
EXTRA_CONFIG="$EXTRA_CONFIG, \"decode\": {\"dp_size\": 1, \"tp_size\": $TP_SIZE}}"

KV_TRANSFER_CONFIG="{\"kv_connector\": \"MooncakeConnectorV2\", \"kv_role\": \"$KV_ROLE\", \
\"kv_port\": \"$KV_PORT\", \"engine_id\": \"$ENGINE_ID\", \"kv_connector_extra_config\": $EXTRA_CONFIG}"

echo "role=$ROLE devices=$ASCEND_RT_VISIBLE_DEVICES port=$PORT kv_port=$KV_PORT ip=$HCCL_IF_IP"
echo "kv-transfer-config: $KV_TRANSFER_CONFIG"
echo

# No --kv-sharing-fast-prefill on purpose: it moves Gemma4 onto the
# self/cross decoder-layer split, a different path from the one the
# connector's owner/borrower handling targets.
exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name kv-sharing-test \
  --tensor-parallel-size "$TP_SIZE" \
  --distributed-executor-backend mp \
  --no-enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --seed 42 \
  --enforce-eager \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  "${@:2}"
