#!/usr/bin/env bash
# Decode engine (kv_consumer). Run on every D node.
#
#   bash run_decode.sh [node_index] [extra vllm args...]
#
# node_index is this node's position in D_NODE_IPS (default 0). Index 0 serves
# the API; the others run headless. Start index 0 first.
# DECODE_GRAPH=1 captures FULL_DECODE_ONLY graphs; the default stays eager.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NODE_INDEX="${1:-${NODE_INDEX:-0}}"
if (($# > 0)); then
    # Extra vllm args therefore require the node index first.
    shift
fi

ml4_setup_engine_env D "$NODE_INDEX"
ml4_common_serve_args
ml4_dp_args "$D_DP_RPC_PORT"

KV_TRANSFER_CONFIG="$(ml4_kv_transfer_config kv_consumer "ml4-decode-$NODE_INDEX" "$D_KV_PORT")"

if [[ "$DECODE_GRAPH" == 1 ]]; then
    capture_sizes="$(python3 -c "import sys; n = int(sys.argv[1]); print([s for s in (1, 2, 4, 8, 16, 32, 64) if s <= n])" "$D_MAX_NUM_SEQS")"
    GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": $capture_sizes}")
else
    GRAPH_ARGS=(--enforce-eager)
fi

set -x
vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$D_PORT" \
    "${ML4_SERVE_ARGS[@]}" \
    ${ML4_DP_ARGS[@]+"${ML4_DP_ARGS[@]}"} \
    --max-num-seqs "$D_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$D_MAX_NUM_BATCHED_TOKENS" \
    "${GRAPH_ARGS[@]}" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    "$@" 2>&1 | tee "$LOG_DIR/decode-$NODE_INDEX.log"
