#!/usr/bin/env bash
# Prefill engine (kv_producer), eager. Run on every P node.
#
#   bash run_prefill.sh [node_index] [extra vllm args...]
#
# node_index is this node's position in P_NODE_IPS (default 0). Index 0 serves
# the API; the others run headless. Start index 0 first.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

NODE_INDEX="${1:-${NODE_INDEX:-0}}"
if (($# > 0)); then
    # Extra vllm args therefore require the node index first.
    shift
fi

ml4_setup_engine_env P "$NODE_INDEX"
ml4_common_serve_args
ml4_dp_args "$P_DP_RPC_PORT"

KV_TRANSFER_CONFIG="$(ml4_kv_transfer_config kv_producer "ml4-prefill-$NODE_INDEX" "$P_KV_PORT")"

set -x
vllm serve "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$P_PORT" \
    "${ML4_SERVE_ARGS[@]}" \
    ${ML4_DP_ARGS[@]+"${ML4_DP_ARGS[@]}"} \
    --max-num-seqs "$P_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$P_MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    "$@" 2>&1 | tee "$LOG_DIR/prefill-$NODE_INDEX.log"
