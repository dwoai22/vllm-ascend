#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Prefill (kv_producer) role for the tiny GLM-5.3-Flash 1P1D layerwise deployment.
#
# Usage: bash run_p.sh <visible_devices> <api_port> <dp_size> <dp_rank> \
#                      <dp_address> <dp_rpc_port> <tp_size> <pcp_size>
#
# Required environment: MODEL_PATH, LOCAL_IP, NIC_NAME.
# Optional environment: see the defaults below and README.md.
set -euo pipefail

if (( $# < 7 )); then
  echo "usage: bash $(basename "$0") <visible_devices> <api_port> <dp_size> <dp_rank> <dp_address> <dp_rpc_port> <tp_size> [<pcp_size>]" >&2
  exit 2
fi

DEVICES="$1"
API_PORT="$2"
DP_SIZE="$3"
DP_RANK="$4"
DP_ADDRESS="$5"
DP_RPC_PORT="$6"
TP_SIZE="$7"
PCP_SIZE="${8:-1}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the tiny model directory}"
LOCAL_IP="${LOCAL_IP:?set LOCAL_IP to the communication IP of this node}"
NIC_NAME="${NIC_NAME:?set NIC_NAME to the network interface of this node}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
P_MAX_NUM_SEQS="${P_MAX_NUM_SEQS:-8}"
P_MAX_NUM_BATCHED_TOKENS="${P_MAX_NUM_BATCHED_TOKENS:-8192}"
P_GPU_UTIL="${P_GPU_UTIL:-0.85}"
P_KV_PORT="${P_KV_PORT:-36680}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
PREFIX_CACHING="${PREFIX_CACHING:-0}"
MTP_TOKENS="${MTP_TOKENS:-0}"
D_DP="${D_DP:-1}"
D_TP="${D_TP:-1}"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
# A3 supernodes whose device RoCE ports are down talk over HCCS. Export
# ASCEND_GLOBAL_RESOURCE_CONFIG before calling this script to override.
if [[ -n "${ASCEND_GLOBAL_RESOURCE_CONFIG:-}" ]]; then
  export ASCEND_GLOBAL_RESOURCE_CONFIG
fi

# The layerwise connector learns the real peer topology from each request's
# kv_transfer_params; prefill/decode sizes are kept here for parity with the
# documented Mooncake examples and for easier log correlation.
KV_TRANSFER_CONFIG=$(printf '{"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_producer", "engine_id": "%s", "kv_port": "%s", "kv_connector_extra_config": {"prefill": {"dp_size": %s, "tp_size": %s, "pcp_size": %s}, "decode": {"dp_size": %s, "tp_size": %s, "pcp_size": 1}}}' \
  "${P_ENGINE_ID:-0}" "$P_KV_PORT" "$DP_SIZE" "$TP_SIZE" "$PCP_SIZE" "$D_DP" "$D_TP")

ARGS=(
  --host 0.0.0.0
  --port "$API_PORT"
  --data-parallel-size "$DP_SIZE"
  --tensor-parallel-size "$TP_SIZE"
  --prefill-context-parallel-size "$PCP_SIZE"
  --seed 1024
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype bfloat16
  --load-format "$LOAD_FORMAT"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$P_MAX_NUM_SEQS"
  --max-num-batched-tokens "$P_MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$P_GPU_UTIL"
  --trust-remote-code
  --tool-call-parser glm47
  --reasoning-parser glm47
  --enable-auto-tool-choice
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --enforce-eager
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

if (( DP_SIZE * TP_SIZE * PCP_SIZE > 1 )); then
  # Expert parallelism spans the DP*TP*PCP ranks of this role.
  ARGS+=(--enable-expert-parallel)
fi

if (( DP_SIZE > 1 )); then
  # One API server per DP rank: external DP load balancing.
  ARGS+=(--data-parallel-rank "$DP_RANK")
  ARGS+=(--data-parallel-address "$DP_ADDRESS")
  ARGS+=(--data-parallel-rpc-port "$DP_RPC_PORT")
fi

if [[ "$PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi

if (( MTP_TOKENS > 0 )); then
  ARGS+=(--speculative-config "{\"num_speculative_tokens\": ${MTP_TOKENS}, \"method\": \"deepseek_mtp\"}")
fi

echo "[run_p.sh] devices=$DEVICES port=$API_PORT dp=$DP_SIZE/rank=$DP_RANK tp=$TP_SIZE pcp=$PCP_SIZE kv_port=$P_KV_PORT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[run_p.sh] would run: vllm serve %q' "$MODEL_PATH"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

exec vllm serve "$MODEL_PATH" "${ARGS[@]}"
