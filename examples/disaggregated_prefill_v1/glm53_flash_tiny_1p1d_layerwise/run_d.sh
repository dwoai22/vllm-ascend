#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Decode (kv_consumer) role for the tiny GLM-5.3-Flash 1P1D layerwise deployment.
#
# Usage: bash run_d.sh <visible_devices> <api_port> <dp_size> <dp_rank> \
#                      <dp_address> <dp_rpc_port> <tp_size> [<pcp_size>]
#
# The decode role always runs PCP=1: the layerwise connector rejects a
# consumer with prefill context parallelism, so the 8th argument is accepted
# (for launcher compatibility) but must be 1.
#
# Required environment: MODEL_PATH, LOCAL_IP, NIC_NAME.
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

if [[ "$PCP_SIZE" != "1" ]]; then
  echo "[run_d.sh] decode must run with PCP=1 (got $PCP_SIZE)" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the tiny model directory}"
LOCAL_IP="${LOCAL_IP:?set LOCAL_IP to the communication IP of this node}"
NIC_NAME="${NIC_NAME:?set NIC_NAME to the network interface of this node}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
D_MAX_NUM_SEQS="${D_MAX_NUM_SEQS:-16}"
D_MAX_NUM_BATCHED_TOKENS="${D_MAX_NUM_BATCHED_TOKENS:-256}"
D_GPU_UTIL="${D_GPU_UTIL:-0.85}"
D_KV_PORT="${D_KV_PORT:-36580}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
PREFIX_CACHING="${PREFIX_CACHING:-0}"
MTP_TOKENS="${MTP_TOKENS:-0}"
D_CUDAGRAPH="${D_CUDAGRAPH:-0}"
P_DP="${P_DP:-1}"
P_TP="${P_TP:-1}"
P_PCP="${P_PCP:-1}"

export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"
export HCCL_IF_IP="$LOCAL_IP"
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
if [[ -n "${ASCEND_GLOBAL_RESOURCE_CONFIG:-}" ]]; then
  export ASCEND_GLOBAL_RESOURCE_CONFIG
fi

KV_TRANSFER_CONFIG=$(printf '{"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_consumer", "engine_id": "%s", "kv_port": "%s", "kv_connector_extra_config": {"prefill": {"dp_size": %s, "tp_size": %s, "pcp_size": %s}, "decode": {"dp_size": %s, "tp_size": %s, "pcp_size": 1}}}' \
  "${D_ENGINE_ID:-1}" "$D_KV_PORT" "$P_DP" "$P_TP" "$P_PCP" "$DP_SIZE" "$TP_SIZE")

ARGS=(
  --host 0.0.0.0
  --port "$API_PORT"
  --data-parallel-size "$DP_SIZE"
  --tensor-parallel-size "$TP_SIZE"
  --seed 1024
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype bfloat16
  --load-format "$LOAD_FORMAT"
  --block-size "$BLOCK_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$D_MAX_NUM_SEQS"
  --max-num-batched-tokens "$D_MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$D_GPU_UTIL"
  --trust-remote-code
  --tool-call-parser glm47
  --reasoning-parser glm47
  --enable-auto-tool-choice
  --limit-mm-per-prompt '{"image":1,"video":0}'
  --kv-transfer-config "$KV_TRANSFER_CONFIG"
)

if (( DP_SIZE * TP_SIZE > 1 )); then
  # Expert parallelism spans the DP*TP ranks of this role.
  ARGS+=(--enable-expert-parallel)
fi

if (( DP_SIZE > 1 )); then
  ARGS+=(--data-parallel-rank "$DP_RANK")
  ARGS+=(--data-parallel-address "$DP_ADDRESS")
  ARGS+=(--data-parallel-rpc-port "$DP_RPC_PORT")
fi

if [[ "$PREFIX_CACHING" == "1" ]]; then
  ARGS+=(--enable-prefix-caching)
else
  ARGS+=(--no-enable-prefix-caching)
fi

if [[ "$D_CUDAGRAPH" == "1" ]]; then
  ARGS+=(--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}')
else
  ARGS+=(--enforce-eager)
fi

if (( MTP_TOKENS > 0 )); then
  ARGS+=(--speculative-config "{\"num_speculative_tokens\": ${MTP_TOKENS}, \"method\": \"deepseek_mtp\"}")
fi

echo "[run_d.sh] devices=$DEVICES port=$API_PORT dp=$DP_SIZE/rank=$DP_RANK tp=$TP_SIZE kv_port=$D_KV_PORT"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[run_d.sh] would run: vllm serve %q' "$MODEL_PATH"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi

exec vllm serve "$MODEL_PATH" "${ARGS[@]}"
