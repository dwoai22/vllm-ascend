#!/usr/bin/env bash
# Shared settings for Mistral Large 4 Prefill/Decode disaggregation on Ascend.
#
# One engine per role. A role may span several nodes: one DP rank per node,
# tensor parallelism inside the node, expert parallelism across the role's
# nodes (the layout validated for large MoE models on Ascend). The first node
# of each role runs the API server; the rest run headless and register with it,
# so each role still exposes a single endpoint for the proxy.
#
#   2 nodes, shrunk model : P = node 0,    D = node 1       (1 node per role)
#   4 nodes, full weights : P = nodes 0-1, D = nodes 2-3     (2 nodes per role)
#
# Pick a preset instead of setting everything by hand:
#
#   export ML4_PRESET=preset_full_4node.sh
#
# Everything below can also be overridden from the environment.

set -o pipefail

# A preset is sourced first; the `${VAR:-default}` defaults below then only
# fill in what it did not set.
if [[ -n "${ML4_PRESET:-}" ]]; then
    # shellcheck source=/dev/null
    source "$(dirname "${BASH_SOURCE[0]}")/$ML4_PRESET"
fi

# --- cluster -------------------------------------------------------------------
# Space-separated node addresses per role, in rank order. The first is the head
# node (API server). P_IP/D_IP are accepted as the single-node spelling.
export P_NODE_IPS="${P_NODE_IPS:-${P_IP:-}}"
export D_NODE_IPS="${D_NODE_IPS:-${D_IP:-}}"
export NIC_NAME="${NIC_NAME:-}" # interface owning this node's address

# --- model ---------------------------------------------------------------------
export MODEL_DIR="${MODEL_DIR:-/root/.cache/ml4-pd-mini}"
# dummy   : reduced model from shrink_ml4_config.py (random weights)
# mistral : real checkpoint, or shrink_ml4_config.py --truncate-only
export LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ml4}"
# 1: stream safetensors lazily. Worth it for the full ~1 TB checkpoint.
export SAFETENSORS_LAZY="${SAFETENSORS_LAZY:-0}"
# Empty: take the KV cache dtype from the checkpoint (both roles must agree,
# so set it on both or neither). fp8_ds_mla selects the packed C8 sparse layout.
export KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

# --- topology / ports ----------------------------------------------------------
export NPUS_PER_NODE="${NPUS_PER_NODE:-8}"
# Tensor parallelism stays inside a node; DP size is the role's node count.
export TP_SIZE="${TP_SIZE:-$NPUS_PER_NODE}"
export P_PORT="${P_PORT:-8000}"
export D_PORT="${D_PORT:-8000}"
export PROXY_PORT="${PROXY_PORT:-9000}"
# DP coordination RPC, on each role's head node. Distinct per role.
export P_DP_RPC_PORT="${P_DP_RPC_PORT:-16600}"
export D_DP_RPC_PORT="${D_DP_RPC_PORT:-16700}"
# MooncakeConnectorV2 port block, per role: with N = DP x TP x PCP x PP,
# worker handshakes take kv_port .. kv_port+N-1 and the scheduler control
# sockets kv_port+N .. kv_port+N+DP-1. DP rank r uses kv_port + r*TP upward.
# P at 30000 with DP2/TP8 -> 30000-30017; D at 30100 -> 30100-30117.
# The transfer engine also binds its own RPC port, so open the peer node
# generously if a firewall sits between the roles.
export P_KV_PORT="${P_KV_PORT:-30000}"
export D_KV_PORT="${D_KV_PORT:-30100}"

# --- engine limits -------------------------------------------------------------
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export P_MAX_NUM_SEQS="${P_MAX_NUM_SEQS:-16}"
export P_MAX_NUM_BATCHED_TOKENS="${P_MAX_NUM_BATCHED_TOKENS:-16384}"
export D_MAX_NUM_SEQS="${D_MAX_NUM_SEQS:-16}"
export D_MAX_NUM_BATCHED_TOKENS="${D_MAX_NUM_BATCHED_TOKENS:-512}"
# 0: Decode runs eager (first bring-up). 1: FULL_DECODE_ONLY graphs on Decode.
export DECODE_GRAPH="${DECODE_GRAPH:-0}"
# 1: Model Runner V2, which PR #21014 enables graphs and DSpark on. 0: V1.
export USE_V2_MODEL_RUNNER="${USE_V2_MODEL_RUNNER:-1}"
# 1: DSpark drafting with the drafter in the checkpoint (depth 3 is fixed by
# the checkpoint). Needs real weights and the V2 runner. Unvalidated here.
export SPEC_DECODE="${SPEC_DECODE:-0}"
export SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-3}"

export LOG_DIR="${LOG_DIR:-$PWD/logs}"

ml4_die() {
    echo "error: $*" >&2
    exit 1
}

# Node addresses of a role, as an array: ml4_role_nodes P -> ML4_NODES
ml4_role_nodes() {
    local role="$1" ips
    case "$role" in
    P) ips="$P_NODE_IPS" ;;
    D) ips="$D_NODE_IPS" ;;
    *) ml4_die "unknown role $role" ;;
    esac
    # shellcheck disable=SC2206 # deliberate word splitting of the IP list
    ML4_NODES=($ips)
    [[ ${#ML4_NODES[@]} -gt 0 ]] || ml4_die "set ${role}_NODE_IPS (addresses of the $role nodes, head first)"
}

ml4_require_cluster() {
    ml4_role_nodes P
    P_HEAD_IP="${ML4_NODES[0]}"
    P_NODE_COUNT=${#ML4_NODES[@]}
    ml4_role_nodes D
    D_HEAD_IP="${ML4_NODES[0]}"
    D_NODE_COUNT=${#ML4_NODES[@]}
    export P_HEAD_IP D_HEAD_IP P_NODE_COUNT D_NODE_COUNT
}

# Process environment for one engine node.
# $1 = role (P|D), $2 = this node's index within the role.
ml4_setup_engine_env() {
    local role="$1" node_index="$2"
    ml4_require_cluster
    ml4_role_nodes "$role"
    ML4_ROLE_NODE_COUNT=${#ML4_NODES[@]}
    [[ "$node_index" =~ ^[0-9]+$ ]] || ml4_die "node index must be a number, got '$node_index'"
    ((node_index < ML4_ROLE_NODE_COUNT)) ||
        ml4_die "node index $node_index is outside the $ML4_ROLE_NODE_COUNT $role node(s)"
    ML4_LOCAL_IP="${ML4_NODES[$node_index]}"
    ML4_HEAD_IP="${ML4_NODES[0]}"
    ML4_NODE_INDEX="$node_index"

    [[ -n "$NIC_NAME" ]] || ml4_die "set NIC_NAME (interface carrying $ML4_LOCAL_IP on this node)"
    [[ -f "$MODEL_DIR/params.json" ]] ||
        ml4_die "$MODEL_DIR/params.json missing: vLLM reads Mistral Large 4 from params.json"
    [[ ! -e "$MODEL_DIR/config.json" ]] ||
        ml4_die "$MODEL_DIR/config.json exists: vLLM would merge it into params.json; remove it"

    unset http_proxy https_proxy ftp_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY
    unset HCCL_INTRA_ROCE_ENABLE

    export HCCL_IF_IP="$ML4_LOCAL_IP"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-1024}"
    export HCCL_EXEC_TIMEOUT=1800
    export HCCL_CONNECT_TIMEOUT=1800
    export HCCL_ALGO="${HCCL_ALGO:-level0:fullmesh}"
    if ((ML4_ROLE_NODE_COUNT > 1)); then
        # Expert parallelism crosses nodes: its all-to-all needs its own buffer.
        # Device memory, charged outside --gpu-memory-utilization.
        export HCCL_BUFFSIZE_EP="${HCCL_BUFFSIZE_EP:-2048}"
    fi

    export VLLM_RPC_TIMEOUT=3600000
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
    # Weight load, MXFP8 requantization and graph capture; past the 600 s default.
    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-6000}"
    export VLLM_USE_V2_MODEL_RUNNER="$USE_V2_MODEL_RUNNER"
    export VLLM_DISABLE_COMPILE_CACHE=1

    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export OPENBLAS_NUM_THREADS=1
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NPUS_PER_NODE - 1)))}"
    export ASCEND_LOCAL_COMM_RES='{"version":"1.3"}'

    mkdir -p "$LOG_DIR"
    export ASCEND_PROCESS_LOG_PATH="$LOG_DIR/ascend"

    echo "role=$role node=$node_index/$ML4_ROLE_NODE_COUNT ip=$ML4_LOCAL_IP head=$ML4_HEAD_IP" \
        "dp=$ML4_ROLE_NODE_COUNT tp=$TP_SIZE ep=$((ML4_ROLE_NODE_COUNT * TP_SIZE))"
}

# Data-parallel flags for this node. A single-node role gets none, so its
# command stays exactly what a one-node deployment ran.
# Call after ml4_setup_engine_env. $1 = DP RPC port.
ml4_dp_args() {
    local rpc_port="$1"
    ML4_DP_ARGS=()
    ((ML4_ROLE_NODE_COUNT > 1)) || return 0
    ML4_DP_ARGS=(
        --data-parallel-size "$ML4_ROLE_NODE_COUNT"
        --data-parallel-size-local 1
        --data-parallel-address "$ML4_HEAD_IP"
        --data-parallel-rpc-port "$rpc_port"
    )
    if ((ML4_NODE_INDEX > 0)); then
        # A worker serves no API: the head's DP load balancer feeds it.
        # --data-parallel-start-rank is deliberately omitted on the head,
        # where it would switch vLLM to external load balancing and expect
        # one endpoint per rank.
        ML4_DP_ARGS+=(--headless --data-parallel-start-rank "$ML4_NODE_INDEX")
    fi
}

# $1 = kv_producer|kv_consumer, $2 = engine_id, $3 = kv_port.
# Needs ml4_require_cluster to have run.
ml4_kv_transfer_config() {
    # MooncakeConnectorV2 exchanges the real layouts during its handshake and
    # ignores these "prefill"/"decode" blocks; they are kept because the
    # published examples carry them and the V1 connector does read them.
    # "fail" surfaces a broken transfer instead of silently recomputing on D.
    cat <<EOF
{"kv_connector": "MooncakeConnectorV2",
 "kv_role": "$1",
 "engine_id": "$2",
 "kv_port": $3,
 "kv_load_failure_policy": "fail",
 "kv_connector_extra_config": {
   "use_ascend_direct": true,
   "prefill": {"dp_size": $P_NODE_COUNT, "tp_size": $TP_SIZE},
   "decode": {"dp_size": $D_NODE_COUNT, "tp_size": $TP_SIZE}}}
EOF
}

# Flags shared by both engines.
ml4_common_serve_args() {
    ML4_SERVE_ARGS=(
        --served-model-name "$SERVED_MODEL_NAME"
        --config-format mistral
        --tokenizer-mode mistral
        --load-format "$LOAD_FORMAT"
        --dtype bfloat16
        --tensor-parallel-size "$TP_SIZE"
        --enable-expert-parallel
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --no-enable-prefix-caching
        --seed 0
        # Block-FP8 weights are selected from the checkpoint's own
        # quantization_config; --quantization is for ModelSlim checkpoints.
    )
    [[ "$SAFETENSORS_LAZY" == 1 ]] && ML4_SERVE_ARGS+=(--safetensors-load-strategy lazy)
    [[ -n "$KV_CACHE_DTYPE" ]] && ML4_SERVE_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    if [[ "$SPEC_DECODE" == 1 ]]; then
        ML4_SERVE_ARGS+=(--speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $SPEC_NUM_TOKENS}")
    fi
    # Vision encoder still in params.json (shrink_ml4_config.py --keep-vision,
    # or the untouched checkpoint): serve text only.
    if python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); sys.exit(not (p.get("vision_encoder") or p.get("multimodal")))' \
        "$MODEL_DIR/params.json"; then
        ML4_SERVE_ARGS+=(--language-model-only)
    fi
}

# $1 = URL, $2 = timeout seconds
ml4_wait_healthy() {
    local url="$1" deadline=$((SECONDS + ${2:-3600}))
    until curl -sf -o /dev/null "$url"; do
        ((SECONDS < deadline)) || ml4_die "timed out waiting for $url"
        sleep 10
    done
    echo "healthy: $url"
}
