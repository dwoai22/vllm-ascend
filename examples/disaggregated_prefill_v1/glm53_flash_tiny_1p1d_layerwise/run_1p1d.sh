#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start a single-node 1P1D layerwise (MooncakeLayerwiseConnector) deployment of
# the tiny GLM-5.3-Flash model: the Prefill engine(s), the Decode engine(s) and
# the layerwise load-balancing proxy, all on this host.
#
#   MODEL_PATH=/root/.cache/glm53-flash-tiny \
#   LOCAL_IP=<this node's IP> NIC_NAME=<interface> \
#   TOPOLOGY=min bash run_1p1d.sh
#
# Set DRY_RUN=1 to only validate the topology and print the plan.
#
# TOPOLOGY presets (override any value with the matching environment variable):
#   min    P: 1 device  (DP1/TP1/PCP1)   D: 1 device  (DP1/TP1)     - 2 devices
#   8dev   P: 4 devices (DP1/TP1/PCP4)   D: 4 devices (DP4/TP1)     - 8 devices
#   16dev  P: 8 devices (DP1/TP1/PCP8)   D: 8 devices (DP4/TP2)     - 16 devices
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the tiny model directory}"
LOCAL_IP="${LOCAL_IP:?set LOCAL_IP to the communication IP of this node}"
NIC_NAME="${NIC_NAME:?set NIC_NAME to the network interface of this node}"
export MODEL_PATH LOCAL_IP NIC_NAME

TOPOLOGY="${TOPOLOGY:-min}"
case "$TOPOLOGY" in
  min)
    : "${P_DP:=1}" "${P_TP:=1}" "${P_PCP:=1}" "${P_DEV_START:=0}"
    : "${D_DP:=1}" "${D_TP:=1}" "${D_DEV_START:=1}"
    ;;
  8dev)
    : "${P_DP:=1}" "${P_TP:=1}" "${P_PCP:=4}" "${P_DEV_START:=0}"
    : "${D_DP:=4}" "${D_TP:=1}" "${D_DEV_START:=4}"
    ;;
  16dev)
    : "${P_DP:=1}" "${P_TP:=1}" "${P_PCP:=8}" "${P_DEV_START:=0}"
    : "${D_DP:=4}" "${D_TP:=2}" "${D_DEV_START:=8}"
    ;;
  custom)
    : "${P_DP:?set P_DP for TOPOLOGY=custom}" "${P_TP:?set P_TP}" "${P_PCP:?set P_PCP}" "${P_DEV_START:=0}"
    : "${D_DP:?set D_DP}" "${D_TP:?set D_TP}" "${D_DEV_START:?set D_DEV_START}"
    ;;
  *)
    echo "unknown TOPOLOGY '$TOPOLOGY' (expected min, 8dev, 16dev or custom)" >&2
    exit 1
    ;;
esac
export P_DP P_TP P_PCP D_DP D_TP

P_API_PORT0="${P_API_PORT0:-9081}"
D_API_PORT0="${D_API_PORT0:-9900}"
P_DP_RPC_PORT="${P_DP_RPC_PORT:-12325}"
D_DP_RPC_PORT="${D_DP_RPC_PORT:-12425}"
P_KV_PORT="${P_KV_PORT:-36680}"
D_KV_PORT="${D_KV_PORT:-36580}"
PROXY_PORT="${PROXY_PORT:-8081}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"
START_PROXY="${START_PROXY:-1}"
export P_KV_PORT D_KV_PORT

P_DEV_PER_RANK=$((P_TP * P_PCP))
D_DEV_PER_RANK=$((D_TP))
P_WORLD=$((P_DP * P_TP * P_PCP))
D_WORLD=$((D_DP * D_TP))
P_HEAD_SHARDS=$((P_TP * P_PCP))
D_HEAD_SHARDS=$((D_TP))

fail() { echo "[run_1p1d.sh] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Topology rules enforced by the layerwise component transfer
# (vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py)
# ---------------------------------------------------------------------------
[[ $P_TP -eq 1 || $P_PCP -eq 1 ]] || fail "prefill must use TP=1 or PCP=1 (recurrent state is sharded over one axis)"
[[ $P_DP -ge 1 && $D_DP -ge 1 ]] || fail "DP sizes must be >= 1"
if (( P_HEAD_SHARDS > D_HEAD_SHARDS )); then
  (( P_HEAD_SHARDS % D_HEAD_SHARDS == 0 )) || fail "P TP*PCP ($P_HEAD_SHARDS) and D TP ($D_HEAD_SHARDS) need an integer ratio"
else
  (( D_HEAD_SHARDS % P_HEAD_SHARDS == 0 )) || fail "D TP ($D_HEAD_SHARDS) and P TP*PCP ($P_HEAD_SHARDS) need an integer ratio"
fi

# Device ranges must not overlap.
P_DEV_END=$((P_DEV_START + P_DP * P_DEV_PER_RANK - 1))
D_DEV_END=$((D_DEV_START + D_DP * D_DEV_PER_RANK - 1))
if (( P_DEV_START <= D_DEV_END && D_DEV_START <= P_DEV_END )); then
  fail "prefill devices [$P_DEV_START-$P_DEV_END] overlap decode devices [$D_DEV_START-$D_DEV_END]"
fi

# KV handshake port ranges must not overlap: the producer uses
# kv_port + dp_rank*PCP*TP + pcp_rank*TP + tp_rank, the consumer
# kv_port + dp_rank*TP + tp_rank.
P_KV_END=$((P_KV_PORT + P_WORLD - 1))
D_KV_END=$((D_KV_PORT + D_WORLD - 1))
if (( P_KV_PORT <= D_KV_END && D_KV_PORT <= P_KV_END )); then
  fail "P kv_port range [$P_KV_PORT-$P_KV_END] overlaps D kv_port range [$D_KV_PORT-$D_KV_END]"
fi
[[ "$P_DP_RPC_PORT" != "$D_DP_RPC_PORT" ]] || fail "P and D need different --data-parallel-rpc-port values on one host"

# Model-shape rules, read from the tiny config.
if [[ -f "$MODEL_PATH/config.json" ]]; then
  read -r LINEAR_HEADS ATTN_HEADS EXPERTS KPOOL <<<"$(python3 -c '
import json, sys
text = json.load(open(sys.argv[1]))["text_config"]
print(text["linear_attn_config"]["num_heads"], text["num_attention_heads"],
      text["n_routed_experts"], text["index_kpool"])
' "$MODEL_PATH/config.json")"
  (( LINEAR_HEADS % P_HEAD_SHARDS == 0 )) || fail "linear_attn num_heads ($LINEAR_HEADS) not divisible by P TP*PCP ($P_HEAD_SHARDS)"
  (( LINEAR_HEADS % D_HEAD_SHARDS == 0 )) || fail "linear_attn num_heads ($LINEAR_HEADS) not divisible by D TP ($D_HEAD_SHARDS)"
  (( ATTN_HEADS % P_TP == 0 )) || fail "num_attention_heads ($ATTN_HEADS) not divisible by P TP ($P_TP)"
  (( ATTN_HEADS % D_TP == 0 )) || fail "num_attention_heads ($ATTN_HEADS) not divisible by D TP ($D_TP)"
  (( EXPERTS % P_WORLD == 0 )) || fail "n_routed_experts ($EXPERTS) not divisible by the prefill EP size ($P_WORLD)"
  (( EXPERTS % D_WORLD == 0 )) || fail "n_routed_experts ($EXPERTS) not divisible by the decode EP size ($D_WORLD)"
  (( ${BLOCK_SIZE:-128} % KPOOL == 0 )) || fail "block size (${BLOCK_SIZE:-128}) not divisible by index_kpool ($KPOOL)"
else
  echo "[run_1p1d.sh] WARNING: $MODEL_PATH/config.json not found, skipping model-shape checks"
fi

devices_for() { # start count -> "a,b,c"
  local start="$1" count="$2" i out=""
  for ((i = 0; i < count; i++)); do
    out+="$((start + i))"
    (( i + 1 < count )) && out+=","
  done
  printf '%s' "$out"
}

wait_health() { # host port label [path]
  local host="$1" port="$2" label="$3" path="${4:-/health}" waited=0
  echo "[run_1p1d.sh] waiting for $label on $host:$port"
  while ! curl --fail --silent --max-time 2 "http://$host:$port$path" >/dev/null; do
    sleep 5
    waited=$((waited + 5))
    if (( waited >= HEALTH_TIMEOUT )); then
      fail "$label did not become healthy within ${HEALTH_TIMEOUT}s, see $LOG_DIR"
    fi
  done
  echo "[run_1p1d.sh] $label is up"
}

echo "[run_1p1d.sh] topology=$TOPOLOGY"
echo "[run_1p1d.sh]   P: DP$P_DP/TP$P_TP/PCP$P_PCP devices $P_DEV_START-$P_DEV_END api $P_API_PORT0.. kv $P_KV_PORT.."
echo "[run_1p1d.sh]   D: DP$D_DP/TP$D_TP/PCP1 devices $D_DEV_START-$D_DEV_END api $D_API_PORT0.. kv $D_KV_PORT.."
echo "[run_1p1d.sh]   logs: $LOG_DIR"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[run_1p1d.sh] DRY_RUN=1: topology checks passed, nothing was started"
  exit 0
fi

mkdir -p "$LOG_DIR"
PID_FILE="$LOG_DIR/pids"
: >"$PID_FILE"

# Decode first: the layerwise proxy sends each request to a decode engine,
# which allocates blocks and hands its own host/port back to the prefill engine.
D_PORTS=()
for ((rank = 0; rank < D_DP; rank++)); do
  dev=$(devices_for $((D_DEV_START + rank * D_DEV_PER_RANK)) "$D_DEV_PER_RANK")
  port=$((D_API_PORT0 + rank))
  D_PORTS+=("$port")
  nohup bash "$SCRIPT_DIR/run_d.sh" "$dev" "$port" "$D_DP" "$rank" "$LOCAL_IP" "$D_DP_RPC_PORT" "$D_TP" 1 \
    >"$LOG_DIR/d_rank${rank}.log" 2>&1 &
  echo "$!" >>"$PID_FILE"
  echo "[run_1p1d.sh] started D rank $rank (devices $dev, port $port, pid $!)"
done
for port in "${D_PORTS[@]}"; do wait_health "$LOCAL_IP" "$port" "decode :$port"; done

P_PORTS=()
for ((rank = 0; rank < P_DP; rank++)); do
  dev=$(devices_for $((P_DEV_START + rank * P_DEV_PER_RANK)) "$P_DEV_PER_RANK")
  port=$((P_API_PORT0 + rank))
  P_PORTS+=("$port")
  nohup bash "$SCRIPT_DIR/run_p.sh" "$dev" "$port" "$P_DP" "$rank" "$LOCAL_IP" "$P_DP_RPC_PORT" "$P_TP" "$P_PCP" \
    >"$LOG_DIR/p_rank${rank}.log" 2>&1 &
  echo "$!" >>"$PID_FILE"
  echo "[run_1p1d.sh] started P rank $rank (devices $dev, port $port, pid $!)"
done
for port in "${P_PORTS[@]}"; do wait_health "$LOCAL_IP" "$port" "prefill :$port"; done

if [[ "$START_PROXY" == "1" ]]; then
  PROXY_PORT="$PROXY_PORT" \
  P_PORTS="${P_PORTS[*]}" D_PORTS="${D_PORTS[*]}" \
    nohup bash "$SCRIPT_DIR/run_proxy.sh" >"$LOG_DIR/proxy.log" 2>&1 &
  echo "$!" >>"$PID_FILE"
  echo "[run_1p1d.sh] started layerwise proxy on $LOCAL_IP:$PROXY_PORT (pid $!)"
  wait_health "$LOCAL_IP" "$PROXY_PORT" "proxy :$PROXY_PORT" "/healthcheck"
fi

echo
echo "[run_1p1d.sh] 1P1D is ready. Send a request with:"
echo "  PROXY_URL=http://$LOCAL_IP:$PROXY_PORT bash $SCRIPT_DIR/smoke_test.sh"
echo "[run_1p1d.sh] stop everything with: bash $SCRIPT_DIR/stop_1p1d.sh"
