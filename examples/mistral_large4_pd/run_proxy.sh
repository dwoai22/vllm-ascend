#!/usr/bin/env bash
# Node 0 (or any host that reaches both engines): the PD proxy.
# Waits for both engines, then routes /v1/completions and /v1/chat/completions
# to P (max_tokens=1) and on to D with the returned kv_transfer_params.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
ml4_require_cluster
# One endpoint per role: each role's head node fronts its own DP ranks.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

if [[ -z "${VLLM_ASCEND_SRC:-}" ]]; then
    # Editable installs resolve to the checkout; the release image ships it here.
    VLLM_ASCEND_SRC="$(python3 -c 'import os, vllm_ascend; print(os.path.dirname(os.path.dirname(vllm_ascend.__file__)))' 2>/dev/null || true)"
fi
PROXY="$VLLM_ASCEND_SRC/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py"
[[ -f "$PROXY" ]] || PROXY=/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py
[[ -f "$PROXY" ]] || ml4_die "proxy script not found; set VLLM_ASCEND_SRC to your vllm-ascend checkout"

ml4_wait_healthy "http://$P_HEAD_IP:$P_PORT/health"
ml4_wait_healthy "http://$D_HEAD_IP:$D_PORT/health"

mkdir -p "$LOG_DIR"
set -x
python3 "$PROXY" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --prefiller-hosts "$P_HEAD_IP" --prefiller-ports "$P_PORT" \
    --decoder-hosts "$D_HEAD_IP" --decoder-ports "$D_PORT" \
    "$@" 2>&1 | tee "$LOG_DIR/proxy.log"
