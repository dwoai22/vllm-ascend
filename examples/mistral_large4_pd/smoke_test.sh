#!/usr/bin/env bash
# Any host that reaches both nodes: one request through the proxy, then pd_check.py.
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh
ml4_require_cluster
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

PROXY_URL="http://$P_HEAD_IP:$PROXY_PORT"
ml4_wait_healthy "http://$P_HEAD_IP:$P_PORT/health" 600
ml4_wait_healthy "http://$D_HEAD_IP:$D_PORT/health" 600
ml4_wait_healthy "$PROXY_URL/healthcheck" 600

echo "--- one completion through the proxy"
curl -sS "$PROXY_URL/v1/completions" -H "Content-Type: application/json" -d "{
    \"model\": \"$SERVED_MODEL_NAME\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 16,
    \"temperature\": 0
}"
echo

echo "--- pd_check.py"
CHECK_ARGS=(
    --proxy "$PROXY_URL"
    --prefill "http://$P_HEAD_IP:$P_PORT"
    --decode "http://$D_HEAD_IP:$D_PORT"
    --model "$SERVED_MODEL_NAME"
)
if [[ "$LOAD_FORMAT" != dummy ]]; then
    # Real weights (--truncate-only): outputs depend on the KV, so P->D must match P alone.
    CHECK_ARGS+=(--expect-match)
fi
python3 pd_check.py "${CHECK_ARGS[@]}"
