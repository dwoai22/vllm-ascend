#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Start the layerwise load-balancing proxy for the tiny GLM-5.3-Flash 1P1D
# deployment.
#
# The layerwise proxy must be used with MooncakeLayerwiseConnector: it sends
# each request to a decode engine first, receives that engine's KV transfer
# parameters on /v1/metaserver, and only then forwards the request to a
# prefill engine, which pushes KV layer by layer.
#
#   LOCAL_IP=<node ip> P_PORTS="9081" D_PORTS="9900 9901" bash run_proxy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOCAL_IP="${LOCAL_IP:?set LOCAL_IP to the IP of this node; the proxy refuses a wildcard host}"
PROXY_PORT="${PROXY_PORT:-8081}"
P_HOST="${P_HOST:-$LOCAL_IP}"
D_HOST="${D_HOST:-$LOCAL_IP}"
read -r -a p_ports <<<"${P_PORTS:-9081}"
read -r -a d_ports <<<"${D_PORTS:-9900}"

PROXY_SCRIPT="${PROXY_SCRIPT:-}"
if [[ -z "$PROXY_SCRIPT" ]]; then
  for candidate in \
    "$SCRIPT_DIR/../load_balance_proxy_layerwise_server_example.py" \
    "/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py"; do
    if [[ -f "$candidate" ]]; then
      PROXY_SCRIPT="$candidate"
      break
    fi
  done
fi
[[ -f "$PROXY_SCRIPT" ]] || {
  echo "layerwise proxy script not found; set PROXY_SCRIPT to load_balance_proxy_layerwise_server_example.py" >&2
  exit 1
}

p_hosts=(); for _ in "${p_ports[@]}"; do p_hosts+=("$P_HOST"); done
d_hosts=(); for _ in "${d_ports[@]}"; do d_hosts+=("$D_HOST"); done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "[run_proxy.sh] $PROXY_SCRIPT on $LOCAL_IP:$PROXY_PORT -> P ${p_ports[*]} / D ${d_ports[*]}"
exec python3 "$PROXY_SCRIPT" \
  --host "$LOCAL_IP" \
  --port "$PROXY_PORT" \
  --prefiller-hosts "${p_hosts[@]}" \
  --prefiller-ports "${p_ports[@]}" \
  --decoder-hosts "${d_hosts[@]}" \
  --decoder-ports "${d_ports[@]}"
