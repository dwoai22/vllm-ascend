#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Send one request through the layerwise PD proxy.
#
# The tiny model runs with random weights, so the generated text is
# meaningless. A successful run means the request was prefilled on the
# producer, its KV cache (MLA + compressed indexer + circular compressor tail
# + KDA recurrent state) was pushed layer by layer to the consumer, and the
# consumer decoded the requested number of tokens.
set -euo pipefail

PROXY_URL="${PROXY_URL:?set PROXY_URL, e.g. http://127.0.0.1:8081}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm}"
MAX_TOKENS="${MAX_TOKENS:-16}"
PROMPT="${PROMPT:-Summarize the architecture of a prefill-decode disaggregated inference system.}"
export SERVED_MODEL_NAME MAX_TOKENS PROMPT

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "[smoke_test.sh] /v1/completions"
curl --fail --silent --show-error "$PROXY_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["SERVED_MODEL_NAME"],
    "prompt": os.environ["PROMPT"],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
')" | python3 -m json.tool

echo
echo "[smoke_test.sh] /v1/chat/completions"
curl --fail --silent --show-error "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["SERVED_MODEL_NAME"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": 0,
}))
')" | python3 -m json.tool
