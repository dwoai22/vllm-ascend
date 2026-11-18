#!/usr/bin/env bash
# shellcheck disable=SC2034  # these are consumed by env.sh after it sources this file
# Reduced Mistral Large 4, 1P1D on two nodes: one node per role, TP8 + EP8.
#
#   export ML4_PRESET=preset_shrunk_2node.sh
#
# This is the bring-up configuration: dummy weights, no accuracy. Build the
# model directory first (on both nodes, same path):
#
#   python3 shrink_ml4_config.py --src <real checkpoint> --dst /root/.cache/ml4-pd-mini
#
# For a reduced model whose output does depend on the KV cache, build it with
# --truncate-only and set LOAD_FORMAT=mistral.

MODEL_DIR="${MODEL_DIR:-/root/.cache/ml4-pd-mini}"
LOAD_FORMAT="${LOAD_FORMAT:-dummy}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ml4-mini}"

# One node per role: no data-parallel flags are emitted.
P_NODE_IPS="${P_NODE_IPS:-${P_IP:-}}"
D_NODE_IPS="${D_NODE_IPS:-${D_IP:-}}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DECODE_GRAPH="${DECODE_GRAPH:-0}"
