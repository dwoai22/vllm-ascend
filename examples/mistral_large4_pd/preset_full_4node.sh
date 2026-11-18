#!/usr/bin/env bash
# shellcheck disable=SC2034  # these are consumed by env.sh after it sources this file
# Full-weight Mistral Large 4, 1P1D on four 950DT nodes (96 GB x 8 each).
#
#   export ML4_PRESET=preset_full_4node.sh
#
# Then set P_NODE_IPS, D_NODE_IPS and NIC_NAME for your cluster, and MODEL_DIR
# to the real checkpoint. Sourced by env.sh before its own defaults.
#
# Why two nodes per role: the FP8 checkpoint is ~0.99 TiB of weights, which is
# ~127 GiB per NPU on a single 8-NPU node — more than the 96 GB each NPU has.
# Two nodes per role shard the routed experts 16 ways (EP16) and land at
# ~64 GiB per NPU, leaving ~11 GiB per NPU for the KV cache at utilization
# 0.85. Three nodes for one role and one for the other does not work: the
# single node still cannot hold the weights.
#
# Layout per role: one DP rank per node, TP8 inside the node, EP16 across both.
# TP16 spanning both nodes is the alternative (near-identical memory) but puts
# an attention all-reduce on the inter-node link every layer.

# The real checkpoint directory: params.json plus consolidated*.safetensors.
# Must be the same path on all four nodes, and must NOT contain config.json.
MODEL_DIR="${MODEL_DIR:-/root/.cache/modelscope/hub/models/mistralai/Mistral-Large-4-1T-A52B}"
LOAD_FORMAT=mistral
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ml4}"
SAFETENSORS_LAZY=1

# P = first two nodes, D = last two. Head node of each role listed first.
P_NODE_IPS="${P_NODE_IPS:-}"
D_NODE_IPS="${D_NODE_IPS:-}"

# 65536 x ~41 KiB/token = ~2.6 GiB of KV for one full-length request, against
# ~11 GiB per NPU at utilization 0.85. Raise --max-num-seqs or the context
# only as far as that budget allows; the engine reports the KV cache size it
# settled on at startup.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Prefill: the upstream recipe's batch budget. Decode: one token per sequence
# per step, with headroom.
P_MAX_NUM_SEQS="${P_MAX_NUM_SEQS:-16}"
P_MAX_NUM_BATCHED_TOKENS="${P_MAX_NUM_BATCHED_TOKENS:-8192}"
D_MAX_NUM_SEQS="${D_MAX_NUM_SEQS:-16}"
D_MAX_NUM_BATCHED_TOKENS="${D_MAX_NUM_BATCHED_TOKENS:-512}"

# Start eager on both roles; turn graphs on for Decode once P->D works.
DECODE_GRAPH="${DECODE_GRAPH:-0}"
