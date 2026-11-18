#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight check: does this model actually exercise cross-layer KV sharing?

Reads a local (or cached) HF ``config.json`` and reproduces the mapping that
``vllm/model_executor/models/gemma4.py`` builds at init time: the last
``num_kv_shared_layers`` layers read the KV cache of the last *non-shared*
layer of the same attention type (``full_attention`` / ``sliding_attention``).

The owners it prints are the layers MooncakeConnectorV2 publishes and
transfers; the borrowers are the layers that must be accepted as bound to an
owner's cache. Run it before starting any server -- it needs no NPU.

    python check_kv_sharing.py /models/gemma-4-E2B-it
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ATTN_SUFFIX = "self_attn.attn"


def load_text_config(model_path: str) -> tuple[dict, str]:
    """Return the text config and the attention-layer name prefix."""
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        sys.exit(
            f"No config.json under {model_path!r}. Pass the local snapshot directory "
            "(e.g. ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/<rev>)."
        )
    config = json.loads(config_path.read_text())
    # Multimodal checkpoints nest the decoder under text_config; vLLM then
    # names its attention layers language_model.model.layers.<i>.self_attn.attn.
    if "text_config" in config:
        return config["text_config"], "language_model.model.layers"
    return config, "model.layers"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_path", help="Local directory holding the model's config.json")
    args = parser.parse_args()

    config, prefix = load_text_config(args.model_path)

    num_layers = config["num_hidden_layers"]
    num_shared = config.get("num_kv_shared_layers", 0)
    layer_types = config.get("layer_types") or ["full_attention"] * num_layers
    sliding_window = config.get("sliding_window")

    print(f"model path          : {args.model_path}")
    print(f"layer name prefix   : {prefix}.<i>.{ATTN_SUFFIX}")
    print(f"num_hidden_layers   : {num_layers}")
    print(f"num_kv_shared_layers: {num_shared}")
    print(f"sliding_window      : {sliding_window}")
    print(f"attention types     : {sorted(set(layer_types))}")

    if num_shared <= 0:
        print(
            "\nRESULT: this checkpoint declares no KV-sharing layers, so it will NOT\n"
            "exercise the KVCacheLayerTopology borrower path. Pick a checkpoint whose\n"
            "config sets num_kv_shared_layers > 0."
        )
        return 1

    first_shared = num_layers - num_shared
    prev_layers = layer_types[:first_shared]

    print(f"\nowner layers (published and transferred): 0..{first_shared - 1}")
    print(f"KV-sharing layers (borrowers)           : {first_shared}..{num_layers - 1}\n")
    print(f"{'borrower':>8}  {'type':<18}  ->  owner")

    owners_used = set()
    for idx in range(first_shared, num_layers):
        layer_type = layer_types[idx]
        if layer_type not in prev_layers:
            print(f"{idx:>8}  {layer_type:<18}  ->  NONE  <-- no earlier layer of this type")
            continue
        owner_idx = len(prev_layers) - 1 - prev_layers[::-1].index(layer_type)
        owners_used.add(owner_idx)
        print(f"{idx:>8}  {layer_type:<18}  ->  {owner_idx}")

    groups = sorted({layer_types[i] for i in range(first_shared, num_layers)})
    print(
        f"\nRESULT: {num_shared} borrower layer(s) across {len(groups)} attention type(s) "
        f"{groups}, borrowing from owner layer(s) {sorted(owners_used)}."
    )
    print(
        "Each borrower must appear in the connector's borrower_to_owner map and in no\n"
        "published layer list. Confirm against the server log lines quoted in README.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
