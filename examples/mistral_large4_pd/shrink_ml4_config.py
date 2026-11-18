#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shrink a Mistral Large 4 config so 1P1D fits on two 8-NPU nodes.

The full FP8 checkpoint (~1.06 TB of weights) cannot be held by one 8-NPU
node, and 1P1D gives each role exactly one node. This tool writes a *reduced*
model directory: full layer widths, fewer layers, dummy weights. Accuracy is
meaningless; the point is to exercise the P->D KV transfer path with the same
per-layer cache layout the real model has.

What vLLM actually reads
------------------------
vLLM loads Mistral Large 4 from ``params.json`` (``--config-format mistral``),
not from the HF ``config.json``: ``MistralLarge4ForConditionalGeneration`` is
not a registered vLLM architecture. So this tool rewrites ``params.json``.

Two parser behaviours shape the output directory:

* ``MistralConfigParser`` merges every key of a ``config.json`` found next to
  ``params.json`` into the config as *defaults*. A 72-layer ``config.json``
  would leak 72-entry per-layer lists into the reduced model, so the output
  directory never contains ``config.json``. The shrunk HF config is written as
  ``hf_config.shrunk.json``, for reference only.
* Without ``dtype`` in ``params.json`` the parser infers it from
  ``consolidated*.safetensors``. The output has no weights, so ``dtype`` is
  written explicitly.

What the reduced layout keeps
-----------------------------
Every per-layer feature that changes the KV cache layout or the Ascend
adapter's behaviour, re-derived for the new depth from the original pattern:

* NoPE layers: the original ``nope`` pattern, expanded the way vLLM expands it.
* KV sharing: the same fraction of NoPE layers borrow the latent of one
  earlier NoPE layer (72 layers: 39,43,...,71 -> 35).
* Shared indexer: the same ``shared,shared,full,full`` unit, starting at the
  same relative depth (72 layers: from layer 17).
* The dense first layer(s) and the routed-expert count (128 by default).

Run with ``--layers 72`` to check that the derivation reproduces the original
layout; the tool prints whether it does on every run.

Usage::

    python shrink_ml4_config.py --src /models/Mistral-Large-4 \
        --dst /models/ml4-pd-mini --layers 16
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

# Files the Mistral tokenizer (--tokenizer-mode mistral) may need. They are
# small, so they are copied rather than symlinked: the reduced directory then
# works on a node that does not mount the real checkpoint.
TOKENIZER_GLOBS = (
    "tekken*.json",
    "tokenizer.model*",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "SYSTEM_PROMPT.txt",
)

# Unprefixed / prefixed spellings of the shared-indexer key inside `dsa`
# (vllm.transformers_utils.configs.mistral._DSA_KEY_ALIASES).
DSA_SHARING_KEYS = ("sharing", "index_sharing")
VISION_KEYS = ("vision_encoder", "multimodal")
FULL, SHARED = "full", "shared"
MIN_LAYERS = 8
LAYER_GRANULARITY = 4
MAX_SHARING_PERIOD = 8
DEFAULT_DTYPE = "bfloat16"
DEFAULT_EXPERT_PARALLEL_SIZE = 8


def expand_layer_pattern(pattern: list[Any], num_layers: int) -> list[Any]:
    """Same expansion as vLLM's ``_expand_layer_pattern``: repeat, then cut."""
    if not pattern:
        raise ValueError("Per-layer patterns must not be empty.")
    return (pattern * math.ceil(num_layers / len(pattern)))[:num_layers]


def nope_flags(params: dict, num_layers: int) -> list[bool]:
    nope = params.get("nope")
    if not nope:
        return [False] * num_layers
    return expand_layer_pattern([v == "nope" for v in nope.split(",")], num_layers)


def sharing_key(dsa: dict) -> str | None:
    present = [key for key in DSA_SHARING_KEYS if key in dsa]
    if len(present) > 1:
        raise ValueError(f"params.json dsa has both {present}; keep one.")
    return present[0] if present else None


def index_sharing(params: dict, num_layers: int) -> list[str]:
    dsa = params.get("dsa") or {}
    key = sharing_key(dsa)
    if key is None or not dsa[key]:
        return [FULL] * num_layers
    return expand_layer_pattern(dsa[key].split(","), num_layers)


def kv_sources(params: dict, num_layers: int) -> list[int]:
    kv_sharing = params.get("kv_sharing") or {}
    serialized = kv_sharing.get("source_layers_str")
    if not serialized:
        return list(range(num_layers))
    sources = [int(v) for v in serialized.split(",")]
    if len(sources) != num_layers:
        raise ValueError(f"kv_sharing.source_layers_str has {len(sources)} entries for {num_layers} layers")
    return sources


def smallest_period(sequence: list[str]) -> int:
    for period in range(1, min(MAX_SHARING_PERIOD, len(sequence)) + 1):
        if all(sequence[i] == sequence[i % period] for i in range(len(sequence))):
            return period
    raise NotImplementedError(
        f"The shared-indexer pattern after its full prefix is not periodic within "
        f"{MAX_SHARING_PERIOD} layers: {sequence}. Edit dsa.sharing by hand."
    )


def derive_index_sharing(original: list[str], num_layers: int) -> list[str]:
    """Same full prefix ratio and repeating unit as the original, at a new depth."""
    if SHARED not in original:
        return [FULL] * num_layers
    original_layers = len(original)
    first_shared = original.index(SHARED)
    tail = original[first_shared:]
    period = smallest_period(tail)
    unit = tail[:period]
    start = max(1, round(first_shared * num_layers / original_layers))
    while start % period != first_shared % period:
        start += 1
    if start >= num_layers:
        raise ValueError(f"{num_layers} layers leave no room for a shared-indexer layer; use more layers.")
    return [FULL] * start + expand_layer_pattern(unit, num_layers - start)


def derive_kv_sources(
    original_sources: list[int],
    original_nope: list[bool],
    nope: list[bool],
) -> list[int]:
    """Same fraction of NoPE layers borrowing one earlier NoPE layer's latent."""
    num_layers = len(nope)
    sharers = [i for i, src in enumerate(original_sources) if src != i]
    if not sharers:
        return list(range(num_layers))
    source_layers = {original_sources[i] for i in sharers}
    original_nope_layers = [i for i, flag in enumerate(original_nope) if flag]
    if len(source_layers) != 1 or not all(original_nope[i] for i in sharers):
        raise NotImplementedError(
            "Only the Mistral Large 4 KV-sharing shape is derived automatically "
            "(NoPE layers borrowing one NoPE source). Pass --no-kv-sharing or edit "
            "kv_sharing.source_layers_str by hand."
        )
    nope_layers = [i for i, flag in enumerate(nope) if flag]
    num_sharers = max(1, round(len(sharers) * len(nope_layers) / len(original_nope_layers)))
    if len(nope_layers) < num_sharers + 1:
        raise ValueError(f"{num_layers} layers have {len(nope_layers)} NoPE layers: too few to keep KV sharing.")
    source = nope_layers[-num_sharers - 1]
    sources = list(range(num_layers))
    for layer in nope_layers[-num_sharers:]:
        sources[layer] = source
    return sources


def check_layout(sources: list[int], nope: list[bool], sharing: list[str]) -> None:
    """The checks vLLM's ``_remap_mistral_large_4_args`` makes, plus ML4's own shape."""
    for layer, source in enumerate(sources):
        if not (source <= layer and sources[source] == source):
            raise AssertionError(f"layer {layer} must share KV with an earlier, non-sharing layer")
        if nope[layer] != nope[source]:
            raise AssertionError(f"layer {layer} and its KV source {source} disagree on NoPE")
    if sharing[0] != FULL:
        raise AssertionError("layer 0 must run its own indexer")
    if not set(sharing) <= {FULL, SHARED}:
        raise AssertionError(f"unknown index sharing values {set(sharing)}")


def remap_target_layers(serialized: str, original_layers: int, num_layers: int) -> str:
    """Scale DSpark ``target_activation_layers`` into range, unique and ascending."""
    layers = [int(v) for v in serialized.split(",") if v.strip()]
    scale = (num_layers - 1) / max(1, original_layers - 1)
    remapped: list[int] = []
    for layer in layers:
        candidate = min(num_layers - 1, round(layer * scale))
        if remapped and candidate <= remapped[-1]:
            candidate = remapped[-1] + 1
        if candidate >= num_layers:
            break
        remapped.append(candidate)
    return ",".join(str(layer) for layer in remapped)


def layer_table(sources: list[int], nope: list[bool], sharing: list[str], first_dense: int) -> str:
    rows = ["layer  rope   kv_source        indexer  mlp"]
    for i, (src, is_nope, kind) in enumerate(zip(sources, nope, sharing)):
        kv = "own" if src == i else f"borrows {src}"
        rows.append(
            f"{i:>5}  {'NoPE' if is_nope else 'RoPE':<5}  {kv:<15}  {kind:<7}  {'dense' if i < first_dense else 'moe'}"
        )
    return "\n".join(rows)


def estimate_weights_gib_per_npu(params: dict, parallel_size: int) -> float:
    """Rough weight bytes per NPU for one engine at TP = EP = ``parallel_size``.

    Block-FP8 weights become MXFP8 on Ascend 950: one byte per weight plus one
    E8M0 scale per 32. Embedding, LM head and the attention projections that
    the checkpoint keeps in BF16 are counted at two bytes. Activations, KV
    cache and communication buffers are not included.
    """
    fp8 = 1 + 1 / 32 if params.get("quantization") else 2.0
    bf16 = 2.0
    dim, layers = int(params["dim"]), int(params["n_layers"])
    moe = params.get("moe") or {}
    first_dense = int(moe.get("first_k_dense_replace", 0))
    moe_layers = layers - first_dense
    expert = 3 * dim * int(moe.get("expert_hidden_dim", 0))
    routed = moe_layers * int(moe.get("num_experts", 0)) * expert * fp8 / parallel_size
    shared = moe_layers * int(moe.get("num_shared_experts", 0)) * expert * fp8 / parallel_size
    dense = first_dense * 3 * dim * int(params.get("hidden_dim", 0)) * fp8 / parallel_size
    heads, lora = int(params["n_heads"]), int(params.get("kv_lora_rank", 0))
    nope_dim, rope_dim = int(params.get("qk_nope_head_dim", 0)), int(params.get("qk_rope_head_dim", 0))
    v_dim = int(params.get("v_head_dim", 0))
    attention = (
        dim * heads * (nope_dim + rope_dim) * fp8 / parallel_size  # q_proj
        + heads * v_dim * dim * fp8 / parallel_size  # o_proj
        + lora * heads * (nope_dim + v_dim) * bf16 / parallel_size  # kv_b_proj
        + dim * (lora + rope_dim) * bf16  # kv_a_proj_with_mqa, replicated
    )
    dsa = _normalize_dsa(params.get("dsa") or {})
    indexer = (
        dim * int(dsa.get("n_heads", 0)) * int(dsa.get("head_dim", 0)) * fp8 + dim * int(dsa.get("head_dim", 0)) * bf16
    )
    embeddings = 2 * int(params["vocab_size"]) * dim * bf16 / parallel_size
    total = routed + shared + dense + layers * (attention + indexer) + embeddings
    return total / 2**30


def _normalize_dsa(dsa: dict) -> dict:
    aliases = {"index_n_heads": "n_heads", "index_head_dim": "head_dim"}
    return {aliases.get(key, key): value for key, value in dsa.items()}


def shrink_params(params: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    params = copy.deepcopy(params)
    original_layers = int(params["n_layers"])
    num_layers = args.layers
    moe = params.get("moe") or {}
    first_dense = int(moe.get("first_k_dense_replace", 0))

    original_nope = nope_flags(params, original_layers)
    original_sources = kv_sources(params, original_layers)
    original_sharing = index_sharing(params, original_layers)

    # Self-check: the derivation must reproduce the original at its own depth.
    reproduces = (
        derive_index_sharing(original_sharing, original_layers) == original_sharing
        and derive_kv_sources(original_sources, original_nope, original_nope) == original_sources
    )

    nope = nope_flags(params, num_layers)  # `nope` stays as-is; vLLM expands it.
    if args.truncate_only:
        # Real weights: layer i of the reduced model *is* checkpoint layer i, so
        # every per-layer pattern must stay the original's first N entries.
        sharing = original_sharing[:num_layers]
        sources = original_sources[:num_layers]
    else:
        sharing = derive_index_sharing(original_sharing, num_layers)
        if args.no_kv_sharing:
            sources = list(range(num_layers))
        else:
            sources = derive_kv_sources(original_sources, original_nope, nope)
    check_layout(sources, nope, sharing)

    params["n_layers"] = num_layers
    if any(src != i for i, src in enumerate(sources)):
        params["kv_sharing"] = {"source_layers_str": ",".join(str(s) for s in sources)}
    else:
        params.pop("kv_sharing", None)

    if not args.truncate_only:
        dsa = params["dsa"]
        key = sharing_key(dsa) or "sharing"
        dsa[key] = ",".join(sharing)

    original_experts = int(moe["num_experts"])
    num_experts = args.experts or original_experts
    top_k = int(moe.get("num_experts_per_tok", 1))
    if num_experts < top_k:
        raise ValueError(f"--experts {num_experts} is below num_experts_per_tok {top_k}")
    if num_experts % args.ep_size:
        raise ValueError(f"--experts {num_experts} must divide by the expert-parallel size {args.ep_size}")
    moe["num_experts"] = num_experts

    mtp = params.get("mtp")
    if mtp and mtp.get("target_activation_layers"):
        mtp["target_activation_layers"] = remap_target_layers(
            mtp["target_activation_layers"], original_layers, num_layers
        )

    dropped_vision = []
    if not args.keep_vision:
        for vision_key in VISION_KEYS:
            if params.pop(vision_key, None) is not None:
                dropped_vision.append(vision_key)

    if params.get("dtype") is None:
        params["dtype"] = DEFAULT_DTYPE

    report = {
        "original_layers": original_layers,
        "layers": num_layers,
        "original_experts": original_experts,
        "experts": num_experts,
        "first_k_dense_replace": first_dense,
        "nope_layers": [i for i, flag in enumerate(nope) if flag],
        "kv_sharing_layers": {i: src for i, src in enumerate(sources) if src != i},
        "shared_indexer_layers": [i for i, kind in enumerate(sharing) if kind == SHARED],
        "dspark_target_activation_layers": (mtp or {}).get("target_activation_layers"),
        "dropped_vision_keys": dropped_vision,
        "derivation_reproduces_original": reproduces,
        "mode": "truncate-only (real weights)" if args.truncate_only else "derived layout (dummy weights)",
        "weights_gib_per_npu": round(estimate_weights_gib_per_npu(params, args.ep_size), 1),
        "full_model_weights_gib_per_npu": round(
            estimate_weights_gib_per_npu(
                {**params, "n_layers": original_layers, "moe": {**moe, "num_experts": original_experts}}, args.ep_size
            ),
            1,
        ),
        "layer_table": layer_table(sources, nope, sharing, first_dense),
    }
    return params, report


def shrink_hf_config(hf_config: dict, report: dict, layout: dict) -> dict:
    """The same layout in HF ``config.json`` spelling. Reference only."""
    hf_config = copy.deepcopy(hf_config)
    text = hf_config.get("text_config", hf_config)
    num_layers = report["layers"]
    text["num_hidden_layers"] = num_layers
    text["n_routed_experts"] = report["experts"]
    if "layer_types" in text:
        text["layer_types"] = expand_layer_pattern(text["layer_types"], num_layers)
    text["mlp_layer_types"] = ["dense" if i < report["first_k_dense_replace"] else "sparse" for i in range(num_layers)]
    text["indexer_types"] = layout["sharing"]
    text["kv_source_layers"] = layout["sources"]
    text["no_rope_layers"] = [0 if flag else 1 for flag in layout["nope"]]
    return hf_config


def validate_with_vllm(params: dict, report: dict) -> str:
    """Run vLLM's own Mistral config adapter on the result, if vLLM is importable."""
    try:
        from vllm.transformers_utils.configs.mistral import adapt_config_dict
    except Exception as exc:  # noqa: BLE001 - any import failure means "skip"
        return f"skipped (vLLM not importable here: {type(exc).__name__}: {exc})"
    config = adapt_config_dict(copy.deepcopy(params), defaults={})
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    expected_sources = list(range(report["layers"]))
    for layer, src in report["kv_sharing_layers"].items():
        expected_sources[int(layer)] = src
    checks = {
        "num_hidden_layers": text.num_hidden_layers == report["layers"],
        "n_routed_experts": text.n_routed_experts == report["experts"],
        "kv_source_layers": list(text.kv_source_layers) == expected_sources,
        "index_sharing": [i for i, k in enumerate(text.index_sharing) if k == SHARED]
        == report["shared_indexer_layers"],
        "nope_layer_flags": [i for i, f in enumerate(text.nope_layer_flags) if f] == report["nope_layers"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"vLLM adapter disagrees with the shrunk layout on: {failed}")
    return f"ok (architectures={getattr(config, 'architectures', None)}, model_type={text.model_type})"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="Real checkpoint dir (must contain params.json).")
    parser.add_argument("--dst", type=Path, required=True, help="Output dir for the reduced model (created).")
    parser.add_argument("--layers", type=int, default=16, help="Decoder layers to keep (multiple of 4, >= 8).")
    parser.add_argument("--experts", type=int, default=None, help="Routed experts (default: keep the original).")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=DEFAULT_EXPERT_PARALLEL_SIZE,
        help="Expert-parallel size the experts must divide by (8 = one node with --enable-expert-parallel).",
    )
    parser.add_argument("--no-kv-sharing", action="store_true", help="Give every layer its own latent KV.")
    parser.add_argument(
        "--truncate-only",
        action="store_true",
        help=(
            "Keep the first N checkpoint layers unchanged and symlink the real weights, for "
            "--load-format mistral. Outputs then depend on the KV cache, so P/D can be compared "
            "against a local run. No KV-sharing layer exists below layer 39."
        ),
    )
    parser.add_argument(
        "--keep-vision", action="store_true", help="Keep the vision encoder (then serve with --language-model-only)."
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --dst.")
    args = parser.parse_args(argv)
    if args.layers < MIN_LAYERS or args.layers % LAYER_GRANULARITY:
        parser.error(f"--layers must be a multiple of {LAYER_GRANULARITY} and at least {MIN_LAYERS}")
    if args.truncate_only and (args.experts or args.no_kv_sharing):
        parser.error("--truncate-only keeps the checkpoint as-is: it cannot change --experts or KV sharing")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    params_path = args.src / "params.json"
    if not params_path.is_file():
        print(
            f"error: {params_path} not found. vLLM loads Mistral Large 4 from params.json "
            "(--config-format mistral); the HF config.json architecture is not registered in vLLM.",
            file=sys.stderr,
        )
        return 2
    params = json.loads(params_path.read_text())
    if args.layers > int(params["n_layers"]):
        print(f"error: --layers {args.layers} exceeds the original {params['n_layers']}", file=sys.stderr)
        return 2
    try:
        shrunk, report = shrink_params(params, args)
    except (ValueError, NotImplementedError, AssertionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dst.exists():
        if not args.force:
            print(f"error: {args.dst} exists; pass --force to overwrite.", file=sys.stderr)
            return 2
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True)
    (args.dst / "params.json").write_text(json.dumps(shrunk, indent=2) + "\n")

    copied = []
    for pattern in TOKENIZER_GLOBS:
        for path in sorted(glob.glob(str(args.src / pattern))):
            shutil.copy2(path, args.dst)
            copied.append(Path(path).name)
    if not any(name.startswith("tekken") or name.startswith("tokenizer") for name in copied):
        print("warning: no tokenizer file copied; pass --tokenizer <real checkpoint> when serving.", file=sys.stderr)

    linked = []
    if args.truncate_only:
        for path in sorted(glob.glob(str(args.src / "consolidated*.safetensors*"))):
            (args.dst / Path(path).name).symlink_to(Path(path).resolve())
            linked.append(Path(path).name)
        if not linked:
            print(
                "warning: no consolidated*.safetensors found to link; --load-format mistral will fail.", file=sys.stderr
            )

    hf_config_path = args.src / "config.json"
    if hf_config_path.is_file():
        layout = {
            "sources": [report["kv_sharing_layers"].get(i, i) for i in range(report["layers"])],
            "nope": [i in report["nope_layers"] for i in range(report["layers"])],
            "sharing": [SHARED if i in report["shared_indexer_layers"] else FULL for i in range(report["layers"])],
        }
        hf_shrunk = shrink_hf_config(json.loads(hf_config_path.read_text()), report, layout)
        # Deliberately not named config.json: vLLM would merge it into params.json.
        (args.dst / "hf_config.shrunk.json").write_text(json.dumps(hf_shrunk, indent=2) + "\n")

    report["vllm_adapter_check"] = validate_with_vllm(shrunk, report)
    report["tokenizer_files"] = copied
    report["linked_weight_files"] = len(linked)
    (args.dst / "shrink_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {args.dst / 'params.json'}")
    print(
        f"layers {report['original_layers']} -> {report['layers']}, "
        f"experts {report['original_experts']} -> {report['experts']}"
    )
    print(f"NoPE layers: {report['nope_layers']}")
    print(f"KV-sharing layers (layer: source): {report['kv_sharing_layers'] or 'none'}")
    print(f"shared-indexer layers: {report['shared_indexer_layers']}")
    print(f"derivation reproduces the original layout at full depth: {report['derivation_reproduces_original']}")
    print(f"vLLM adapter check: {report['vllm_adapter_check']}")
    print(f"tokenizer files copied: {copied or 'none'}")
    print(f"mode: {report['mode']}" + (f", {len(linked)} weight files linked" if args.truncate_only else ""))
    print(
        f"weights per NPU at TP=EP={args.ep_size}: ~{report['weights_gib_per_npu']} GiB "
        f"(full model: ~{report['full_model_weights_gib_per_npu']} GiB)"
    )
    print(report["layer_table"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
