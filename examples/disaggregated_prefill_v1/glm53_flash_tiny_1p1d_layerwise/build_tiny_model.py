#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a shrunk ("tiny") GLM-5.3-Flash model directory.

The result keeps every architectural feature of GLM-5.3-Flash that the
layerwise PD path exercises -- KDA linear attention (recurrent/Mamba-style
state), DeepSeek sparse attention (MLA + compressed indexer + circular
compressor tail), dense and sparse (MoE) MLPs, an MTP draft layer and the
vision tower -- while reducing depth, expert count and vision depth so one
node can host both a Prefill and a Decode engine.

Accuracy is explicitly out of scope: the generated directory carries no
weights and is meant to be served with ``--load-format dummy`` (random
weights). Tokenizer/processor files are copied from a reference checkpoint
when one is available locally.

Example:
    python3 build_tiny_model.py \
        --out /root/.cache/glm53-flash-tiny \
        --reference-dir /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-Flash-w8a8
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

# Baseline = the published GLM-5.3-Flash config (A3 w8a8 / A5 w8a8-mxfp8 share
# the same architecture fields; the quantization format lives outside
# config.json in the ModelSlim checkpoints).
BASELINE: dict = {
    "architectures": ["Glm5NextForConditionalGeneration"],
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dtype": "bfloat16",
        "eos_token_id": [154820, 154827, 154829],
        "first_k_dense_replace": 3,
        "hc_eps": 1e-06,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "head_dim": 0,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "index_head_dim": 128,
        "index_kpool": 4,
        "index_kpool_always_select_tail": True,
        "index_kpool_compress": True,
        "index_n_heads": 32,
        "index_topk": 2048,
        "index_share_for_mtp_iteration": True,
        "indexer_rope_interleave": True,
        "indexer_types": ["full"] * 45,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "layer_types": [],  # filled in below
        "linear_attn_config": {
            "num_heads": 64,
            "gate_lower_bound": -5.0,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "kda_layers": [],  # filled in below
            "full_attn_layers": [],  # filled in below
        },
        "max_position_embeddings": 1048576,
        "mhc": True,
        "mla_use_nope": True,
        "mlp_layer_types": [],  # filled in below
        "model_type": "glm5_next_text",
        "moe_intermediate_size": 2048,
        "moe_router_dtype": "float32",
        "n_group": 1,
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "norm_topk_prob": True,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 45,
        "num_key_value_heads": 64,
        "num_nextn_predict_layers": 1,
        "output_router_logits": False,
        "pad_token_id": 154820,
        "q_lora_rank": 1536,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "rms_norm_eps": 1e-05,
        "routed_scaling_factor": 2.5,
        "router_aux_loss_coef": 0.001,
        "scoring_func": "sigmoid",
        "swiglu_limit": 10.0,
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "use_cache": True,
        "v_head_dim": 256,
        "vocab_size": 154880,
    },
    "vision_config": {
        "attention_bias": True,
        "attention_dropout": 0.0,
        "depth": 24,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "image_size": 448,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "glm5_next_vision",
        "num_heads": 16,
        "out_hidden_size": 4096,
        "patch_size": 14,
        "projection_intermediate_size": 10240,
        "rms_norm_eps": 1e-05,
        "spatial_merge_size": 2,
        "swiglu_limit": 10.0,
        "temporal_patch_size": 2,
    },
    "image_token_id": 154854,
    "video_token_id": 154855,
    "image_start_token_id": 154830,
    "image_end_token_id": 154831,
    "video_start_token_id": 154832,
    "video_end_token_id": 154833,
    "tie_word_embeddings": False,
    "model_type": "glm5_next",
    "transformers_version": "5.16.0",
}

DSA = "deepseek_sparse_attention"
LINEAR = "linear_attention"

# Files that must not be copied from the reference checkpoint: weights (we run
# with random weights), the weight index, the ModelSlim quantization
# description (the tiny model is served in BF16) and config.json (regenerated).
SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".msgpack", ".h5", ".ckpt")
SKIP_NAMES = (
    "config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
SKIP_PREFIXES = ("quant_model_description",)


def build_config(args: argparse.Namespace, baseline: dict) -> dict:
    cfg = copy.deepcopy(baseline)
    text = cfg["text_config"]
    layers = args.layers
    period = args.pattern_period

    # Attention interleave: every `period`-th layer is a DeepSeek sparse
    # attention (MLA + indexer) layer, the rest are KDA linear-attention
    # layers -- the same phase as the released 45-layer model (indices
    # 3, 7, 11, ...).
    layer_types = [DSA if (i % period) == (period - 1) else LINEAR for i in range(layers)]
    text["layer_types"] = layer_types
    text["indexer_types"] = ["full"] * layers
    text["num_hidden_layers"] = layers
    text["first_k_dense_replace"] = args.dense_layers
    text["mlp_layer_types"] = ["dense" if i < args.dense_layers else "sparse" for i in range(layers)]
    text["linear_attn_config"]["kda_layers"] = [i for i, t in enumerate(layer_types) if t == LINEAR]
    text["linear_attn_config"]["full_attn_layers"] = [i for i, t in enumerate(layer_types) if t == DSA]

    text["n_routed_experts"] = args.experts
    text["num_experts_per_tok"] = args.experts_per_tok
    if args.index_topk is not None:
        text["index_topk"] = args.index_topk
    if args.linear_heads is not None:
        text["linear_attn_config"]["num_heads"] = args.linear_heads
    if args.max_position_embeddings is not None:
        text["max_position_embeddings"] = args.max_position_embeddings
    if args.mtp_layers is not None:
        text["num_nextn_predict_layers"] = args.mtp_layers

    cfg["vision_config"]["depth"] = args.vision_depth

    # The tiny model is served in BF16 with random weights: KDA requires BF16
    # activations, so pin the dtype and drop any quantization description that
    # came from the reference checkpoint.
    text["dtype"] = "bfloat16"
    cfg.pop("quantization_config", None)
    text.pop("quantization_config", None)
    return cfg


def validate(cfg: dict, tp: int, pcp: int, block_size: int) -> list[str]:
    """Check the invariants the vLLM-Ascend GLM5Next implementation enforces."""
    problems: list[str] = []
    text = cfg["text_config"]
    layers = text["num_hidden_layers"]
    layer_types = text["layer_types"]
    linear = text["linear_attn_config"]

    def require(cond: bool, message: str) -> None:
        if not cond:
            problems.append(message)

    require(len(layer_types) == layers, f"layer_types has {len(layer_types)} entries, expected {layers}")
    require(
        len(text["mlp_layer_types"]) == layers,
        f"mlp_layer_types has {len(text['mlp_layer_types'])} entries, expected {layers}",
    )
    require(
        len(text["indexer_types"]) == layers,
        f"indexer_types has {len(text['indexer_types'])} entries, expected {layers}",
    )
    require(set(layer_types) <= {DSA, LINEAR}, f"unexpected layer_types values: {set(layer_types)}")
    kda = [i for i, t in enumerate(layer_types) if t == LINEAR]
    dsa = [i for i, t in enumerate(layer_types) if t == DSA]
    require(linear["kda_layers"] == kda, "linear_attn_config.kda_layers does not match layer_types")
    require(linear["full_attn_layers"] == dsa, "linear_attn_config.full_attn_layers does not match layer_types")
    require(bool(dsa), "at least one deepseek_sparse_attention layer is required (indexer/kpool caches)")
    require(bool(kda), "at least one linear_attention layer is required (recurrent state transfer)")
    require(0 <= text["first_k_dense_replace"] < layers, "first_k_dense_replace must be inside the layer range")

    # MoE
    require(text["n_routed_experts"] >= text["num_experts_per_tok"], "n_routed_experts < num_experts_per_tok")
    require(text["n_routed_experts"] % text["n_group"] == 0, "n_routed_experts must be divisible by n_group")
    require(text["hidden_act"] == "silu", "only silu is supported by Glm5NextMLP")

    # KDA kernel requirements (vllm_ascend/models/glm5next/kda.py)
    require(linear["head_dim"] == 128, "GLM AscendC KDA requires linear head_dim=128")
    require(2 <= linear["short_conv_kernel_size"] <= 4, "KDA causal conv kernel width must be in [2, 4]")
    require(
        linear["short_conv_kernel_size"] == 4,
        "KDA requires conv kernel width 4 when MTP speculative decoding is enabled",
    )
    require(-5.0 <= linear["gate_lower_bound"] < 0.0, "KDA requires gate_lower_bound in [-5, 0)")
    require(text["dtype"] == "bfloat16", "KDA requires BF16 activations")

    # Sparse indexer / kpool (vllm_ascend/models/glm5next/attention.py, kv_cache.py)
    require(text["index_kpool_always_select_tail"] is True, "sparse indexer requires index_kpool_always_select_tail")
    require(text["index_kpool"] >= 1, "index_kpool must be >= 1")
    require(
        block_size % text["index_kpool"] == 0,
        f"KV block size ({block_size}) must be divisible by index_kpool ({text['index_kpool']})",
    )

    # Parallelism: heads must divide over the head-sharded axes.
    heads_axis = tp * pcp
    require(
        text["num_attention_heads"] % tp == 0,
        f"num_attention_heads ({text['num_attention_heads']}) must be divisible by TP ({tp})",
    )
    require(
        linear["num_heads"] % heads_axis == 0,
        f"linear_attn_config.num_heads ({linear['num_heads']}) must be divisible by TP*PCP ({heads_axis})",
    )
    require(tp == 1 or pcp == 1, "layerwise recurrent-state transfer requires TP=1 or PCP=1 on each role")
    return problems


def estimate_bf16_bytes(cfg: dict) -> dict:
    """Rough parameter/byte estimate (BF16). Informational only."""
    text = cfg["text_config"]
    hidden = text["hidden_size"]
    vocab = text["vocab_size"]
    layer_types = text["layer_types"]
    mlp_types = text["mlp_layer_types"]
    linear = text["linear_attn_config"]

    kda_proj = linear["num_heads"] * linear["head_dim"]
    kda_per_layer = (
        hidden * (3 * kda_proj + linear["num_heads"] + 2 * linear["head_dim"])  # fused in_proj
        + 2 * linear["head_dim"] * kda_proj  # f_b_proj, g_b_proj
        + 3 * kda_proj * linear["short_conv_kernel_size"]  # q/k/v conv1d
        + kda_proj * hidden  # o_proj
    )
    qk_head_dim = text["qk_nope_head_dim"] + text["qk_rope_head_dim"]
    mla_per_layer = (
        hidden * text["q_lora_rank"]
        + text["q_lora_rank"] * text["num_attention_heads"] * qk_head_dim
        + hidden * (text["kv_lora_rank"] + text["qk_rope_head_dim"])
        + text["kv_lora_rank"] * text["num_attention_heads"] * (text["qk_nope_head_dim"] + text["v_head_dim"])
        + text["num_attention_heads"] * text["v_head_dim"] * hidden
    )
    indexer_per_layer = (
        text["index_head_dim"] * hidden
        + text["q_lora_rank"] * text["index_n_heads"] * text["index_head_dim"]
        + hidden * (text["index_head_dim"] + text["index_n_heads"])
    )
    moe_per_layer = (
        text["n_routed_experts"] * 3 * hidden * text["moe_intermediate_size"]
        + (text["n_shared_experts"] or 0) * 3 * hidden * text["moe_intermediate_size"]
        + hidden * text["n_routed_experts"]
    )
    dense_per_layer = 3 * hidden * text["intermediate_size"]

    params = 2 * vocab * hidden  # embed_tokens + lm_head (untied)
    for idx, kind in enumerate(layer_types):
        params += kda_per_layer if kind == LINEAR else mla_per_layer + indexer_per_layer
        params += moe_per_layer if mlp_types[idx] == "sparse" else dense_per_layer
    mtp = text.get("num_nextn_predict_layers") or 0
    # Each MTP layer is an MLA(+indexer) decoder layer with a sparse MLP, an
    # eh_proj and its own shared head.
    params += mtp * (mla_per_layer + indexer_per_layer + moe_per_layer + 2 * hidden * hidden + vocab * hidden)

    vision = cfg["vision_config"]
    vision_params = (
        vision["depth"]
        * (4 * vision["hidden_size"] * vision["hidden_size"] + 3 * vision["hidden_size"] * vision["intermediate_size"])
        + 2 * vision["out_hidden_size"] * vision["projection_intermediate_size"]
    )
    params += vision_params
    return {"params": params, "bf16_bytes": 2 * params}


def copy_reference_files(reference_dir: Path, out_dir: Path) -> list[str]:
    copied: list[str] = []
    for src in sorted(reference_dir.iterdir()):
        if src.is_dir() or src.name in SKIP_NAMES:
            continue
        if src.suffix in SKIP_SUFFIXES or src.name.startswith(SKIP_PREFIXES):
            continue
        if src.suffix == ".py":
            # transformers >= 5.16 implements glm5_next natively and
            # vLLM-Ascend provides the runtime model, so remote code is not
            # needed and an auto_map would point at the full-size layout.
            continue
        shutil.copy2(src, out_dir / src.name)
        copied.append(src.name)
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output model directory")
    parser.add_argument(
        "--reference-dir",
        help="full-size GLM-5.3-Flash checkpoint directory; tokenizer/processor files are copied from it",
    )
    parser.add_argument("--reference-config", help="config.json to shrink (defaults to the built-in baseline)")
    parser.add_argument("--layers", type=int, default=8, help="num_hidden_layers (default: 8)")
    parser.add_argument(
        "--pattern-period",
        type=int,
        default=4,
        help="one deepseek_sparse_attention layer every N layers, at index N-1 (default: 4, as released)",
    )
    parser.add_argument("--dense-layers", type=int, default=3, help="first_k_dense_replace (default: 3)")
    parser.add_argument("--experts", type=int, default=32, help="n_routed_experts (default: 32)")
    parser.add_argument("--experts-per-tok", type=int, default=8, help="num_experts_per_tok (default: 8)")
    parser.add_argument("--vision-depth", type=int, default=4, help="vision_config.depth (default: 4)")
    parser.add_argument("--linear-heads", type=int, help="linear_attn_config.num_heads (default: keep 64)")
    parser.add_argument("--index-topk", type=int, help="index_topk (default: keep 2048)")
    parser.add_argument("--mtp-layers", type=int, help="num_nextn_predict_layers (default: keep 1)")
    parser.add_argument("--max-position-embeddings", type=int, help="override max_position_embeddings")
    parser.add_argument("--check-tp", type=int, default=1, help="validate head divisibility for this TP (default: 1)")
    parser.add_argument("--check-pcp", type=int, default=1, help="validate head divisibility for this PCP (default: 1)")
    parser.add_argument("--block-size", type=int, default=128, help="KV block size used at serve time (default: 128)")
    parser.add_argument("--print-only", action="store_true", help="print the config and exit without writing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = BASELINE
    if args.reference_config:
        baseline = json.loads(Path(args.reference_config).read_text(encoding="utf-8"))
    elif args.reference_dir and (Path(args.reference_dir) / "config.json").is_file():
        baseline = json.loads((Path(args.reference_dir) / "config.json").read_text(encoding="utf-8"))

    cfg = build_config(args, baseline)
    problems = validate(cfg, args.check_tp, args.check_pcp, args.block_size)
    estimate = estimate_bf16_bytes(cfg)

    text = cfg["text_config"]
    print(
        "tiny GLM-5.3-Flash: "
        f"{text['num_hidden_layers']} layers "
        f"({len(text['linear_attn_config']['kda_layers'])} KDA, "
        f"{len(text['linear_attn_config']['full_attn_layers'])} DSA), "
        f"{text['n_routed_experts']} routed experts, "
        f"{text['num_nextn_predict_layers']} MTP layer(s), "
        f"vision depth {cfg['vision_config']['depth']}"
    )
    print(f"estimated weights: {estimate['params'] / 1e9:.2f}B params, {estimate['bf16_bytes'] / 2**30:.1f} GiB BF16")

    if problems:
        print("\nconfig validation failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("config validation: OK")

    if args.print_only:
        print(json.dumps(cfg, indent=2))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'config.json'}")

    if args.reference_dir:
        reference_dir = Path(args.reference_dir)
        if not reference_dir.is_dir():
            print(f"reference directory not found: {reference_dir}")
            return 1
        copied = copy_reference_files(reference_dir, out_dir)
        print(f"copied {len(copied)} tokenizer/processor file(s) from {reference_dir}: {', '.join(copied)}")
        if not any(name.startswith("tokenizer") for name in copied):
            print("WARNING: no tokenizer file was copied; vLLM will not be able to serve this directory")
    else:
        print(
            "no --reference-dir given: copy tokenizer_config.json, tokenizer.json, "
            "chat_template.jinja, preprocessor_config.json and generation_config.json "
            "from the real checkpoint into this directory before serving"
        )
    print("serve this directory with --load-format dummy (random weights, no accuracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
