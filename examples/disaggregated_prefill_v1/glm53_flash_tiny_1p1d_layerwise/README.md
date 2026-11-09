# Tiny GLM-5.3-Flash: single-node 1P1D with the layerwise connector

A shrunk GLM-5.3-Flash ("tiny") model plus run scripts for a **1P1D
Prefill/Decode disaggregated deployment on one node** using
`MooncakeLayerwiseConnector`, the layer-by-layer KV push connector.

The published tutorial
([GLM-5.3-Flash](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/GLM5.3-Flash.html#53-1p1d-pd-disaggregated-deployment))
only covers two-node PD with `MooncakeConnectorV2`. This directory covers the
layerwise variant, which for this model needs the component-aware transfer
added by [#20625](https://github.com/vllm-project/vllm-ascend/pull/20625) on
top of `releases/v0.32.0rc`, and shrinks the model so a single node can host
both roles.

**Accuracy is explicitly out of scope.** The tiny model carries no weights and
is served with `--load-format dummy` (random values), so generated text is
meaningless. What it does exercise is the full GLM-5.3-Flash cache topology on
both sides of the PD boundary.

## 1 Requirements

| Component | Requirement |
| --- | --- |
| vLLM-Ascend | `releases/v0.32.0rc` **with [#20625](https://github.com/vllm-project/vllm-ascend/pull/20625) merged** (4 commits adding component-aware Mooncake layerwise cache transfer). Without it the layerwise connector rejects this model: it asserts that merged per-layer cache components share one KV cache group and one block-size scale, which GLM-5.3-Flash violates (MLA + compressed indexer + circular compressor tail + KDA state). |
| vLLM | the upstream revision pinned by that vLLM-Ascend revision |
| Transformers | a version that knows `glm5_next` (5.16+); the model config class comes from Transformers, not from vLLM |
| Mooncake | transfer engine available in the container (preinstalled in the vLLM-Ascend PD images) |
| Hardware | one A3 series node (16 logical devices) or one 950DT node (8 devices). Two devices are enough for the `min` topology. |

## 2 Why the model has to shrink, and what was shrunk

Full GLM-5.3-Flash is 320B parameters: ~305B of that is the 288-expert MoE
spread over 42 sparse layers. One node cannot hold two engines (P and D) of
that. Reducing depth and expert count keeps every *structural* property the
layerwise path depends on while cutting weights to ~8.7B (~16 GiB BF16) per
engine.

| Field | Released | Tiny | Why it is safe to change |
| --- | --- | --- | --- |
| `num_hidden_layers` | 45 | 8 | Depth is free. The 4-layer attention period is preserved: layers 3 and 7 are `deepseek_sparse_attention`, the rest `linear_attention` — the same phase as the released model (3, 7, 11, …). |
| `layer_types`, `indexer_types`, `mlp_layer_types`, `linear_attn_config.kda_layers`, `linear_attn_config.full_attn_layers` | 45 entries | 8 entries | Regenerated consistently; the builder validates that the three lists and the two index lists agree. |
| `first_k_dense_replace` | 3 | 3 | Keeps 3 dense MLP layers + 5 sparse (MoE) layers, so both MLP kinds are built. |
| `n_routed_experts` | 288 | 32 | Divisible by every EP size used below (1, 2, 4, 8) and still > `num_experts_per_tok`. This is the main weight saving. |
| `vision_config.depth` | 24 | 4 | The vision tower is still built and still profiled; only its depth drops. |

Everything else is left **exactly** as released, because the Ascend kernels and
the cache layout depend on it:

* `hidden_size` 4096, `num_attention_heads` 64, `q_lora_rank` 1536,
  `kv_lora_rank` 512, `qk_nope_head_dim`/`v_head_dim` 256, `mla_use_nope`,
  `intermediate_size` 12288, `moe_intermediate_size` 2048.
* `linear_attn_config.head_dim` 128 and `short_conv_kernel_size` 4 — the
  AscendC KDA kernel requires head_dim 128 and BF16 activations, and kernel
  width 4 is required when MTP is on
  (`vllm_ascend/models/glm5next/kda.py`).
* `index_kpool` 4, `index_topk` 2048, `index_kpool_always_select_tail` —
  the sparse indexer refuses `index_kpool_always_select_tail=False`, and the
  KV block size must stay divisible by `index_kpool`.
* `vocab_size` 154880 — so the real tokenizer, chat template and the `glm47`
  tool-call/reasoning parsers keep working.
* `num_nextn_predict_layers` 1 — the MTP draft layer is still present, so
  `--speculative-config '{"method": "deepseek_mtp"}'` remains possible
  (`MTP_TOKENS=3`, off by default).
* `mhc`, `hc_mult` 4 and the mHC parameters.

`config.tiny.json` in this directory is the generated result. Regenerate or
resize it with `build_tiny_model.py`:

```bash
python3 build_tiny_model.py --out /root/.cache/glm53-flash-tiny --print-only
```

```bash
python3 build_tiny_model.py \
  --out /root/.cache/glm53-flash-tiny \
  --reference-dir /root/.cache/modelscope/hub/models/vllm-ascend/GLM-5.3-Flash-w8a8 \
  --check-pcp 4
```

The builder writes `config.json` and copies the tokenizer/processor files
(everything except weights, the weight index and the ModelSlim quantization
description) from the reference checkpoint. It then validates the
config against the constraints the vLLM-Ascend GLM5Next implementation
enforces and prints a weight estimate.

If the full checkpoint is not on the node, only its small files are needed:

```bash
modelscope download --model Eco-Tech/GLM-5.3-Flash-w8a8 --local_dir /root/.cache/glm53-ref --include '*.json' '*.jinja' 'tokenizer*'
```

(check the include-flag spelling for your `modelscope` version), then point
`--reference-dir` at `/root/.cache/glm53-ref`.

Smaller still, e.g. for a 2-device node: `--layers 4 --experts 16`
(~4.1B params, ~7.6 GiB BF16).

## 3 Rules the layerwise path imposes on the topology

These come from `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
as changed by #20625, and they are what makes a single-node 1P1D layout a
constrained choice rather than a free one.

| Rule | Where it is enforced |
| --- | --- |
| Component-addressed transfer is selected automatically for this model (circular indexer-tail spec and pooled MLA cache) | `_requires_component_transfer`, `register_kv_caches` |
| **TP=1 or PCP=1 on both roles** (the KDA recurrent state is sharded over one axis only) | `_validate_mamba_parallel_config` |
| **Decode must run PCP=1** | `_prepare_component_request`, and the scheduler asserts it for `kv_consumer` |
| Producer needs **DCP=1 or DCP=PCP**; decode DCP must divide TP (these scripts leave DCP=1) | worker `__init__`, `_prepare_component_request` |
| **PP=1** | `register_kv_caches` |
| P and D `TP*PCP` head-shard counts need an **integer ratio** | `_prepare_component_request`, `_get_state_transfer_meta` |
| `linear_attn_config.num_heads` divisible by `TP*PCP` on each role (64 here) | `Glm5NextLinearAttention` (`assert num_heads % tp_size == 0`; `tp_size` is PCP when TP=1) |
| No transfer-time KV quantization, except the packed SFA-C8 cache | `register_kv_caches` (`enable_fa_quant` / `enable_c8_quant`) |
| `--use-replayssm` (RecoverSSM) is **not** usable with this connector: it requires `MooncakeConnectorV2` or `MooncakePullConnector`, and PCP=1 | `vllm_ascend/models/glm5next/recoverssm_config.py::validate_glm_recoverssm` |
| The **layerwise proxy** is mandatory: it routes each request to a decode engine first and relays that engine's KV transfer parameters to the prefill engine through `/v1/metaserver` | `examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py` |

Consequences for these scripts: BF16 weights and BF16 KV cache (no
`--quantization ascend`, no `--kv-cache-dtype int8/fp8`), no
`--use-replayssm`, decode always PCP=1, and the prefill role scales with
either PCP or TP but never both.

## 4 Topologies

`run_1p1d.sh` ships three presets; every value can be overridden.

| `TOPOLOGY` | Prefill | Decode | Devices | EP size (P / D) |
| --- | --- | --- | --- | --- |
| `min` (default) | DP1/TP1/PCP1, device 0 | DP1/TP1, device 1 | 2 | 1 / 1 |
| `8dev` | DP1/TP1/PCP4, devices 0-3 | DP4/TP1, devices 4-7 | 8 (a full 950DT node, or half an A3 node) | 4 / 4 |
| `16dev` | DP1/TP1/PCP8, devices 0-7 | DP4/TP2, devices 8-15 | 16 (a full A3 node) | 8 / 8 |
| `custom` | set `P_DP`, `P_TP`, `P_PCP`, `P_DEV_START` | set `D_DP`, `D_TP`, `D_DEV_START` | — | — |

Start with `min`: one device per role, no PCP, no TP, no DP, so a failure is
the connector or the model and nothing else. `8dev` mirrors the tutorial's
shape (PCP prefill, DP decode) at single-node scale; `16dev` is the closest
single-node analogue of the documented A3 two-node layout (prefill TP1/PCP8,
decode TP2).

`run_1p1d.sh` refuses a topology that breaks any rule in section 3, checks
that the device ranges and KV port ranges of the two roles do not overlap, and
checks `n_routed_experts` against both EP sizes before starting anything.

## 5 Run it

```bash
export MODEL_PATH=/root/.cache/glm53-flash-tiny
export LOCAL_IP=<this node IP>
export NIC_NAME=<network interface>
# A3 supernode whose device RoCE ports are down:
# export ASCEND_GLOBAL_RESOURCE_CONFIG='{"comm_resource_config.protocol_desc":["hccs:device"]}'

DRY_RUN=1 TOPOLOGY=min bash run_1p1d.sh   # validate the plan, start nothing
TOPOLOGY=min bash run_1p1d.sh
```

It starts the decode engine(s) first, waits for `/health` on each, then the
prefill engine(s), then the proxy, and prints the smoke-test command:

```bash
PROXY_URL=http://$LOCAL_IP:8081 bash smoke_test.sh
```

```bash
bash stop_1p1d.sh
```

Logs land in `./logs` (`d_rank0.log`, `p_rank0.log`, `proxy.log`, `pids`).

### Ports

One node means every port is shared, so unlike the two-node tutorial the two
roles must not reuse the defaults.

| Purpose | Prefill | Decode | Override |
| --- | --- | --- | --- |
| API | 9081 + rank | 9900 + rank | `P_API_PORT0`, `D_API_PORT0` |
| DP RPC | 12325 | 12425 | `P_DP_RPC_PORT`, `D_DP_RPC_PORT` |
| KV handshake base (`kv_port`) | 36680 | 36580 | `P_KV_PORT`, `D_KV_PORT` |
| Proxy | 8081 | — | `PROXY_PORT` |

The producer occupies `kv_port … kv_port + DP*PCP*TP - 1`
(`kv_port + dp_rank*PCP*TP + pcp_rank*TP + tp_rank`) and the consumer
`kv_port … kv_port + DP*TP - 1`. The Mooncake transfer-engine RPC ports are
assigned dynamically. Open the API, DP RPC, KV handshake and proxy ports in
the firewall.

### Other knobs

| Variable | Default | Notes |
| --- | --- | --- |
| `MAX_MODEL_LEN` | 8192 | Small on purpose; the tiny model has no trained long-context behaviour. |
| `P_MAX_NUM_SEQS` / `P_MAX_NUM_BATCHED_TOKENS` | 8 / 8192 | Prefill scheduler limits. |
| `D_MAX_NUM_SEQS` / `D_MAX_NUM_BATCHED_TOKENS` | 16 / 256 | Decode scheduler limits. |
| `P_GPU_UTIL` / `D_GPU_UTIL` | 0.85 | Both roles share one node; leave room for the other role's communication buffers. |
| `PREFIX_CACHING` | 0 | Set to 1 to enable prefix caching on both roles (#20625 also fixed prefix-hit bookkeeping for the layerwise path). |
| `MTP_TOKENS` | 0 | Set to e.g. 3 to enable `deepseek_mtp` speculative decoding on both roles. |
| `D_CUDAGRAPH` | 0 | Set to 1 for `{"cudagraph_mode": "FULL_DECODE_ONLY"}` on decode instead of `--enforce-eager`. |
| `LOAD_FORMAT` | `dummy` | Random weights. Only change this if you have a real checkpoint for the shrunk shape. |
| `BLOCK_SIZE` | 128 | Must stay divisible by `index_kpool` (4). |
| `DRY_RUN` | 0 | `DRY_RUN=1 bash run_1p1d.sh` validates the topology and prints the plan without starting anything; on `run_p.sh` / `run_d.sh` it prints the `vllm serve` command it would run. |
| `P_ENGINE_ID` / `D_ENGINE_ID` | 0 / 1 | Mooncake engine IDs; they only have to differ between the two roles. |
| `START_PROXY` | 1 | Set to 0 to start the engines only and run `run_proxy.sh` yourself. |

## 6 What a healthy run looks like

* Both roles log `Initializing Mooncake work <engine_id>` and register their
  KV caches; the prefill worker creates a sending thread and the decode worker
  a receiving thread.
* The proxy log shows each request hitting a decode engine first, a
  `/v1/metaserver` callback, then the prefill engine.
* `smoke_test.sh` returns a completion with `finish_reason` set and
  `completion_tokens` equal to `max_tokens`. The text is gibberish — the
  weights are random.

Failure modes worth recognising:

* `Mooncake Mamba requires PCP=1 or TP=1`, `Decoder component transfer
  currently requires PCP=1`, `Recurrent state does not match the PCP*TP head
  partition` → topology violates section 3.
* `Mooncake layerwise connector does not support KV cache components with
  different block size scales in one group` or an assertion that merged
  components must share one KV cache group → #20625 is **not** in the build.
* `Component-addressed transfer does not support transfer-time KV
  quantization` → a quantized KV cache is configured.
* `GLM RecoverSSM KV transfer requires MooncakeConnectorV2 or
  MooncakePullConnector` → `--use-replayssm` is on.

## 7 Status

The config generator and its validation, and the shell syntax of every script
here, were checked on a CPU-only host. **Nothing in this directory has been
executed on NPU hardware**: the topology rules and flag choices were derived
by reading the connector, the GLM5Next model code and the tutorial on the
`releases/v0.32.0rc` + #20625 tree, not by running a deployment.
