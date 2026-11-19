# Mistral Large 4: Prefill/Decode disaggregation on Ascend

Scripts for bringing up Mistral Large 4 as a 1P1D deployment with
`MooncakeConnectorV2`, in two sizes:

| Preset | Nodes | Weights | Per role | Purpose |
| --- | --- | --- | --- | --- |
| `preset_shrunk_2node.sh` | 2 | reduced, dummy | 1 node, TP8 + EP8 | exercise the PD path |
| `preset_full_4node.sh` | 4 | full FP8 checkpoint | 2 nodes, DP2/TP8 + EP16 | serve the real model |

Model support comes from [vllm-ascend#21014](https://github.com/vllm-project/vllm-ascend/pull/21014),
which needs a vLLM carrying [vllm#64534](https://github.com/vllm-project/vllm/pull/64534).
That PR is not merged, so no vLLM release or nightly has it. Check before
anything else, with the interpreter that runs `vllm serve`:

```bash
python3 check_ml4_support.py --model /path/to/checkpoint
```

It separates the two causes: a vLLM without #64534, and a checkpoint whose
quantization block vLLM rejects even with it.

Without it, serving fails while parsing `params.json`:

```
AssertionError: Only NO_SCALES and TENSOR (default) are supported for qscheme_act
```

The checkpoint is 128x128 block FP8, so its `params.json` declares
`qscheme_act: BLOCK`; #64534 is what teaches the Mistral config adapter that
scheme. The message comes from the installed vLLM — not from the checkpoint,
and not from vllm-ascend, which does not touch that code path. Patching the
assertion gets nowhere, because the model package is missing too and
vllm-ascend registers its adapter only when that package exists. Run the check
on **every** node: a P/D pair with one side missing #64534 will not work.

## Why four nodes for the full model

The FP8 checkpoint is ~0.99 TiB of weights. Sharded over one 8-NPU node that is
~127 GiB per NPU, which does not fit in 96 GB — so **no role can run on a single
node**, and a 3+1 split is not an option either. Two nodes per role shard the
routed experts 16 ways:

| Layout | EP | Weights/NPU | Free at 0.85 util | KV capacity |
| --- | --- | --- | --- | --- |
| 1 node, TP8 | 8 | ~127 GiB | does not fit | — |
| 2 nodes, DP2/TP8 | 16 | ~64 GiB | ~11 GiB | ~280k tokens |
| 2 nodes, TP16 | 16 | ~64 GiB | ~12 GiB | ~300k tokens |
| 4 nodes colocated, DP4/TP8 | 32 | ~33 GiB | ~43 GiB | ~1.2M tokens |

Estimates, not measurements: MXFP8 weights (1 byte + one E8M0 scale per 32),
BF16 for the embeddings, LM head, `kv_a_proj_with_mqa`, `kv_b_proj` and the
indexer tensors the checkpoint leaves unconverted. KV is ~41 KiB per token per
rank (63 of 72 layers own a latent cache, 44 own an indexer cache); the MLA
latent cache is replicated across TP ranks, so free memory per NPU sets the
capacity. The engine logs the KV cache size it actually settled on — trust that
over this table.

`DP2/TP8` is the default rather than `TP16` because it keeps tensor parallelism
inside a node and puts only the expert all-to-all on the inter-node link. The
same shape is what vLLM-Ascend documents for other large MoE models (one DP rank
per node, TP within, EP across). **Pipeline parallelism is not an option**:
upstream asserts `pipeline_parallel_size == 1` for this model.

If you do not need disaggregation, all four nodes in one colocated engine
(`DP4/TP8`, EP32) is far roomier — the last row above.

## Files

- `check_ml4_support.py` — whether this vLLM carries #64534, and whether the
  checkpoint's quantization block is one it accepts.
- `shrink_ml4_config.py` — writes the reduced model directory (2-node preset).
- `env.sh` — shared settings and helpers; sources `$ML4_PRESET` first.
- `run_prefill.sh` / `run_decode.sh` — one engine role; take the node index.
- `run_proxy.sh` — the PD proxy, on any host that reaches both roles.
- `smoke_test.sh` / `pd_check.py` — one request, then verify the KV came from P.

## Full model on four nodes

Copy this directory to the same path on all four nodes. On every node:

```bash
export ML4_PRESET=preset_full_4node.sh
export P_NODE_IPS="<P0_IP> <P1_IP>"
export D_NODE_IPS="<D0_IP> <D1_IP>"
export NIC_NAME=<interface carrying this node's address>
export MODEL_DIR=/path/to/Mistral-Large-4-1T-A52B
```

`MODEL_DIR` must hold `params.json` and the `consolidated*.safetensors` files,
and must **not** contain `config.json`: vLLM reads this model through
`--config-format mistral`, and a `config.json` sitting beside `params.json` is
merged in as defaults. The scripts refuse to start if one is present.

Start the head node of each role first, then its worker:

```bash
bash run_prefill.sh 0   # on P0 — serves the Prefill API
```

```bash
bash run_prefill.sh 1   # on P1 — headless, joins P0
```

```bash
bash run_decode.sh 0    # on D0 — serves the Decode API
```

```bash
bash run_decode.sh 1    # on D1 — headless, joins D0
```

Then the proxy, on P0 or any host that reaches both roles:

```bash
bash run_proxy.sh
```

```bash
bash smoke_test.sh
```

Each role exposes **one** endpoint — its head node. The head runs vLLM's
data-parallel load balancer and feeds the headless workers, so the proxy is
registered with two addresses, not four. Expect a long first start: weight load
plus the band-by-band MXFP8 requantization of every expert. `VLLM_ENGINE_READY_TIMEOUT_S`
is raised to 6000 s for that reason.

## Ports

With `N = DP × TP × PCP × PP` per role (16 here), `MooncakeConnectorV2` uses
`kv_port .. kv_port+N-1` for worker handshakes and `kv_port+N .. kv_port+N+DP-1`
for scheduler control; DP rank `r` starts at `kv_port + r×TP`. The transfer
engine binds its own RPC port on top, so open the peer role's node generously.

| | Prefill | Decode |
| --- | --- | --- |
| API | 8000 (head only) | 8000 (head only) |
| DP RPC | 16600 (head) | 16700 (head) |
| Mooncake | 30000–30017 | 30100–30117 |
| Proxy | 9000 on P0 | |

## Things that bite

- **One `engine_id` per prefill DP rank.** The decode side caches a producer's
  transfer metadata keyed by `engine_id` alone, so two prefill ranks sharing an
  id would have the second read the first's cache addresses. The scripts derive
  `ml4-prefill-<node index>`. Published templates for other models share one id
  across decode ranks — harmless there, because nothing pulls from the consumer.
- **Both roles must agree on the KV cache dtype.** Leave `KV_CACHE_DTYPE` empty
  and both read it from the checkpoint, or set it on both. Raw transfers carry
  no layout metadata, so a mismatch is silent.
- **`--quantization` is not used.** The block-FP8 scheme comes from the
  checkpoint's own `quantization_config`; `--quantization ascend` is for
  ModelSlim checkpoints and would select the wrong loader.
- **MLAPO and MegaMoE do not engage for this model**, so flags for them copied
  from other tutorials do nothing: MLAPO needs `AscendW8A8LinearMethod` and a
  fused QKV-A projection, and this model has neither (no q-LoRA, block FP8);
  the expert width of 6144 is outside MegaMoE's `[256, 4096]` window, which the
  adapter logs at startup.
- **`HCCL_BUFFSIZE_EP` is device memory outside `--gpu-memory-utilization`.**
  It is set to 2048 MB when a role spans nodes; count it when sizing the KV
  cache, and lower the utilization rather than overcommitting.
- **Decode's `max-num-batched-tokens` stays small** because it only decodes. A
  request sent straight to the Decode endpoint instead of the proxy would
  prefill locally and can exceed it; `pd_check.py` sends its reference request
  to Prefill for that reason.

## Optional

- `DECODE_GRAPH=1` — capture `FULL_DECODE_ONLY` graphs on Decode. Turn it on
  after plain eager P→D works.
- `SPEC_DECODE=1` — DSpark drafting with the drafter in the checkpoint (depth 3
  is fixed by the checkpoint). Needs real weights and the V2 runner.
- `--enable-eplb` as an extra argument to `run_decode.sh` — PR #21014 adds EPLB
  support for this model.
- `GPU_MEMORY_UTILIZATION`, `MAX_MODEL_LEN`, `P_MAX_NUM_SEQS`, `D_MAX_NUM_SEQS`
  — see the memory table; the full preset defaults to a 65536-token context.

None of these scripts carries an accuracy claim. PR #21014 reports no accuracy
validation for this model on Ascend, and the 2-node preset uses dummy weights,
so verify quality on the full checkpoint before drawing conclusions from it.
