# Testing MooncakeConnectorV2 KV sharing on a single node

Scripts to exercise the `KVCacheLayerTopology` owner/borrower split added on
`MCV2/kv-sharing` against a model that declares cross-layer KV sharing
(`kv_sharing_target_layer_name`), e.g. `google/gemma-4-E2B-it`.

Nothing here is wired into CI; these are untracked helper scripts.

## Why Gemma 4 E2B works as the subject

`vllm/model_executor/models/gemma4.py` gives the last `num_kv_shared_layers`
layers no KV cache of their own: each reads the cache of the last *non-shared*
layer **of the same attention type**. Because the model interleaves
`full_attention` and `sliding_attention`, it produces borrowers in both KV
cache groups — which is exactly the case `register_kv_caches` has to publish
owners for and refuse to publish borrowers for.

## 0. Pre-flight: confirm the checkpoint actually shares KV

No NPU needed. Point it at the local snapshot directory:

```bash
python kv_sharing_pd_test/check_kv_sharing.py /models/gemma-4-E2B-it
```

It prints `num_kv_shared_layers`, the owner layer range, and the exact
borrower → owner mapping vLLM will build. If `num_kv_shared_layers` is 0 the
checkpoint will not exercise the feature at all — pick another one before
burning an NPU run.

## 1. Launch 1P1D

Two NPUs, TP1 each. Pick whichever shape you prefer:

**Two shells, no proxy** (preferred for feature testing — `verify_pd.py`
talks to both engines directly, and this path honours `VLLM_LOGGING_LEVEL`):

```bash
MODEL=/models/gemma-4-E2B-it PREFILL_NPUS=0 bash kv_sharing_pd_test/serve_one.sh prefill
```

```bash
MODEL=/models/gemma-4-E2B-it DECODE_NPUS=1 bash kv_sharing_pd_test/serve_one.sh decode
```

**One shell, with the load-balance proxy** (wraps
`examples/disaggregated_prefill_v1/launch_single_node_pd.py`, adds a proxy on
:8080, writes logs to `$LOG_DIR`):

```bash
bash kv_sharing_pd_test/run_pd_1p1d.sh /models/gemma-4-E2B-it
```

Both default to `TP_SIZE=1`, NPU 0 for prefill and NPU 1 for decode,
`MAX_MODEL_LEN=8192`, prefix caching off, eager mode. For TP2+TP2 on four
cards: `TP_SIZE=2 PREFILL_NPUS=0,1 DECODE_NPUS=2,3`.

Neither script passes `--kv-sharing-fast-prefill`, and you should leave it
off: it moves Gemma 4 onto the `Gemma4SelfDecoderLayers` /
`Gemma4CrossDecoderLayers` split, a different execution path from the one the
connector change targets. Test that combination separately, if at all.

## 2. Check the connector classified the layers correctly

Each engine logs the split once, at registration:

```bash
grep "Registered Mooncake KV caches" prefill.log
```

```text
Registered Mooncake KV caches: engine_id=prefill-0, num_layers=30,
num_kv_sharing_layers=5, num_blocks=..., num_regions=..., registered_bytes=...
```

Expect `num_kv_sharing_layers` to equal `num_kv_shared_layers` from step 0,
and `num_layers` (the published owners) to equal
`num_hidden_layers - num_kv_shared_layers`. If `num_kv_sharing_layers` is 0
while step 0 found borrowers, the borrowers are being published as layers of
their own and the point of the change is lost.

For the full map, restart one engine under DEBUG (`serve_one.sh` only):

```bash
VLLM_LOGGING_LEVEL=DEBUG MODEL=/models/gemma-4-E2B-it bash kv_sharing_pd_test/serve_one.sh prefill
```

```bash
grep "kv_sharing_layers=" prefill.log
```

Compare it entry by entry against step 0's table.

## 3. Check the transfer is actually complete

```bash
python kv_sharing_pd_test/verify_pd.py \
  --prefill-url http://127.0.0.1:13700 \
  --decode-url  http://127.0.0.1:13701
```

For each prompt it generates the completion twice on the same pair of engines:
once entirely on the prefiller, once over the P→D route (prefill emits one
token, hands `kv_transfer_params` to the decoder, decoder finishes). Greedy
decoding makes a complete transfer byte-identical, so a divergence is a real
signal — KV that a borrower layer reads did not arrive, or arrived stale.

A short prompt, a long multi-block prompt, and a counting prompt are covered;
the long one is what catches a mistake confined to the sliding-window group.

With the proxy path from step 1 you can also just curl it:

```bash
curl -s http://127.0.0.1:8080/v1/completions -H 'Content-Type: application/json' \
  -d '{"model": "kv-sharing-test", "prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}'
```

## 4. Cover the push (layerwise) connector too

`MooncakeConnectorV2` picks `MooncakePullConnector` by default and
`MooncakePushConnector` when `use_layerwise` is set. The change lives in the
shared `base_worker`, so both modes need a run:

```bash
USE_LAYERWISE=1 MODEL=/models/gemma-4-E2B-it bash kv_sharing_pd_test/serve_one.sh prefill
```

```bash
USE_LAYERWISE=1 MODEL=/models/gemma-4-E2B-it bash kv_sharing_pd_test/serve_one.sh decode
```

Then repeat step 3.

## Notes

- `HCCL_IF_IP` is auto-detected from the default route and must be the host IP,
  not `127.0.0.1`: the connector advertises it to the peer in the handshake.
  Override `NIC` or `HCCL_IF_IP` if the node has several interfaces.
- Mooncake must be built with `-DUSE_ASCEND_DIRECT=ON`, and
  `libmooncake*.so` loadable; `serve_one.sh` adds the installed `mooncake`
  package directory to `LD_LIBRARY_PATH` when it can find it.
- Both engines run eager by default. To exercise the decode graph path add
  `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` and drop
  `--enforce-eager` from `serve_one.sh`.
