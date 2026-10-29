# Mooncake KV Connectors in vLLM-Ascend — Architecture Analysis

A code-level comparison of the three Mooncake-based P/D (prefill/decode) disaggregation
connectors in vLLM-Ascend:

| Config name | Class | File |
| --- | --- | --- |
| `MooncakeConnectorV2` / `MooncakePullConnector` | `MooncakePullConnector` | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/` (package) |
| `MooncakeHybridConnector` | `MooncakeConnector` | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` |
| `MooncakeLayerwiseConnector` | `MooncakeLayerwiseConnector` | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` |

---

## 0. Scope and method

- **Source analysed:** `vllm-project/vllm-ascend`, branch `main`, commit
  `1eda0dd` *("[BugFix] Keep GLM KPool ND and scope NZ handling to A5 vision (#19564)")*.
- Everything in this document was read directly from that tree. This fork's own `main`
  is far behind upstream, so the analysis was done against a fresh `upstream/main`
  worktree, not against the local checkout.
- Reproduce the exact tree with:

```bash
git remote add upstream https://github.com/vllm-project/vllm-ascend.git
```

```bash
git fetch upstream main --depth=1
```

```bash
git worktree add /tmp/upstream-main upstream/main
```

**Companion artifact:** an interactive diagram page that illustrates the three data
paths, the control planes, and the differences side by side. The link is in the chat
message that accompanied this report.

### Where these connectors are registered

All KV connector names resolve through `KVConnectorFactory`, populated in
`vllm_ascend/distributed/kv_transfer/__init__.py::register_connector()`:

```text
MooncakeConnectorV1        -> kv_p2p/mooncake_connector.py          : MooncakeConnector         (4409 lines, legacy monolith)
MooncakeD2RHConnectorV1    -> kv_p2p/mooncake_d2rh_connector.py     : MooncakeD2RHConnectorV1
MooncakeConnectorV2        -> kv_p2p/mooncake/connector.py          : MooncakeConnector (= MooncakePullConnector)
MooncakePullConnector      -> kv_p2p/mooncake/connector.py          : MooncakePullConnector
MooncakeHybridConnector    -> kv_p2p/mooncake_hybrid_connector.py   : MooncakeConnector
MooncakeLayerwiseConnector -> kv_p2p/mooncake_layerwise_connector.py: MooncakeLayerwiseConnector
```

Two naming traps worth internalising before reading any code:

1. **`mooncake_hybrid_connector.py` exports a class called `MooncakeConnector`**, the
   same class name used by the legacy `MooncakeConnectorV1` module and by the V2
   package facade. Grepping for `class MooncakeConnector` finds three unrelated
   implementations. Always disambiguate by module path.
2. **"Hybrid" in `MooncakeHybridConnector` means hybrid *KV cache* (several
   `kv_cache_groups`: full attention + SWA + Mamba + compressed), not a hybrid
   transfer strategy.** Separately, `docs/source/user_guide/feature_guide/mooncake_hybrid_attention.md`
   is *not* about this connector at all — it documents `AscendStoreConnector` with
   `backend="mooncake", use_layerwise=true`. Do not use that page to understand
   `MooncakeHybridConnector`.

### Code size

| Connector | Lines | Shape |
| --- | --- | --- |
| V2 (`mooncake/` package) | 4 024 across 8 modules (+ a 41-line `__init__.py`) | layered: facade / base / mode-specific |
| Hybrid | 2 263 in 1 module | monolith |
| Layerwise | 2 299 in 1 module (+ ~300 lines of layerwise-specific P/D mapping helpers at `utils/utils.py:50-347`) | monolith |
| (V1, for reference) | 4 409 in 1 module | monolith ancestor of both monoliths |

V2 package layout:

```text
kv_p2p/mooncake/
├── connector.py        279  vLLM-facing facade; MooncakeBaseConnector / Pull / Push
├── base_scheduler.py   267  scheduler logic shared by all transfer modes
├── base_worker.py      431  worker logic shared by all modes (registration + metadata publish)
├── pull_scheduler.py   767  pull-mode scheduler: control-plane threads, delayed free
├── pull_worker.py     1634  pull-mode worker: topology inference, block mapping, READ submit
├── metadata.py         182  typed handshake + request metadata dataclasses
├── stats.py            153  KVConnectorStats + Prometheus metrics
└── utils.py            311  zmq helpers, register-region collection, block grouping
```

---

## 1. Executive summary

| | **MooncakeConnectorV2 (Pull)** | **MooncakeHybridConnector** | **MooncakeLayerwiseConnector** |
| --- | --- | --- | --- |
| **Transfer direction** | **D pulls** (`batch_transfer_sync_read`) | **D pulls** (`batch_transfer_sync_read`) | **P pushes** (`batch_transfer_sync_write`) |
| **When KV moves** | After P finishes the whole prompt | After P finishes the whole prompt | **Per layer, during P's forward pass** |
| **Control plane** | **One ZMQ ROUTER per P engine, on the scheduler** | One ZMQ ROUTER **per P worker rank** | One ZMQ ROUTER **per D worker rank** (direction reversed) + **HTTP metaserver** |
| **Topology discovery** | **Inferred** from exchanged metadata (layer names, block shapes, strides, PP/PCP/TP/DCP sizes) | **Declared** in `kv_connector_extra_config.prefill/decode` | **Declared** + propagated in `kv_transfer_params` |
| **Pipeline parallel** | Yes, both sides, uneven splits, KVPP (per-layer KV ownership) | P-side only (`pp_layer_partition`); D must be PP=1 | **No PP support at all** |
| **Decode context parallel (DCP)** | Yes, including P/D DCP mismatch and block-size mismatch | **Asserted `dcp_size == 1`** | Yes (via `parallel_info` / CP groups) |
| **Prefill context parallel (PCP)** | Yes, both sides | P-side only | P-side only |
| **Unequal P/D TP** | Yes, by head-interval intersection | Yes, by block sub-slicing + post-transfer `reformat_kv_cache` | Yes, by `pd_head_ratio` **AllToAll resharding on P** |
| **KV quantisation (C8/FA-quant)** | **Rejected** on GQA (raw byte copy) | **Rejected** on GQA (raw byte copy) | **Supported** — quantises into a staging buffer before transfer |
| **Per-request failure reporting** | `finished_recving` + `failed_recving` + `get_block_ids_with_load_errors()` | **None** — failures only logged | `done` / `failed` sets + `get_block_ids_with_load_errors()` |
| **Metrics** | `KVConnectorStats` + Prometheus histograms | None | None |
| **P blocks freed by** | Scheduler↔scheduler DONE, with timeout fallback | Worker↔worker DONE with per-port counting, timeout fallback | Nothing to delay — `request_finished` always returns `False` |
| **Mamba / hybrid KV groups** | Yes, incl. unequal-TP conv/SSM projection splitting | Yes (equal TP only for Mamba) | Yes (P TP ≥ D TP for Mamba) |
| **Needs an external proxy for KV routing** | No (proxy only for request routing) | No (proxy only for request routing) | **Yes** — the metaserver is on the KV control path |

**One-sentence characterisation of each:**

- **V2** is the re-architected, topology-agnostic pull connector: the P scheduler
  publishes a complete, validated description of its memory layout once, and each D
  worker *derives* what to read from where. It is the only one with observability and
  structured failure propagation, and the only one approved for KVPP and for
  component-major MLA page layouts.
- **Hybrid** is the V1 lineage, hardened for multi-group (hybrid) KV caches and for
  GQA head-sharded P/D TP ratios, with NPU post-processing kernels to reassemble
  cache after pulling. It trades V2's inference for explicitly declared topology and
  is the most restricted on parallelism (no DCP, no D-side PP, no D-side PCP).
- **Layerwise** is the latency-oriented one: it overlaps transfer with P's forward
  pass by pushing each layer's KV as soon as that layer's cache write completes, so
  TTFT does not pay for a full-prompt transfer at the end. That inversion costs it
  PP support, requires a metaserver, and makes it the only connector that can
  quantise KV before it crosses the wire.

---

## 2. The shared foundation

All three sit on the same two layers.

### 2.1 vLLM's `KVConnectorBase_V1` lifecycle

Each connector is instantiated twice per engine — once with
`KVConnectorRole.SCHEDULER` and once per worker with `KVConnectorRole.WORKER` — and
delegates to a `*Scheduler` / `*Worker` pair. The hooks that matter here:

| Hook | Side | Role in these connectors |
| --- | --- | --- |
| `on_new_request` | scheduler | P-side prompt truncation for stateful models |
| `get_num_new_matched_tokens` | scheduler (D) | "this many prompt tokens will arrive from a remote engine, asynchronously" |
| `update_state_after_alloc` | scheduler (D) | capture the locally allocated block ids to be filled |
| `build_connector_meta` | scheduler | pack per-step work into `KVConnectorMetadata` for the workers |
| `request_finished` / `request_finished_all_groups` | scheduler (P) | decide whether to **delay freeing** P blocks, and emit the `kv_transfer_params` the proxy hands to D |
| `register_kv_caches` | worker | register NPU memory with the Mooncake transfer engine, publish address metadata |
| `start_load_kv` | worker (D) | queue/begin the actual transfer |
| `save_kv_layer` / `wait_for_save` | worker (P) | **only layerwise uses these** |
| `get_finished` / `get_transfer_results` | worker | report completions back to the scheduler |

`SupportsHMA` (hybrid memory allocator) is implemented by all three, so all three can
run under vLLM's hybrid KV cache manager with multiple `kv_cache_groups`.

### 2.2 Mooncake transfer engine, shared process-wide

`vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py` holds a module
singleton `global_te`:

- `get_transfer_engine(hostname, device_name)` lazily creates one
  `mooncake.engine.TransferEngine`, initialised as
  `engine.initialize(hostname, "P2PHANDSHAKE", "ascend", device_name)`, under a
  double-checked lock. A failed init is sticky — the process must restart (works
  around a resource leak in Mooncake ≤ 0.3.11.post1).
- `register_buffer(ptrs, sizes)` pins NPU KV cache memory for RDMA. **It is
  idempotent and first-caller-wins** (`if self.is_register_buffer: return`), so when
  two connectors run under `MultiConnector`, only the first one's regions are
  registered.
- `unregister_buffer` / `reregister_buffer` / `disconnect_all_peers` support
  sleep/wake.

The only two data-plane primitives any of the three use:

```text
engine.batch_transfer_sync_read (session_id, src_addrs, dst_addrs, lengths)   # V2, Hybrid
engine.batch_transfer_sync_write(session_id, src_addrs, dst_addrs, lengths)   # Layerwise
```

`session_id` is always `"{remote_ip}:{remote_te_rpc_port}"`. Everything above these
two calls is address arithmetic: turn "request R's logical blocks" into flat
`(src, dst, len)` triples. **That arithmetic is the entire substance of the three
connectors, and it is where they genuinely differ.**

A consequence worth stating plainly: since the payload is a raw byte copy between two
registered regions, **P and D must agree bit-for-bit on page layout and dtype**.
`ascend_config.py` enforces one instance of this directly — C8 KV-cache quantisation
on a GQA model is rejected for any `MooncakeConnector*`/`V1`/`V2`, with the error text
pointing the user at `MooncakeLayerwiseConnector`, which quantises *before* the copy:

```text
"MooncakeConnector does not support C8 KV cache quantization on GQA models.
 The producer keeps KV cache in bf16 while the consumer allocates int8 KV cache, so raw
 Mooncake transfer would reinterpret bf16 bytes as int8. Please disable C8 KV cache
 quantization or use MooncakeLayerwiseConnector, which quantizes KV cache before transfer."
```

### 2.3 Stateful-model prompt truncation (all three)

For Mamba/linear-attention and compressed-KV models, P deliberately computes only
`N-1` prompt tokens so D can recompute the last token and derive `h(N)` itself.
Each connector implements the same trick with slightly different names:

- V2: `_needs_prefill_token_truncation` → `_truncate_request_for_prefill`, guarded by
  `params["_p_side_truncated"]`, called from `on_new_request` so the local prefix-cache
  lookup uses the shortened prompt.
- Hybrid: `need_truncate = use_compress` (plus Mamba groups) → identical mechanism.
- Layerwise: `_has_attn_mamba_hybrid_cache` → `_truncate_request_for_hybrid_prefill`,
  plus `_trim_hybrid_remote_block_ids` to drop the now-unwritten tail block.

---

## 3. MooncakeConnectorV2 — pull, with inferred topology

### 3.1 Process and socket topology

```text
P engine                                           D engine
┌───────────────────────────────┐                  ┌──────────────────────────────┐
│ Scheduler                     │                  │ Scheduler                    │
│  MooncakeSchedulerSendingThread│ ZMQ ROUTER       │  MooncakeSchedulerRecvingThread│
│  bind kv_port + max_device_id  │◄────REQ──────────│  (REQ pool, no bind)         │
│       + dp_index               │  GET_META_MSG    │                              │
│                                │  DONE_RECVING_MSG│                              │
└───────────────────────────────┘                  └──────────────────────────────┘
        ▲ metadata                                            │ queue
        │ (set_xfer_handshake_metadata*)                      ▼
┌───────────────────────────────┐                  ┌──────────────────────────────┐
│ Workers (no background thread,│   Mooncake RDMA  │ Workers                      │
│  no socket bound)             │◄═════READ════════│  MooncakePullRecvingThread   │
│  register_kv_caches() only    │                  │  + 32-thread executor        │
└───────────────────────────────┘                  └──────────────────────────────┘
```

This is the structural change that defines V2: **exactly one ZMQ listener per P
engine, owned by the scheduler process.** Verified by grep — the only `zmq.ROUTER`
bind in the whole package is `pull_scheduler.py:340`; workers only ever open `zmq.REQ`
sockets (`pull_worker.py:1523`).

Port allocation (`base_scheduler.py:83`, `base_worker.py:146`):

```text
worker handshake ports : [kv_port, kv_port + dp_size*pp_size*pcp_size*tp_size)
   device_index = (pp_rank * pcp_size + pcp_rank) * tp_size + tp_rank
scheduler control port : kv_port + max_device_id + data_parallel_index
```

The scheduler port is deliberately placed *past* the worker range so it can never
collide with a worker handshake socket. (Worker `handshake_port` is still published in
the metadata as an identifier, but in V2 no worker binds it; connections are made to
`te_rpc_port` for data and to the scheduler port for control.)

### 3.2 Handshake metadata — the heart of V2

`metadata.py` defines a typed, validated, hierarchical description of P's memory:

```text
MooncakeTransferMetadataGroups          # what a D worker fetches with GET_META_MSG
├── engine_id, scheduler_host/port
├── pp_size, pcp_size, dcp_size, tp_size, use_kv_pp
└── metadata_by_pp_rank: {pp_rank -> MooncakePPTransferMetadata}
        ├── block_size, num_blocks
        ├── layer_names[]            ◄── PP-union, sorted; matching is BY NAME
        ├── layer_block_sizes[]
        ├── group_indices[]          ◄── which kv_cache_group each layer belongs to
        ├── block_strides[][]        ◄── per layer, per cache tensor (K, V, scale, …)
        ├── block_lens[][]
        ├── block_shapes[][]
        ├── block_size_scales[][]    ◄── kernel blocks per scheduler block
        └── metadata_by_pcp_rank: {pcp_rank -> {tp_rank -> MooncakeTPTransferMetadata}}
                 ├── te_rpc_port, local_ip, handshake_port
                 ├── layer_indices[]           ◄── layers this rank physically owns (KVPP)
                 └── kv_caches_base_addr[][]   ◄── aligned to the PP-union layer order
```

Three properties of this structure carry the whole design:

1. **Layers are matched by name, not by index.** `MooncakeSchedulerSendingThread._merge_metadata_by_pp_rank`
   builds a PP-union layer list, pads each worker's address table to it, and records
   per-worker `layer_indices`. That is what makes KVPP ("LayerSplit", different layers
   owned by different PCP/TP workers) expressible at all.
2. **Everything structural is cross-checked at merge time.** Per-layer signatures
   (`group_index`, `layer_block_size`, strides, lens, shapes, scales) must be identical
   across PCP/TP ranks of the same PP rank, PP/PCP/TP rank sets must be complete, and
   duplicate layers are rejected — all with named-field error messages. The D side
   re-validates on receipt (`_validate_remote_metadata`).
3. **The scheduler does the merging, not the worker.** Workers publish their own
   `MooncakeTransferMetadata` through `set_xfer_handshake_metadata{,_pp_aware}`; the
   scheduler merges, msgpack-encodes once in `__init__`, and then every
   `GET_META_MSG` is a single `send_multipart` of a precomputed blob.

Worker-side registration (`base_worker.py::register_kv_caches`) is where the physical
layout is read off the live tensors: `cache.data_ptr()`, `cache.stride(0)*element_size`,
`prod(shape[1:])*element_size`, `shape[0] // num_blocks`. Two special cases:

- `_get_shared_page_metadata` detects several views packed into one physical page and
  publishes the page as a single descriptor. It refuses to merge unless the views'
  contiguous spans **exactly cover** the page stride (holes and padding disqualify it),
  which prevents mistaking conventional planar K/V (same storage, pages apart) for a
  packed page.
- `_is_component_major_mla_binding` recognises the `[all NoPE][all RoPE][padding]`
  MLA slot layout and publishes the whole physical slot as a flat byte descriptor.
  This is why `vllm_ascend/core/kv_cache_interface.py` only allows the newer MLA page
  layout for `MooncakeConnectorV2` / `MooncakePullConnector` — a raw transfer carries
  no metadata to distinguish component-major from token-fused pages, so both endpoints
  must agree, and only V2 is known to describe it correctly.

`get_required_kvcache_layout()` returns `"HND"` for non-MLA models (and `None` for MLA,
where head/dim ordering is irrelevant), so no reformat kernel is ever needed after the
copy. **V2 is the only one of the three that implements this classmethod** — it
*prevents* the layout mismatch that Hybrid *repairs* afterwards.

### 3.3 Request flow

**P side, request finished** (`pull_scheduler.py::request_finished`): only acts when
`do_remote_decode` is set and `status == FINISHED_LENGTH_CAPPED`. It computes transfer
blocks via `_get_transfer_block_ids` — which is per-group and spec-aware:

```text
circular spec   -> keep all blocks (one request-lifetime ring page; it is state, not a prompt sequence)
MambaSpec       -> blocks[:-num_speculative_blocks]  (one state page holds the verify window)
SlidingWindow   -> cdiv(prompt_len, block_size * 1)            (SWA keeps its logical block size)
other attention -> cdiv(prompt_len, block_size * max(1, dcp_size))
```

then registers the request for delayed free and returns the params the proxy forwards
to D (`remote_block_ids`, `remote_num_prompt_tokens`, `remote_engine_id`,
`remote_host/port` = **scheduler** host/port, `remote_request_id`).

**D side:** `get_num_new_matched_tokens` reports `prompt_len - num_computed_tokens` and
`True` (asynchronous). `update_state_after_alloc` stores
`blocks.get_unhashed_block_ids_all_groups()` *and* `blocks.get_block_ids()` — both are
needed: SWA block mapping is computed from the difference between the full and unhashed
lists (`local_unhashed_start_idx`). If `num_external_tokens == 0` (full local prefix
hit) it skips straight to acknowledging P.

**D worker:** `start_load_kv` groups requests by `remote_engine_id` and queues them.
`MooncakePullRecvingThread.run` then, per batch:

1. fetch + cache P's metadata (`SizedDict`, bounded at 16 000 engines),
2. `_build_remote_transfer_layout` → per-PP-rank layer pairs and candidate remote
   PCP/TP rank groups,
3. `_build_transfer_block_buckets` → bucket by `(remote_pcp_rank, remote_tp_rank)`,
   then spec, then layer pair; block mappings are memoised per
   `(spec_index, candidate_topology)` so layers that share a spec compute it once,
4. submit one `_execute_tp_transfer_bucket` per bucket to a 32-thread executor
   (each thread pinned with `torch.npu.set_device`),
5. each bucket flattens to `src/dst/length` lists and issues **one**
   `batch_transfer_sync_read`,
6. `as_completed` collects per-bucket failures and attributes them only to the
   requests present in that bucket.

### 3.4 Topology inference — what makes V2 "topology-agnostic"

`_get_layer_remote_tp_rank_groups` dispatches on the KV cache spec and produces, for
each remote head/TP piece this local rank needs, a list of candidate remote ranks
(interchangeable replicas when remote DCP=1; the participating DCP shards when >1):

| Spec | Sharding model used |
| --- | --- |
| `MambaSpec` (`tp_replicated`) | every remote PCP/TP rank holds the full state → all are replicas |
| `MambaSpec` | each PCP/TP worker owns a *head shard*; integer ratio required |
| circular spec | every rank owns a complete ring page → treat PCP/TP/DCP as replicas |
| `AscendSFAIndexerCacheSpec` (replicated) | full sequence on every DCP rank → effective DCP=1 |
| `SlidingWindowMLASpec`, `MLAAttentionSpec` | one latent head; DCP handled for plain MLA |
| `SlidingWindowSpec` | head count from `block_shapes[...][0][0]`; DCP forced to 1 |
| `FullAttentionSpec` | head count from block shape; `_get_full_attention_dcp_sizes` picks TP-head sharding vs DCP |

`_infer_total_num_kv_heads` then reconciles local and remote head counts, TP sizes and
DCP sizes into a single total-head number and **raises if the inferences disagree**.
`_get_attention_remote_tp_rank_groups` intersects the local rank's head interval with
each remote head shard's interval; `_get_head_interval` handles both
`total_heads >= tp_size` (sharding) and `total_heads < tp_size` (replication).

Address generation (`_append_spec_transfer_addresses`) then slices inside the block:

```text
local_head_len   = local_block_len  // local_num_heads
transfer_len     = num_transfer_heads * local_head_len
local_inner_off  = (transfer_head_start - local_head_start)  * local_head_len
remote_inner_off = (transfer_head_start - remote_head_start) * remote_head_len
```

and merges contiguous blocks with `group_concurrent_contiguous` **only when stride ==
block_len == transfer_len on both sides** — otherwise it falls back to one range per
block. This is precisely the condition under which a multi-block copy stays a single
contiguous byte range.

The DCP path in `_compute_group_block_ids` is the most intricate code in the package:
it maps logical blocks through *virtual* block space
(`local_block_size * local_dcp_size`) to find the owning remote DCP rank, and handles
`local_block_size != remote_block_size` by walking kernel blocks token-start by
token-start. One logical request can therefore be split across several remote TP ranks.

Mamba unequal-TP transfer (`_append_mamba_transfer_addresses`) goes further and slices
*inside* the conv state: it decomposes the conv row into per-projection widths that
depend on the Mamba variant (MAMBA1 / MAMBA2 / GDN_ATTN have different orderings of
`(state, B, C)`), then emits one range per row per projection.

### 3.5 Completion, failure, observability

- **D → P completion is scheduler-to-scheduler.** The D worker reports
  `finished_recving`/`failed_recving` through `KVConnectorTransferResults`; the D
  scheduler's `update_connector_output` turns the aggregated per-request completion
  into **one** `DONE_RECVING_MSG` sent by `MooncakeSchedulerRecvingThread` over a
  pooled, persistent REQ socket. The P scheduler's `_mark_finished_locked` then moves
  the request from `delayed_free_requests` into `finished_requests`, which
  `update_connector_output` folds into `connector_output.finished_sending` — vLLM's
  standard delayed-free path. Unknown ids are warned about but still ACKed, so D stops
  retrying.
- **Timeout fallback:** `_retrieve_expired_requests_locked` force-frees anything older
  than `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` and logs at ERROR.
- **Failure granularity:** bucket-level. A failed executor task marks only the
  requests in that bucket. `get_block_ids_with_load_errors()` returns the affected
  local block ids so the scheduler can invalidate them — but **only when there is a
  single KV cache group** (`can_report_invalid_block_ids = len(kv_cache_groups) == 1`),
  because with several groups the block id spaces are not disjoint.
- **Metrics:** `stats.py` gives `MooncakeKVConnectorStats` (durations, bytes, failures;
  `reduce()` emits avg/P90 latency, avg MB, MB/s) and `MooncakePromMetrics`
  (`vllm:mooncake_xfer_time_seconds`, `vllm:mooncake_bytes_transferred`,
  `vllm:mooncake_num_failed_transfers`). **Neither other connector has any of this.**

### 3.6 Known shape of the roadmap

`connector.py` declares `MooncakePushConnector` and immediately raises
`NotImplementedError("MooncakePushConnector is not implemented yet.")`. The
base/scheduler/worker split exists specifically so that a push mode can reuse
registration, metadata and stats. `MooncakeConnector = MooncakePullConnector` is kept
as a backwards-compatible alias.

---

## 4. MooncakeHybridConnector — pull, with declared topology and post-transfer reformat

### 4.1 Process and socket topology

```text
P engine                                            D engine
┌────────────────────────────────┐                  ┌───────────────────────────────┐
│ Scheduler (no thread, no socket)│                 │ Scheduler (no thread/socket)  │
│  multi_nodes_meta_mapping:      │                 │                               │
│   port_offset -> {host,engine}  │                 │                               │
└────────────────────────────────┘                  └───────────────────────────────┘
┌────────────────────────────────┐                  ┌───────────────────────────────┐
│ Worker rank r                   │ ZMQ ROUTER      │ Worker rank r                 │
│  KVCacheSendingThread           │◄───REQ──────────│  KVCacheRecvingThread         │
│  bind side_channel_port +       │  GET_META_MSG   │   + 32-thread executor,       │
│   (pp*pcp+pcp)*tp + tp          │  DONE_RECVING   │     per-peer FIFO queues      │
│  KVCacheTaskTracker             │                 │  Mooncake READ ═══════════════╡
└────────────────────────────────┘                  └───────────────────────────────┘
```

Here the handshake is **per worker rank** and the control messages travel
worker↔worker. The schedulers only exchange the port→host mapping needed for
multi-node deployments.

### 4.2 Declared topology

`MooncakeConnectorWorker._get_prefill_decode_size` reads, and asserts the presence of:

```json
"kv_connector_extra_config": {
  "prefill": {"tp_size": 16, "dp_size": 2, "pp_size": 1, "pp_layer_partition": null},
  "decode":  {"tp_size": 16, "dp_size": 2, "pp_size": 1}
}
```

with `_decode_pp_size == 1` asserted, and `_prefill_tp_size >= _decode_tp_size`
enforced. Further hard constraints in `__init__`:

```python
assert self.dcp_size == 1, "Mooncake Hybrid Connector requires decode_context_parallel_size=1."
assert not (self.pp_size > 1 and self.pcp_size > 1), "pp and pcp cannot open in same time"
if self.kv_role == "kv_consumer" and self.pcp_size > 1:  raise ValueError(...)   # PCP is prefill-only
```

`tp_num_need_pulls` — how many separate pulls each D rank must issue per P stage — is
derived from head counts, not from metadata:

```python
num_d_block_heads = max(1, num_key_value_heads // tp_size)
num_p_block_heads = max(1, num_key_value_heads // prefill_tp_size)
tp_num_need_pulls = num_d_block_heads // num_p_block_heads     # 1 for MLA
```

### 4.3 Rank selection, with randomisation

`_get_remote_ranks_for_req` is where Hybrid differs most visibly from V2. Rather than
intersecting head intervals, it *samples* which redundant P replica a request reads
from, seeded by the request id:

```python
seed = string_to_int64_hash(req_id)
rand = random.Random(seed)
num_groups = max(1, len(ori_data[0]) // num_kv_head)   # redundant replicas per PP stage
rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
```

Deterministic given the request id (so P and D agree without extra messages), while
spreading load across replicas across requests. PCP replica choice uses a separate
seed namespace: `random.Random(string_to_int64_hash(f"pcp:{req_id}"))`. The unselected
PCP replicas on P are told `add_not_transfer_request(req_id)` so they finish locally
instead of waiting for a DONE that will never come.

### 4.4 Two transfer paths

`_handle_request` branches on `use_hybrid`:

**`_transfer_kv_cache_all_groups` (hybrid KV, multiple groups).** Iterates
`hma_group_size` groups; per group, trims `remote_block_ids` to the tail matching
`local_block_ids` (except Mamba), groups contiguous blocks, and walks the flat
group-major address list. With P-side PP > 1 it slices the *local* address list to the
remote stage's group range so stage 1 lands in D's groups 8–15 instead of overwriting
0–7. `addr_group_idx` filters out addresses that do not belong to the current group.

**`_transfer_kv_cache` (single group).** The GQA path. Each pull takes a sub-slice of
the block:

```python
inner_block_len = block_len // tp_num_need_pulls
src = src_base + local_block_id  * block_len        + inner_offset * inner_block_len
dst = dst_base + remote_block_id * inner_block_len
```

P's layer range comes from `pp_layer_indices[prefill_pp_rank]` (via
`get_prefill_pp_indices`, supporting an explicit `pp_layer_partition`), with a `+1` on
the last stage for the shared MTP layer.

### 4.5 Post-transfer reformat — Hybrid's signature step

Because D pulls head-sliced pieces from several P ranks into one block, the block is
*interleaved wrongly* until it is reassembled. On the last pull of a request
(`global_offset == tp_num_need_pulls * prefill_pp_size - 1`):

```text
need_cat_cache = tp_num_need_pulls > 1  and is_kv_transfer_end
need_nz_cache  = enable_kv_nz          and is_kv_transfer_end
```

Two implementations:

- **Fused op** (`enable_transpose_kv_cache_by_block` + custom op available):
  `torch.ops._C_ascend.transpose_kv_cache_by_block(...)` — one call for all layers.
- **Fallback:** per layer, `torch_npu.npu_gather_pa_kv_cache` into staging buffers →
  `_cat_kv_cache` (view as `(num_blocks, tp_num_need_pulls, block_size, -1)`,
  `transpose_(1, 2)`, contiguous) → `torch_npu.npu_scatter_pa_kv_cache` back. NZ format
  conversion reshapes with `nz_fmt_last_dim = 16` and scatters again.

The fallback carries a frank `# FIXME` that an unexplained
`torch.npu.synchronize()` is required or GQA crashes.

This is the architectural trade versus V2: **V2 forces `HND` layout up front so no
reassembly is needed; Hybrid pulls whatever layout exists and runs NPU kernels
afterwards to fix it.** Hybrid's way costs extra device work on D's critical path and
extra device memory for staging buffers; V2's way requires the layout classmethod to be
honoured by the model runner on both sides.

### 4.6 Completion and the gaps

- **Per-port DONE counting.** D's `_send_done_recv_signal` reports per handshake port;
  P's `run_busy_loop` counts arrivals per request against
  `remote_port_send_num[handshake_port]["num"]` and only then calls
  `update_done_task_count`. `_send_done_signal_to_free_remote_port` additionally sends
  DONE to ports with `num == 0` (ranks that were never pulled from) so they do not leak.
- `KVCacheTaskTracker` is the same delayed-free + `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`
  force-free pattern as V2, but at worker level.
- **No failure propagation.** `_handle_request` catches every exception, logs it, and
  then still sends the DONE signal in `finally`. There is no
  `get_block_ids_with_load_errors`, no `failed_recving`. A failed transfer is observable
  only in logs — D will proceed with whatever is in those blocks.
- **Peer fairness:** per-peer FIFO deques with `MAX_REQUESTS_PER_PEER_HANDLER = 5`, so
  one busy peer cannot monopolise an executor worker when peers outnumber `max_workers`.
- `_reconstruct_shared_pages` + `_SharedPage` handle the DeepSeek-V4 "compress" layout,
  rebuilding physical pages from `kv_cache_tensors` descriptors (`offset`,
  `layer_stride`, `block_stride`) so that inner views such as indexer K and its scale
  do not become independent full-page transfers.

---

## 5. MooncakeLayerwiseConnector — push, per layer, during the forward pass

### 5.1 Inverted control plane

```text
                       ┌──────────────── HTTP metaserver / proxy ───────────────┐
                       │  D posts its own kv_transfer_params here;              │
                       │  the proxy forwards the request to a P instance        │
                       └───────────▲──────────────────────────┬─────────────────┘
                     POST (httpx)  │                          │ request + params
                                   │                          ▼
┌──────────────────────────────┐   │              ┌───────────────────────────────┐
│ D scheduler                  │───┘              │ P scheduler                   │
│  update_state_after_alloc    │                  │  _reqs_need_send_layerwise    │
│   → allocate blocks,         │                  │   (per-step chunk accounting) │
│     publish remote_block_ids │                  │  request_finished → (False,None)│
└──────────────────────────────┘                  └───────────────────────────────┘
┌──────────────────────────────┐                  ┌───────────────────────────────┐
│ D worker rank r              │  ZMQ ROUTER      │ P worker rank r               │
│  KVCacheRecvingLayerThread   │◄────REQ──────────│  update_decoder_info()         │
│  bind side_channel_port+tp_r │  GET_META_MSG    │   fetches D's layer metadata   │
│  serves ITS OWN metadata     │  DONE_SENDING_MSG│  KVCacheSendingLayerThread     │
│                              │  FAILED_SENDING  │   (queue consumer, no socket)  │
│  ◄══════ Mooncake WRITE ═════╧══════════════════╡   batch_transfer_sync_write    │
└──────────────────────────────┘                  └───────────────────────────────┘
```

Three inversions relative to V2/Hybrid:

1. **D publishes, P fetches.** `MooncakeAgentMetadata` here is
   `{te_rpc_port, layer_metadata: {layer_name -> LayerMetadata}}` and is served by the
   **D** worker's ROUTER. P's `update_decoder_info` issues `GET_META_MSG` to D and
   caches the result per `(engine_id, port)`.
2. **P writes into D's memory** (`batch_transfer_sync_write`), so D's block ids must be
   known before P computes anything. That is why D allocates first and POSTs its params
   to the metaserver, and why `request_finished` can unconditionally return
   `(False, None)` — there is nothing to delay-free on P.
3. **The metaserver is on the KV control path**, not just the request-routing path. It
   is an `httpx.Client` (optionally mTLS via
   `kv_connector_extra_config.tls_config`: `ssl_enable`, `ssl_certfile`, `ssl_keyfile`,
   `ssl_keyfile_password`, `ssl_ca_certs`) and POSTs are retried 3× from a 32-thread
   executor on the scheduler.

### 5.2 The layerwise pipeline

Per forward step on P:

```text
bind_connector_metadata(meta)
   └─ prepare_layerwise_step(meta)        # once per step, BEFORE the forward pass
        ├─ _align_remote_block_ids        # scale D's block ids if P/D block sizes differ
        ├─ per group: _get_kv_split_metadata[_for_mamba]
        │     → P/D CP-group selection, head mapping, port mapping, trans_count
        ├─ _get_kernel_block_ids          # expand logical → kernel block ids
        └─ if resharding: build per-group rearrange block ids, block tables, token counts

for each transformer layer L:
   ... attention computes, writes KV into the paged cache ...
   on_kv_cache_written(layer)             # record an npu.Event at scatter time
   save_kv_layer(layer_name, kv_layer, …) # ← the whole mechanism lives here
        ├─ optional staging work on a separate npu.Stream:
        │     npu_gather_pa_kv_cache → (pd_head_ratio>1 ? transpose + kv_alltoall_and_rearrange)
        │                            → (C8 ? round/clamp to int8 ; FA-quant ? torch.ops.vllm.quantize)
        │                            → trans_nd_to_nz for the NZ layouts
        ├─ _mark_layer_reuse_pending(L)   # gate reuse of the physical slot
        └─ send_queue.put(SendTask(wait_event=…, k/v[_quant]_cache=…, layer_idx=L, …))

KVCacheSendingLayerThread (one background thread):
   task = send_queue.get()
   wait_event.synchronize()               # or resharding_stream.synchronize()
   per session: batch_transfer_sync_write(session, src[], dst[], len[])
   if layer_idx == total_layers - 1 and req.chunk_finish:
        callback → send_done_send_signal(DONE_SENDING_MSG | FAILED_SENDING_MSG)
   reuse_completion_callback(L, error)    # release the slot
```

The `send_queue` is deliberately **bounded** when resharding is active
(`maxsize = len(kv_cache_specs)` for hybrid, else 1), which makes `save_kv_layer`
block and provides backpressure: the forward pass cannot run arbitrarily far ahead of
the network.

### 5.3 Chunked-prefill streaming

The P scheduler tracks, per request, a `SendReqInfo(local_block_ids,
local_transferred_tokens, local_computed_tokens, request)` and updates it every step
from `scheduler_output`:

```python
send_req_info.update_transferred_tokens(round_down(local_computed_tokens, min(block_size)))
send_req_info.update_computed_tokens(computed_tokens + scheduled_tokens - spec_decode_tokens)
chunk_finish = local_computed_tokens >= len(request.all_token_ids)
```

So KV leaves P **as each chunk completes**, not after the whole prompt — the reason
this connector exists. `trans_count` per group tells D how many distinct senders must
report DONE before the request is complete; `KVCacheRecvingLayerThread.update_done_task`
accumulates a *set* of sender paths and only marks the request done when
`len(set) == trans_count`.

### 5.4 Unequal-TP resharding on the producer

This is the only connector that performs a **collective** as part of KV transfer.
With `pd_head_ratio > 1` (derived in `ascend_config.py` from
`prefill.tp_size / decode.tp_size`, clamped by `total_num_kv_heads`, with
`num_head_replica = prefill_tp_size // num_kv_head`):

1. gather the step's blocks out of the paged cache into a contiguous staging buffer,
2. reorder so each head group is contiguous
   (`view(num_blocks, pd_head_ratio, -1, …).transpose(0,1)`),
3. `kv_alltoall_and_rearrange(pd_head_ratio, keys, values)` — an **AllToAll across P's
   TP group**, so every P rank ends up holding exactly the heads one D rank wants,
4. write from the staging buffer, offsetting the destination by
   `block_len * ((tp_rank // num_head_replica) % pd_head_ratio)`.

`create_kv_buffer` allocates and Mooncake-registers those 2 MB-aligned staging buffers
up front. `update_decoder_info` even pre-creates the RDMA link with a dummy 128-byte
write when `pd_head_ratio > 1`, "to prevent alltoall out of memory".

Compare the three approaches to unequal P/D TP:

```text
V2       : D computes the head-interval intersection and reads only those bytes.      (no extra compute)
Hybrid   : D reads head sub-slices from several P ranks, then runs NPU transpose/scatter. (compute on D)
Layerwise: P AllToAlls heads across its TP group, then writes whole pieces.            (collective on P)
```

### 5.5 Buffer-reuse gating

`supports_layerwise_buffer_reuse = True` and `requires_full_blocks_on_update_after_alloc = True`
are class attributes that `AscendMultiConnector` keys off. Physical KV slots are tracked
by address (`layer_storage_slots: layer_idx -> (slot_id, …)`), each with a
`threading.Event`. `_mark_layer_reuse_pending` clears them before a send is queued;
`_complete_layer_reuse` sets them when the transfer finishes (recording any error).
`wait_for_layer_reuse(layer_idx)` blocks the model until the slot is free, logging every
10 s and failing fast if the send thread died.

`AscendMultiConnector._configure_layerwise_reuse_completion` discovers connectors with
this capability and installs `_wait_for_external_slot_release` as a composite waiter, so
a co-resident `AscendStoreConnector` cannot start an H2D copy into a slot whose outbound
P/D transfer is still in flight. Layerwise is the only Mooncake connector participating
in that protocol.

### 5.6 Failure handling

- Transfer errors set `failed_reqs`; on the last layer the callback sends
  `FAILED_SENDING_MSG` instead of `DONE_SENDING_MSG`.
- D's `get_finished` maps failed external request ids back to internal ids and adds
  their local block ids to `_invalid_block_ids`, surfaced through
  `get_block_ids_with_load_errors()`.
- `update_decoder_info` failures mark the request failed for that layer without killing
  the step.
- `do_virtual` requests (no actual transfer) are completed immediately on D.
- Request id bookkeeping: `get_external_request_id(request_id) = request_id[:-9]` strips
  the 9-character EngineCore suffix added by vLLM PR #27987, and `request_map` maps back.
  **This is a hard coupling to a vLLM-internal id format.**

### 5.7 What it gives up

- **No pipeline parallel.** Grep finds no `pp_rank`, `pp_size` or `get_pp_group` in the
  entire file. PP is simply not modelled.
- `prefill_context_parallel_size == 1` is asserted on decoder nodes.
- Different P/D block sizes require `pcp*dcp*remote_pcp*remote_dcp == 1`.
- Mamba groups require P TP ≥ D TP.
- `assert len(transfer_mappings) <= 1` — a request may not fan out to multiple D peers
  per group.
- No metrics, no Prometheus.

---

## 6. Side-by-side matrices

### 6.1 Control plane

| | V2 | Hybrid | Layerwise |
| --- | --- | --- | --- |
| ZMQ listeners per engine | **1** (scheduler) | `dp*pp*pcp*tp` (P workers) | `dp*pcp*tp` (**D** workers) |
| Who serves layout metadata | P scheduler (merged, pre-encoded) | each P worker (its own) | each **D** worker (its own) |
| Who requests it | D workers | D workers | **P workers** |
| Metadata type | `MooncakeTransferMetadataGroups` (PP→PCP→TP tree, validated) | `MooncakeAgentMetadata` (flat addr list + `block_lens` + `ssm_sizes`) | `MooncakeAgentMetadata` (`{layer_name -> LayerMetadata}`) |
| Completion message | `DONE_RECVING_MSG`, **scheduler→scheduler, once per request** | `DONE_RECVING_MSG`, worker→worker, counted per port | `DONE_SENDING_MSG` / `FAILED_SENDING_MSG`, P worker→D worker, counted per `trans_count` |
| Socket reuse | pooled persistent REQ sockets per endpoint | pooled persistent REQ sockets per endpoint | fresh `zmq_ctx(REQ)` per DONE, 3 retries |
| External HTTP dependency | none | none | **metaserver POST (with optional mTLS)** |
| Port formula | `kv_port + max_device_id + dp_index` (sched); `kv_port + dp_index*tp*pp*pcp + (pp*pcp+pcp)*tp + tp` (worker) | `kv_port + dp_rank*tp*pp*pcp`, `+ (pp*pcp+pcp)*tp + tp` | `kv_port + dp_rank*pcp*tp + pcp_rank*tp`, `+ tp_rank` |

### 6.2 Data plane

| | V2 | Hybrid | Layerwise |
| --- | --- | --- | --- |
| Primitive | `batch_transfer_sync_read` | `batch_transfer_sync_read` | `batch_transfer_sync_write` |
| Initiator | D worker | D worker | **P worker** |
| Granularity | one batch per `(pp, pcp, tp)` bucket, all layers + all requests merged | one batch per `(request, remote port)` | one batch per `(layer, session)` |
| Concurrency | 32-thread executor, `as_completed` | 32-thread executor, per-peer FIFO (max 5/round) | **1 background thread**, bounded queue |
| Overlap with compute | none (post-prefill) | none (post-prefill) | **yes — per layer, gated by `npu.Event`** |
| Staging buffers | none | only for reformat (transient) | **persistent, registered, 2 MB-aligned** |
| Post-transfer device work | none | `transpose_kv_cache_by_block` or gather/cat/scatter (+NZ) | none on D (P did the work) |
| Contiguity merging | `group_concurrent_contiguous` + `split_if_not_byte_contiguous`, only when stride==len==transfer_len | `group_concurrent_contiguous`, skipped when `tp_num_need_pulls > 1` | `group_concurrent_contiguous` |

### 6.3 Parallelism support

| | V2 | Hybrid | Layerwise |
| --- | --- | --- | --- |
| TP | yes, any integer ratio via head intervals | P TP ≥ D TP, head-divisible | yes, via `pd_head_ratio` AllToAll |
| DP | yes | yes | yes |
| PP on P | **yes, uneven splits, stage-local groups** | yes, via `pp_layer_partition` | **no** |
| PP on D | **yes** | no (`assert decode pp_size == 1`) | no |
| PCP on P | yes | yes | yes |
| PCP on D | **yes** | no (explicit `raise`) | no (`assert == 1`) |
| DCP | **yes, incl. P/D mismatch and block-size mismatch** | no (`assert dcp_size == 1`) | yes, with CP-group math; block-size mismatch needs CP=1 |
| KVPP (`enable_kvpp`) | **yes — the only P2P connector allowed** (`_KVPP_SUPPORTED_CONNECTORS = ("MooncakeConnectorV2", "AscendStoreConnector")`, and not on a decode-only node) | no | no |
| Mamba + unequal TP | **yes**, conv/SSM projection-level splitting per Mamba variant | equal TP only | P TP ≥ D TP |

### 6.4 Model / KV-spec support

| | V2 | Hybrid | Layerwise |
| --- | --- | --- | --- |
| `FullAttentionSpec` | yes | yes | yes |
| `SlidingWindowSpec` | yes (unhashed-prefix alignment) | yes (`get_sw_clipped_blocks`) | yes |
| `MLAAttentionSpec` | yes | yes | yes |
| `SlidingWindowMLASpec` | yes | — | — |
| `MambaSpec` | yes (incl. `tp_replicated`, MAMBA1/2/GDN) | yes | yes (`mamba_cache_mode` aware) |
| circular / ring-state spec | yes (explicit branch) | yes (`is_circular_kv_cache_spec`) | — |
| `AscendSFAIndexerCacheSpec` | yes, incl. DCP-replicated indexer | — | `sparse_sfa_c8_layers` handling |
| DeepSeek-V4 compressed pages | via `_get_shared_page_metadata` | via `_reconstruct_shared_pages` | via layer metadata |
| Component-major MLA pages | **yes — only V2/`MooncakePullConnector` are permitted** | no | no |
| Required KV layout | **`HND` enforced** for non-MLA | whatever exists, repaired after | whatever exists |
| C8 / FA KV quantisation on GQA | **rejected** | **rejected** | **supported** |
| GLM RecoverSSM models | required (`MooncakeConnectorV2` or `MooncakePullConnector` only) | no | no |

### 6.5 Robustness and observability

| | V2 | Hybrid | Layerwise |
| --- | --- | --- | --- |
| `failed_recving` set | **yes** | no | yes (as failed set) |
| `get_block_ids_with_load_errors()` | yes (single-group only) | **not implemented** | yes |
| Failure attribution | per `(pp, pcp, tp)` bucket | none | per request per layer |
| Delayed-free timeout | yes (scheduler) | yes (worker) | n/a (no delayed free) |
| Structured validation of peer metadata | **extensive, both directions** | minimal | minimal |
| `KVConnectorStats` | **yes** | no | no |
| Prometheus metrics | **3 metrics** | no | no |
| Bounded caches | `SizedDict(16000)` on all peer maps | `SizedDict(16000)` | `SizedDict(16000)` |

### 6.6 Configuration

```jsonc
// MooncakeConnectorV2 — topology is inferred; prefill/decode sizes in the
// e2e configs are legacy and unread by the V2 package itself.
{"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer",
 "kv_port": "30000", "engine_id": "0",
 "kv_connector_extra_config": {"use_ascend_direct": true,
   "prefill": {"dp_size": 4, "tp_size": 1}, "decode": {"dp_size": 16, "tp_size": 1}}}

// MooncakeHybridConnector — prefill/decode tp_size+dp_size are REQUIRED (asserted).
{"kv_connector": "MooncakeHybridConnector", "kv_role": "kv_producer",
 "kv_port": "30000", "engine_id": "0",
 "kv_connector_extra_config": {
   "prefill": {"dp_size": 2, "tp_size": 16, "pp_size": 1, "pp_layer_partition": null},
   "decode":  {"dp_size": 2, "tp_size": 16}}}

// MooncakeLayerwiseConnector — plus a reachable metaserver in the request params.
{"kv_connector": "MooncakeLayerwiseConnector", "kv_role": "kv_producer",
 "kv_port": "30000", "engine_id": "0",
 "kv_connector_extra_config": {"use_ascend_direct": true,
   "prefill": {"dp_size": 2, "tp_size": 8}, "decode": {"dp_size": 32, "tp_size": 1},
   "tls_config": {"ssl_enable": false}}}
```

Shared, read by all three: `qos_priority` (0–4, merged into
`ASCEND_GLOBAL_RESOURCE_CONFIG`), `ASCEND_TRANSFER_TIMEOUT` (derived from
`HCCL_RDMA_TIMEOUT`/`HCCL_RDMA_RETRY_CNT` when unset), `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`,
`MOONCAKE_CONFIG_PATH`.

Note the asymmetry: V2 e2e configs still carry `prefill`/`decode` blocks, but a grep of
`kv_p2p/mooncake/` shows the package never reads `get_from_extra_config("prefill")`.
Those keys survive because `ascend_config.py` uses them to derive `pd_tp_ratio` /
`pd_head_ratio` globally — which is also why they must stay accurate even for V2 if
other features read them.

---

## 7. Choosing between them

**Use `MooncakeConnectorV2`** for new deployments, and in particular whenever any of
these is true:

- the deployment uses PP on the decode side, DCP, decode-side PCP, or `enable_kvpp`;
- P and D have mismatched block sizes or mismatched DCP;
- the model needs component-major MLA pages, or is a GLM RecoverSSM model (both
  *require* V2 / `MooncakePullConnector`);
- you need transfer metrics or must react to failed loads (block invalidation + retry)
  rather than silently serving corrupt KV;
- the model has multiple heterogeneous KV cache groups with Mamba under unequal TP.

**Use `MooncakeHybridConnector`** when you are on a validated DeepSeek-V4-class recipe
that already depends on it — the e2e suite pins it for
`DeepSeek-V4-flash-w8a8-PD-prefix`, `DeepSeek-V4-Pro-w4a8-1M-PD` and similar. Its
GQA head-split + reformat path and its `_reconstruct_shared_pages` handling are
battle-tested for those layouts. Do not reach for it on a new topology: `dcp_size == 1`,
decode `pp_size == 1` and decode `pcp_size == 1` are hard asserts, and a failed transfer
is invisible to the scheduler.

**Use `MooncakeLayerwiseConnector`** when TTFT dominates and the topology is simple
(no PP, decode PCP=1), or when you specifically need **quantised KV on the wire** —
C8 / FA-quant on GQA models is rejected by the other two and `ascend_config.py` names
this connector as the remedy. Budget for the metaserver as a control-plane dependency,
for the registered staging buffers, and for an AllToAll on P when
`prefill.tp_size > decode.tp_size`.

---

## 8. Observations and gotchas

1. **Three classes named `MooncakeConnector`.** `mooncake_connector.py` (V1),
   `mooncake_hybrid_connector.py` (registered as `MooncakeHybridConnector`) and
   `mooncake/connector.py` (alias of `MooncakePullConnector`, registered as
   `MooncakeConnectorV2`). Always read module paths, not class names.
2. **`mooncake_hybrid_attention.md` does not document `MooncakeHybridConnector`.** It
   documents the `AscendStoreConnector` + mooncake-backend layerwise path. There is no
   feature-guide page for `MooncakeHybridConnector` in the tree; the code is the spec.
3. **`global_te.register_buffer` is first-caller-wins.** Under `MultiConnector`
   (`MooncakeConnectorV2` + `AscendStoreConnector`, as the e2e configs do), the second
   connector's `register_buffer` call is a no-op. This is intentional — both describe the
   same KV cache — but it means a connector cannot assume its own region list was the one
   registered.
4. **Hybrid swallows transfer failures.** `_handle_request` logs the exception and then
   signals DONE anyway. With no `get_block_ids_with_load_errors`, D continues with
   whatever bytes are in those blocks. If silent corruption on a flaky fabric is
   unacceptable, this is a reason to prefer V2.
5. **The GQA reformat `torch.npu.synchronize()` is an unexplained workaround.** The code
   says so: *"if we skip synchronization at this point, the system will crash in GQA
   scenarios. However, we still haven't identified the root cause."* It is a full device
   sync on D's critical path.
6. **Layerwise is coupled to a vLLM-internal request-id format.**
   `get_external_request_id = request_id[:-9]`, hard-coded against vLLM PR #27987's
   9-character suffix. A change in that suffix length breaks completion matching silently
   (ids simply stop resolving through `request_map`).
7. **V2's single scheduler-side control socket is the biggest operational win.** At
   DP=16, TP=16 the other designs bind hundreds of ROUTER sockets per engine; V2 binds
   one, and the metadata blob is merged and msgpack-encoded exactly once at startup.
8. **V2's `can_report_invalid_block_ids` silently degrades with multiple KV groups.**
   `len(kv_cache_groups) == 1` gates block-id reporting. On a hybrid model, a failed pull
   is still reported as `failed_recving` for the request, but the block ids are not
   returned, so group-wise invalidation is unavailable.
9. **`MooncakePushConnector` is declared and raises.** The V2 base/mode split exists for
   it; a push mode on V2's metadata plane is the obvious successor to the layerwise
   monolith, but it is not implemented on this commit.
10. **Layerwise's bounded `send_queue` is load-bearing, not an optimisation.** With
    resharding enabled, `maxsize` is 1 (or `len(kv_cache_specs)`), so `save_kv_layer`
    blocks when the sender falls behind. Without that backpressure the staging buffers
    would be overwritten mid-flight.

---

## 9. File index for further reading

```text
Registration & dispatch
  vllm_ascend/distributed/kv_transfer/__init__.py          connector name -> module/class
  vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py  MultiConnector + layerwise slot release

Shared infrastructure
  vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py   global_te singleton
  vllm_ascend/distributed/kv_transfer/utils/utils.py        QoS, timeouts, register regions,
                                                            CP groups, P/D block-port mappings,
                                                            kv_alltoall_and_rearrange
  vllm_ascend/ascend_config.py                              pd_tp_ratio / pd_head_ratio /
                                                            num_head_replica derivation, KVPPConfig,
                                                            C8-on-GQA rejection
  vllm_ascend/core/kv_cache_interface.py                    which connectors may use the new MLA layout

Connectors
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/*.py              V2 (pull)
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py      V1 (lineage)
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_d2rh_connector.py D2RH variant

Tests (best executable specification available)
  tests/ut/kv_offload/mooncake_v2/            8 modules: scheduler, worker, metadata, stats, utils
  tests/ut/kv_offload/test_mooncake_hybrid_connector.py
  tests/ut/kv_offload/test_mooncake_layerwise_connector.py
  tests/ut/distributed/kv_transfer/test_connector_registration.py
  tests/e2e/cases/models/configs/DeepSeek/*PD*.yaml          real deployment configs
  tests/e2e/cases/features/kv_pool/*.yaml                    V2 + AscendStore MultiConnector configs

Docs (read with the caveats in §0)
  docs/source/tutorials/features/pd_disaggregation_mooncake_single_node.md
  docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md
  docs/source/tutorials/features/pd_colocated_mooncake_multi_instance.md
  examples/disaggregated_prefill_v1/mooncake_connector_deployment_guide.md   (V1-era, still the clearest walkthrough)
```
