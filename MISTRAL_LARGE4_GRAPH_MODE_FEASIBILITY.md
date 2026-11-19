# Mistral Large 4 on Ascend — `FULL_AND_PIECEWISE` feasibility

**Branch:** `claude/vigilant-meitner-yawgv1` @ `1386357` (vllm-ascend)
**Question:** does this branch support full-decode + piecewise-prefill graph mode, and should we add it?
**Date:** 2026-11-17

---

## TL;DR

1. **`FULL_AND_PIECEWISE` is not blocked on this branch.** Nothing in the Mistral adapter rejects
   it, the SFA/indexer backends declare the capability level that upstream accepts for it, and the
   custom op it needs to split on is already registered and already carries the eager-break
   decorator. It is in fact what a **default** launch resolves to today, because you never pass
   `cudagraph_mode`.

2. **But it is completely unvalidated**, and the task is not "add a feature" — it's "validate a
   path that is already reachable".

3. **For a PD deployment, `FULL_AND_PIECEWISE` is the wrong target in both directions**: a D node
   never dispatches its piecewise half, a P node never dispatches its full half. If the goal is to
   get the **prefill node off eager** — which it is — the mode to evaluate is plain `PIECEWISE`,
   and the cheaper thing to try *first* is `FULL_DECODE_ONLY`, which already works on this model
   and gives prefill npugraph_ex fusion. See [§2.6](#26-the-prefill-node-the-case-that-actually-matters-here).

4. **Prefill graphs can be captured at full chunk size.** `max_prefill_cudagraph_capture_tokens`
   takes any value up to `max_num_batched_tokens`; the low default on Ascend is inherited, not
   inherent. But **vllm-ascend references none of the MRV2 split capture fields**, so that path is
   untested here and bypasses Ascend's TP-aligned capture-size filter. See §2.6.

5. **The one thing I would put on the TODO list now** is the inverse of what was asked: *pin or
   document the mode you actually support*, because users launching with default flags currently
   land on an untested path. That's a few lines; see [Recommendation](#recommendation).

---

## 1. Does the current branch support it?

### 1.1 The mode survives resolution

Upstream narrows `cudagraph_mode` against the weakest attention backend in the model, in
`CompilationConfig.resolve_cudagraph_mode_and_sizes` (`vllm/config/compilation.py:1531`). For
Mistral Large 4 the weakest backend reports `AttentionCGSupport.UNIFORM_BATCH`:

| Backend | Declared support | Source |
|---|---|---|
| `AscendSFAMetadataBuilder` | `UNIFORM_BATCH` | `vllm_ascend/attention/sfa_v1.py:558` |
| `AscendSFAIndexerBackend` | `UNIFORM_BATCH` | `vllm_ascend/attention/indexer.py:619` |
| `AscendMLAMetadataBuilder` | `UNIFORM_BATCH` | `vllm_ascend/attention/mla_v1.py:456` |

Walking the downgrade branches with `min_cg_support = UNIFORM_BATCH` and a requested
`FULL_AND_PIECEWISE`:

- **mixed-batch check** — fires only when `mixed_mode() == FULL`. For `FULL_AND_PIECEWISE`,
  `mixed_mode()` is `PIECEWISE`. Not taken.
- **decode-FULL check** — fires only when support is `NEVER`. Not taken.
- **spec-decode check** — fires when `support.value < UNIFORM_BATCH.value`, i.e. `2 < 2`. Not taken.
  (So DSpark does not disqualify it either.)
- **`piecewise_capture_available`** — satisfied by a compiled submodule; the model class carries
  `@support_torch_compile` (`vllm_ascend/models/mistral_large4.py:621`).

**The mode passes through unchanged.** Ascend MRV2 goes through exactly this function — its
`initialize_kv_cache` calls `_initialize_kv_cache_upstream` inside `graph_manager_wrapper`
(`vllm_ascend/worker/v2/model_runner.py:497`), and only swaps in `ModelAclGraphManager`.

### 1.2 The split point already exists

For piecewise, the platform appends the attention ops to `splitting_ops`
(`vllm_ascend/platform.py:1440`–`1460`), including `vllm::mla_forward`. Mistral's attention
dispatches through precisely that op: `AscendMistralMLAAttention` builds an
`AscendMultiHeadLatentAttention` (`mistral_large4.py:304`), whose `forward` calls
`torch.ops.vllm.mla_forward` (`vllm_ascend/ops/mla.py:321`). That op is also decorated
`@eager_break_during_capture` (`vllm_ascend/ops/mla.py:324`) — the breakable-cudagraph break marker.

So both piecewise routes (torch.compile FX splitting, and breakable stream-capture breaks) have
their boundary in place for this model already.

### 1.3 Nothing rejects it

`_reject_unsupported` (`mistral_large4.py:487`) refuses only `fuse_allreduce_rms` and
`use_sequence_parallel_moe`. No graph-mode gate.

### 1.4 …but it has never been run

| Coverage | Mode |
|---|---|
| `test_mistral_large4_dummy_forward_and_logprob_replay` | `enforce_eager=True` |
| `test_mistral_large4_decode_replays_under_graph_capture` | `cudagraph_mode: "FULL_DECODE_ONLY"` |
| two-card + PD e2e | eager / `FULL_DECODE_ONLY` |
| anything piecewise | **none** |

Repo-wide, the ratio is stark — `grep -rho` over `tests/`:

```
384  FULL_DECODE_ONLY
 15  "FULL"
 13  "PIECEWISE"
 11  FULL_AND_PIECEWISE
```

and of those 11, 8 are unit tests of platform/padding logic. Every large MoE / MLA / sparse model
in the accuracy-and-perf suite runs `FULL_DECODE_ONLY`: DeepSeek V4 and V3.1-Terminus, GLM-5.1 /
5.2 / 5.3, Kimi K2.5 / K2.6 / K2.7 / K3, Qwen3.5 / 3.6, MiniMax-M3 — **including the PD configs**.

> **Caveat on test state:** all the Mistral e2e cases are gated on
> `_upstream_has_module("vllm.models.mistral.mistral_large4")` (`vllm_ascend/models/__init__.py:131`).
> The vLLM checkout in this workspace is at Nov 4 2026 and does **not** contain that module
> (it has Mistral Large *3* only), so these tests currently skip here. Upstream
> [vllm-project/vllm#64534](https://github.com/vllm-project/vllm/pull/64534) landed around Nov 13,
> so this should resolve on the next pin bump — worth confirming, since it means the
> `FULL_DECODE_ONLY` capture test may not have run in CI yet either.

**Verdict:** reachable, default-selected, zero validation.

---

## 2. Should we add it? The evidence

### 2.1 The headline numbers do not measure what you would gain

This is the single most important point. **Every published benchmark I found compares
`FULL_AND_PIECEWISE` against *eager* or against *piecewise-only* — never against
`FULL_DECODE_ONLY`.**

From [vllm-project/vllm#47089](https://github.com/vllm-project/vllm/issues/47089) (B200), where
"PW" is `PIECEWISE` and "FCG" is `FULL_AND_PIECEWISE`:

| Model | BS | PW tok/s | FCG tok/s | Gain |
|---|---|---|---|---|
| Qwen3-0.6B TP1 | 1 | 274.5 | 654.3 | +138% |
| Qwen3-235B-A22B-FP8 EP2 | 1 | 74.8 | 78.7 | +5.2% |
| Qwen3-235B-A22B-FP8 EP2 | 128 | 33.1 | 33.7 | +1.7% |
| DeepSeek-V3 EP8 | 16 | 44.5 | 43.6 | **−2.2%** |
| DSV4-Flash NVFP4 TP4 | 1 | 31.7 | 156.6 | +394% |
| Qwen3.5-397B NVFP4 TP2 | 128 | 44.6 | 45.3 | +1.7% |

And from [vllm-ascend#17995](https://github.com/vllm-project/vllm-ascend/pull/17995) — the closest
analogue to us, a sparse-attention MoE on Ascend (DeepSeek-V4-Flash-W8A8):

| Mode | tok/s | vs eager |
|---|---|---|
| Eager | 11.62 | 1.00× |
| `PIECEWISE` | 20.82 | 1.79× |
| `FULL_AND_PIECEWISE` | 44.94 | 3.87× |

No `FULL_DECODE_ONLY` row anywhere. All of these are **output-throughput** runs (#47089 used input
length 1 throughout), so they are decode-dominated — which means the large gains are overwhelmingly
**the FULL decode graph**, the half we already have.

> The honest expected delta of `FULL_DECODE_ONLY → FULL_AND_PIECEWISE` is the **prefill / mixed-batch
> half only**. Nobody has published that number. The two large-MoE data points above that *are*
> near-apples-to-apples on the decode side (Qwen3-235B at +1.7%, DeepSeek-V3 at −2.2%) suggest the
> residual is in the noise for big models.

Also note #47089's own conclusion: gains shrink as model size and batch size grow, because bigger
models do more work per layer so the fixed per-step host overhead matters less. Mistral Large 4
is 1T-A52B — the far end of that curve. The issue was closed "not planned" in Nov 2026.

### 2.2 On Ascend the trade is not free: you lose npugraph_ex — *but only vs a `FULL_DECODE_ONLY` baseline*

> **Correction (2026-11-19).** This section, as first written, was framed against a
> `FULL_DECODE_ONLY` baseline and I wrongly carried it over to the prefill node. **In eager mode
> npugraph_ex is already disabled** (`platform.py:1435`–`1439`: `cudagraph_mode == NONE` forces
> `enable_npugraph_ex`, `enable_static_kernel` and `enable_super_kernel` all to `False`), so a P
> node that runs `--enforce-eager` today loses nothing here by moving to a piecewise mode. The
> argument below applies to the **decode** node, where the baseline keeps npugraph_ex. For the
> prefill node see [§2.6](#26-the-prefill-node-the-case-that-actually-matters-here).

This is the Ascend-specific wrinkle with no GPU equivalent, and it is decisive.

`_setup_compile_backend` (`vllm_ascend/platform.py:1371`) branches on the mode:

| Requested mode | `splitting_ops` | `enable_npugraph_ex` | static / super kernel |
|---|---|---|---|
| `FULL`, `FULL_DECODE_ONLY` | `[]` (unsplit) | **stays on** | available |
| anything containing `PIECEWISE` | attention ops appended | **forced `False`** (`platform.py:1463`) | **forced off** |

`docs/source/user_guide/feature_guide/graph_mode.md` states the same: npugraph_ex is "enabled by
default on supported hardware when `cudagraph_mode` is `FULL` or `FULL_DECODE_ONLY`" and
"automatically disabled in `FULL_AND_PIECEWISE`, `PIECEWISE`, and `NONE` modes."

npugraph_ex is compile-time FX operator fusion (`add + rms_norm → npu_add_rms_norm`, etc.). Under
`FULL_DECODE_ONLY` the fused callable is what decode replays **and** what prefill executes eagerly —
so prefill already benefits today. Moving to `FULL_AND_PIECEWISE` trades that fusion, everywhere,
for host-side launch-overhead reduction on prefill/mixed batches.

### 2.3 There is direct Ascend evidence that this trade went negative

[vllm-ascend#22100](https://github.com/vllm-project/vllm-ascend/pull/22100) moved the `step3p7`
nightly (`Step3_7_Flash_W8A8_A5`) **off** `FULL_AND_PIECEWISE` and back to `FULL_DECODE_ONLY`,
reporting "a clear prefill regression for long inputs". Its stated reasoning:

- `FULL_AND_PIECEWISE` splits the compiled prefill graph at attention boundaries — "20+ splitting
  points" — paying per-segment dispatch and synchronization overhead.
- It forcibly disables `enable_npugraph_ex`.
- It gains nothing in return, because the prefill chunk (~16k tokens) is far above that config's
  cudagraph capture ceiling (288), so prefill never replays a captured graph in *either* mode.
- Piecewise only helps "small mixed prefill and decode batches".

⚠️ **That PR reports no measurements** — "CI and Nightly tests", no TTFT or throughput figures. Treat
it as a strong directional signal from the people who own this path, not as a measured number.

There is a matching signal upstream. `vllm/config/vllm.py:145` explains why ROCm opts
`DeepseekV41ForCausalLM` into breakable cudagraphs by default:

> "the ROCm sparse SWA backend only reports `AttentionCGSupport.UNIFORM_BATCH`. Default
> `FULL_AND_PIECEWISE` then dies at capture unless breakable CUDA graphs are on."

That is structurally our situation — a sparse backend at `UNIFORM_BATCH` meeting a default
`FULL_AND_PIECEWISE` — and upstream's answer was not "fix piecewise", it was "route around it".

### 2.4 Your deployment shape makes the gain structurally ~zero

Upstream's own mode guidance (`vllm/docs/design/cuda_graphs.md:45-46`):

> `FULL_DECODE_ONLY` — "suitable for **decode instances in a P/D setup** where prefill is not as
> important, this way we can save the memory needed for `PIECEWISE` CUDA Graphs."
>
> `FULL_AND_PIECEWISE` — "(default mode) … generally the most performant setting, **especially for
> low latency with small models or MoEs**, but also requires the most memory and takes the longest
> to capture."

This branch ships PD disaggregation over MooncakeConnectorV2 (`32f0e94`, `deb609a`, `c4085de`).
Under PD:

- **D nodes** see uniform decode batches → already `FULL`. The piecewise half never fires, so
  `FULL_AND_PIECEWISE` on a D node is `FULL_DECODE_ONLY` plus wasted capture.
- **P nodes** see prefill batches only → **no uniform decode batches at all**, so the `FULL` half
  of `FULL_AND_PIECEWISE` never dispatches there either. If you want a graph mode on P, the mode
  to ask for is plain **`PIECEWISE`**, not `FULL_AND_PIECEWISE`.

So `FULL_AND_PIECEWISE` specifically — the dual mode — is the wrong target for a PD deployment in
both directions: each node only ever exercises one of its two halves. See
[§2.6](#26-the-prefill-node-the-case-that-actually-matters-here) for what to do on P instead.

### 2.6 The prefill node: the case that actually matters here

Both PD nodes in the e2e test pass `--enforce-eager`
(`tests/e2e/pull_request/two_card/test_mistral_large4_pd.py:75`, shared by `kv_producer` and
`kv_consumer`). So the P-node baseline is **eager**, not `FULL_DECODE_ONLY`, and there are *two*
upgrades available from there — not one:

| P-node config | compile mode | fusion on prefill | graph replay on prefill |
|---|---|---|---|
| eager (today) | `CompilationMode.NONE` | **none** | none |
| `PIECEWISE` | `VLLM_COMPILE`, split at attention ops | basic FX fusion passes only — npugraph_ex forced off (`platform.py:1463`) | only if batch ≤ capture ceiling |
| `FULL_DECODE_ONLY` | `VLLM_COMPILE`, **unsplit** (`splitting_ops = []`) | **npugraph_ex** — the stronger pass | none, but executes the *fused* callable |

The third row is the one I missed first time. npugraph_ex runs at **compile time** on the FX graph
(`vllm_ascend/compilation/compiler_interface.py:328` → `npugraph_ex_compile`), so it is
batch-agnostic: under `FULL_DECODE_ONLY` a prefill batch dispatches to runtime mode `NONE` and
executes the already-fused compiled callable. You get the `add + rms_norm → npu_add_rms_norm` class
of fusion on prefill, with no FX splitting — and it needs **no new work**, because
`FULL_DECODE_ONLY` is the one graph mode already covered for this model
(`test_mistral_large4_decode_replays_under_graph_capture`, which prefills 64 tokens before
decoding).

**On the capture ceiling — correction (2026-11-19).** An earlier draft of this section said
piecewise graphs "are never replayed for production-sized prefill chunks". **That was wrong.** It
described the Ascend *default*, not a limit. The accurate position:

Model runner V2 has a **dedicated prefill capture knob**, separate from the decode one
(`vllm/config/compilation.py:758`–`770`):

- `prefill_cudagraph_capture_token_sizes` — "Token counts of the CUDA graphs for prefill batches…
  `[]` captures none, and sizes above `max_num_batched_tokens` are dropped."
- `max_prefill_cudagraph_capture_tokens` — "Largest token count of a prefill CUDA graph… **at most
  `max_num_batched_tokens`**. More and wider graphs take memory from the KV cache and lengthen
  startup."

So **`max_num_batched_tokens` is exactly the documented ceiling** — you can capture a prefill
piecewise graph for a full chunk. There is even a purpose-built ladder builder,
`prefill_cudagraph_capture_grid` (`compilation.py:90`), which yields "54 sizes at 8192", so the
padding worry (capture only 8192, then pad a 200-token tail chunk up to it) does not arise either:
the dispatcher picks the smallest graph ≥ the token count.

What *is* true is the default. These fields are `None` unless set, and the wide-grid default
(8192) is gated on `wide_defaults`, which requires SM100 — so on Ascend the `else` branch runs and
`prefill_sizes = list(shared_sizes)` (`vllm/config/vllm.py:2869`), i.e. the decode-oriented shared
list, capped by:

```python
default_max_graph_size = 1024 if is_data_center_blackwell else 512
max_graph_size = min(max_num_seqs * decode_query_len * 2, default_max_graph_size)
```

— **128 tokens** at `max_num_seqs=64` with no spec decode, 256 with a 2-token DSpark decode. That
is why [#22100](https://github.com/vllm-project/vllm-ascend/pull/22100)'s 16k chunk met a 288
ceiling and prefill went unreplayed: nobody had raised the knob, not that it could not be raised.

> ⚠️ **And here is the Ascend-specific catch, which is the real finding.** `grep` across
> `vllm_ascend/` and `tests/` returns **zero** references to
> `prefill_cudagraph_capture_token_sizes`, `max_prefill_cudagraph_capture_tokens` or
> `decode_cudagraph_capture_request_sizes`. The Ascend platform's capture handling
> (`platform.py:1371`–`1430`) manipulates only the **deprecated shared**
> `cudagraph_capture_sizes` / `max_cudagraph_capture_size`: the `reduced_cg_cap` logic, the
> TP-alignment filter, and `update_cudagraph_capture_sizes`. Consequences:
>
> - Setting the prefill knob should work — it is resolved in upstream config — but it is
>   **completely untested on Ascend**, with no code, test or doc acknowledging it.
> - Ascend's TP-aligned capture-size filter (required when SP or shared-expert-DP is on, and which
>   otherwise raises `AssertionError`) operates on the shared list only, so it would **not** apply
>   to prefill sizes set through the new field. For an SP / shared-expert-DP deployment that is a
>   plausible hang or assertion gap worth checking before relying on it.

So the cost of capturing prefill graphs at chunk size is **memory and startup**, not impossibility.
For Mistral Large 4 that cost is steep: piecewise splits at every attention boundary, and the real
checkpoint's layer indices run to at least 71 (the adapter test notes layers 39..71 all read layer
35's latent), so a full 54-size grid is on the order of a few thousand captured graph segments.
Stream *count* is likely fine on a current stack — the release notes put the budget at ~32k graphs
on A3 and ~64k on Ascend 950 after the HDK 25.5.1 / CANN 8.5.0 fix — so the binding constraint is
capture memory competing directly with KV cache on a 1T model, plus the documented Ascend
capture-resource-exhaustion failure mode (`docs/source/faqs.md` item 17).

On top of that, prefill is compute-bound — large GEMMs and big attention — so host launch overhead
is amortized over far more device work than in decode. That is the structural reason every graph
mode helps decode more than prefill, and it is visible in
[#47089](https://github.com/vllm-project/vllm/issues/47089)'s data: gains shrink monotonically as
batch size grows. It is an argument about the *size* of the win, not its existence.

**Net:** piecewise on the prefill node is a real option, and with
`max_prefill_cudagraph_capture_tokens` raised to the chunk size it genuinely does get graph replay
— the ceiling is a knob, not a wall. The open questions are how much a compute-bound phase gains
from removing launch overhead, and whether the capture memory is affordable against KV cache on a
1T model. `FULL_DECODE_ONLY` remains the cheaper first experiment because it needs no new config
surface and is already validated. Settle it by measurement, in the order given in
[Recommendation](#recommendation).

### 2.5 Cost and risk if you do it

Bounded, but real. The template is your own most recent commit, `1386357` (SSMax): a Python-side
publish that was correct at an eager break and **stale inside a captured graph**. Under
`FULL_AND_PIECEWISE` the process alternates between *both* regimes batch by batch, so you must get
both right plus any state shared across them. Specific things that would need checking for this
model:

- the shared `topk_indices_buffer` producer → consumer handoff between layers
  (`mistral_large4.py:105`, `:326`) — a producer writes it, the next `"shared"` layer reads it;
- the nine KV-borrowing layers' write skip (`adopt_shared_kv_skip`), which the constructor already
  guards with a hard count check;
- the indexer's own cache write, run serially inside the SFA impl rather than on a side stream;
- the DSpark draft graphs, which have their own manager and capture path;
- **capture memory**: `FULL_AND_PIECEWISE` captures the most graphs, and capture-resource exhaustion
  is a documented Ascend failure mode *specifically* in piecewise configs (`docs/source/faqs.md`
  item 17, and `graph_mode.md`: "most likely to appear in `PIECEWISE` or `FULL_AND_PIECEWISE`
  configurations because those paths tend to capture more graphs").

For scale, [#17995](https://github.com/vllm-project/vllm-ascend/pull/17995) — adding
`PIECEWISE`/`FULL_AND_PIECEWISE` to PCP for DSA/SFA — touched the PCP manager, two CP attention
backends, padding-trim logic, hidden-state restoration, and had to **veto fused preprocessing
entirely** under PCP after hitting a layout mismatch (`token_x=256` vs `cache_index=512` at PCP=2).
That is the shape of the work when a sparse-attention model meets this mode.

---

## Recommendation

**Do not prioritize adding `FULL_AND_PIECEWISE`.** Reframe the TODO item as *measure, then pin*.
In priority order:

### A. Pin or document the supported mode — *do this one* (small)

Right now a user who serves Mistral Large 4 with default flags gets `FULL_AND_PIECEWISE`
(`vllm/config/vllm.py:337`), which no test has ever exercised for this model. Your `FULL_DECODE_ONLY`
e2e case had to read the resolved mode back from the worker precisely because the platform lowers
modes silently — the same silence applies in the other direction. Either:

- add a graph-mode narrowing + `logger.info_once` to `_reject_unsupported` (`mistral_large4.py:487`),
  the way `_update_compilation_modes` already narrows encoder-decoder models to `PIECEWISE`; or
- state the supported mode in the model's deployment doc and recipe.

This is the actual gap the investigation turned up, and it's cheap.

### B. Measure the **prefill node** ladder, cheapest first (config-only, no code)

The P node runs eager today, so there are two upgrades, and the cheap one is already supported:

1. **P node: eager → `FULL_DECODE_ONLY`**, with `cudagraph_capture_sizes` pinned small (e.g.
   `[1, 2]`) so the decode graphs it captures and never replays cost almost nothing. This buys
   npugraph_ex fusion on prefill and is **already validated on this model** — no new code, no new
   risk. Measure TTFT.
2. **P node: eager → `PIECEWISE`**, and set `max_prefill_cudagraph_capture_tokens` to your
   `max_num_batched_tokens` so the prefill graphs are actually captured and replayed. Without that
   the Ascend default caps prefill graphs at ~128–256 tokens and you measure almost nothing (this
   is the step an earlier draft of this document got wrong). Watch startup time and the KV-cache
   budget — the graph grid competes with it. Also confirm whether Ascend's TP-aligned capture-size
   filter needs to apply here if you run SP or shared-expert-DP; it currently only sees the
   deprecated shared list.
3. If (2) wins on TTFT but the memory cost is too high, sweep
   `max_prefill_cudagraph_capture_tokens` down to find the knee — a ceiling covering your common
   chunk sizes rather than the maximum may capture most of the gain.

Measure **TTFT / prefill latency**, not output tok/s — output tok/s is decode-dominated and will
hide the entire effect you are looking for. Note that (1) and (2) are independent of the D node;
you can set the modes per role.

If you also want the dual mode on a non-PD single instance, that is the separate `FULL_DECODE_ONLY`
vs `FULL_AND_PIECEWISE` comparison on a mixed workload, and §2.1–§2.3 are the reasons to expect
little from it.

### C. Only if B is promising: take the breakable route, not FX splitting

The one sparse-MoE config in this repo that runs `FULL_AND_PIECEWISE` —
`tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V4-Flash-W8A8-A3-Breakable.yaml` — gets there with
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, not through torch.compile. That matters, because breakable
cudagraphs "replace vLLM's torch.compile-based FX graph splitting with runtime stream-capture
breaks" (`vllm/compilation/breakable_cudagraph.py` docstring) — so the "20+ prefill split points"
cause of the #22100 regression does not apply. And `mla_forward` is **already** decorated with
`@eager_break_during_capture`, so the Mistral path is already instrumented for it.

One thing to resolve with the graph-mode owners first: that YAML sets
`"enable_npugraph_ex": true`, but `_setup_compile_backend` forces `enable_npugraph_ex = False` for
*any* mode containing `PIECEWISE` (`platform.py:1463`) and runs *after* the user config is merged in
(`platform.py:630` then `:633`). On my reading the YAML's `true` is overridden — i.e. even the
breakable route loses npugraph_ex today. If that's a bug, fixing it is what would make
`FULL_AND_PIECEWISE` actually attractive on Ascend, and it's a platform-level win rather than a
Mistral one.

### D. Add the regression test regardless (small)

Add a `FULL_AND_PIECEWISE` case to `tests/e2e/pull_request/one_card/test_mistral_large4.py`
mirroring the existing capture test — read the resolved mode back via `collective_rpc`, then assert
the second decode reproduces the first. It is cheap, and it is exactly the shape of test that would
have caught the SSMax bug class. Do this even if you never ship the mode, because the mode is
reachable by default.

---

## Sources

- [vLLM CUDA Graphs design doc](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — mode semantics and intended use
- [vllm-project/vllm#47089](https://github.com/vllm-project/vllm/issues/47089) — PW vs FCG benchmarks on B200/B300
- [vllm-project/vllm-ascend#22100](https://github.com/vllm-project/vllm-ascend/pull/22100) — step3p7 prefill regression, reverted to `FULL_DECODE_ONLY`
- [vllm-project/vllm-ascend#17995](https://github.com/vllm-project/vllm-ascend/pull/17995) — PCP + PIECEWISE/FULL_AND_PIECEWISE for DSA/SFA, with throughput table
- [vllm-project/vllm-ascend#9572](https://github.com/vllm-project/vllm-ascend/pull/9572) — initial `FULL_AND_PIECEWISE` support on Ascend (no benchmarks published)
- [vllm-project/vllm#64534](https://github.com/vllm-project/vllm/pull/64534) — upstream Mistral Large 4
- [vLLM Ascend graph mode guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html) and `docs/source/user_guide/release_notes.md` in-tree
