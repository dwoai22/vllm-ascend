#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Before/after microbenchmark for the Model Runner V2 penalties kernel.

Compares the current 1D-grid kernel against the pre-PR 2D-grid kernel on the
exact shapes sampling produces, without starting an engine or loading a model.
Needs an NPU (torch_npu + triton-ascend); needs no weights and no server.

The baseline below is spliced verbatim out of commit 6ba8bc2
(`vllm_ascend/worker/v2/sample/penalties.py` before this change) and only
renamed, so the comparison is against the real previous implementation rather
than a paraphrase of it. Inputs come from `create_test_data` in the nightly
test next to this file, so they are the same tensors the correctness tests use.

Each shape is measured as: warm up both kernels (so JIT compilation is outside
the timed region), check that both produce identical logits from the same
pristine input, then time both with `torch.npu.Event` over several samples,
alternating which runs first to cancel clock drift. Reported numbers are the
median over samples, in microseconds per call. No absolute threshold is
asserted: the useful signal is the relative change on one machine.

Usage:

    python3 tests/e2e/nightly/single_node/ops/singlecard_ops/triton/\
        benchmark_penalties_triton.py --vocab-size 151936 --spec-tokens 3

    # match a serving config exactly: 128 requests x (1 + 3 MTP positions)
    ... --rows 512 --iters 200 --samples 7
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import statistics
import sys

import torch
import triton
from vllm.triton_utils import tl

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton
from vllm_ascend.worker.v2.sample.penalties import VOCAB_BLOCK_SIZE, apply_penalties

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load_create_test_data():
    """Reuse the nightly test's input builder, imported by path."""
    path = pathlib.Path(__file__).with_name("test_penality.py")
    if not path.exists():
        raise SystemExit(f"cannot find {path}; run this script from inside the repo checkout")
    spec = importlib.util.spec_from_file_location("_penality_test_inputs", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.create_test_data


# ---------------------------------------------------------------------------
# Baseline: pre-PR implementation, spliced from commit 6ba8bc2 and renamed.
# ---------------------------------------------------------------------------
@triton.jit
def _penalties_kernel_baseline(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    NUM_VOCAB_BLOCKS: tl.constexpr,
    VOCAB_GRID_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0

    # NPU doesn't support chained 'or' operations like 'A or B or C'
    use_penalty = use_rep_penalty or use_freq_penalty
    use_penalty = use_penalty or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    vocab_program_idx = tl.program_id(1)
    for vocab_block_idx in tl.range(
        vocab_program_idx,
        NUM_VOCAB_BLOCKS,
        VOCAB_GRID_SIZE,
    ):
        block = vocab_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
        logits = logits.to(tl.float32)

        base_output_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask,
            other=0,
        )

        # Accumulate draft token counts from previous positions directly into
        # output_bin_counts (preserves its native tensor layout, avoiding an
        # expensive shared-memory layout conversion after the loop).
        pos = tl.load(expanded_local_pos_ptr + token_idx)
        start_idx = token_idx - pos
        output_bin_counts = base_output_counts
        for prev_pos in tl.range(pos):
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            token_match = block == prev_token
            output_bin_counts = output_bin_counts + token_match.to(tl.int32)
        output_bin_mask = output_bin_counts != 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_block = vocab_block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
                other=0,
            )
            bit_masks = 1 << tl.arange(0, 32)
            bit_masks_expanded = bit_masks[None, :]
            packed_expanded = packed_mask[:, None]
            bits_matrix = (packed_expanded & bit_masks_expanded) != 0
            prompt_bin_mask = bits_matrix.reshape(BLOCK_SIZE)

            # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
            # If logits are positive, divide by penalty, otherwise multiply by penalty.
            logits *= tl.where(logits > 0, 1.0 / scale, scale)

        # Apply frequency penalties.
        logits -= freq_penalty * output_bin_counts
        # Apply presence penalties.
        logits -= pres_penalty * output_bin_mask
        # Store back to logits.
        tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


def apply_penalties_baseline(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    num_tokens, vocab_size = logits.shape
    BLOCK_SIZE = 4096
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    vocab_grid_size = min(num_vocab_blocks, 65535 // num_tokens)
    _penalties_kernel_baseline[(num_tokens, vocab_grid_size)](
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        prompt_bin_mask.stride(0),
        output_bin_counts,
        output_bin_counts.stride(0),
        vocab_size,
        NUM_VOCAB_BLOCKS=num_vocab_blocks,
        VOCAB_GRID_SIZE=vocab_grid_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def _time_us(fn, iters: int) -> float:
    """Microseconds per call, measured on device over `iters` launches."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    torch.npu.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _grids(rows: int, vocab_size: int, cores: int) -> tuple[str, str, int]:
    blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    total_tiles = rows * blocks
    tiles_per_worker = triton.cdiv(total_tiles, max(1, cores))
    before = f"({rows}, {min(blocks, max(1, 65535 // rows))})"
    return before, f"({cores},)", tiles_per_worker


def _check(data, pristine) -> str:
    """Both implementations from the same input; equality is expected bit for bit."""
    logits = data[0]
    logits.copy_(pristine)
    apply_penalties(*data)
    after_new = logits.clone()
    logits.copy_(pristine)
    apply_penalties_baseline(*data)
    if torch.equal(logits, after_new):
        return "bit-exact"
    diff = (logits.float() - after_new.float()).abs().max().item()
    return f"DIFFERS (max abs {diff:.3e})"


def run(args: argparse.Namespace) -> int:
    create_test_data = _load_create_test_data()
    init_device_properties_triton()
    cores = get_vectorcore_num()
    dtype = _DTYPES[args.dtype]
    print(f"device {args.device}  vector cores {cores}  vocab {args.vocab_size}  dtype {args.dtype}")
    print(f"spec tokens {args.spec_tokens}  iters/sample {args.iters}  samples {args.samples}\n")
    header = (
        f"{'rows':>6}  {'before us':>10}  {'after us':>10}  {'delta':>8}  "
        f"{'grid before':>14}  {'grid after':>11}  {'tiles/worker':>12}  check"
    )
    print(header)
    print("-" * len(header))

    deltas = []
    for rows in args.rows:
        data = create_test_data(
            num_tokens=rows,
            vocab_size=args.vocab_size,
            num_status=max(1, rows // (args.spec_tokens + 1)),
            num_speculative_tokens=args.spec_tokens,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
            penalty_config=(args.repetition_penalty, args.frequency_penalty, args.presence_penalty),
            spec_decode_groups=True,
        )
        pristine = data[0].clone()

        # Warm up both, so JIT compilation is never inside a timed region.
        for _ in range(3):
            apply_penalties(*data)
            apply_penalties_baseline(*data)
        torch.npu.synchronize()

        verdict = _check(data, pristine) if args.check else "skipped"

        before_samples, after_samples = [], []
        before_call = lambda tensors=data: apply_penalties_baseline(*tensors)  # noqa: E731
        after_call = lambda tensors=data: apply_penalties(*tensors)  # noqa: E731
        for sample in range(args.samples):
            data[0].copy_(pristine)
            # Alternate which implementation goes first, so a clock ramp or a
            # thermal drift cannot be mistaken for a difference between them.
            order = (before_samples, before_call), (after_samples, after_call)
            for samples, call in order if sample % 2 == 0 else reversed(order):
                samples.append(_time_us(call, args.iters))

        before_us = statistics.median(before_samples)
        after_us = statistics.median(after_samples)
        delta = (after_us / before_us - 1.0) * 100.0
        deltas.append(delta)
        g_before, g_after, tiles = _grids(rows, args.vocab_size, cores)
        print(
            f"{rows:>6}  {before_us:>10.2f}  {after_us:>10.2f}  {delta:>+7.1f}%  "
            f"{g_before:>14}  {g_after:>11}  {tiles:>12}  {verdict}"
        )
        torch.npu.empty_cache()

    print(f"\nmedian change across shapes: {statistics.median(deltas):+.1f}%  (negative is faster)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=151936, help="model vocab size (Qwen3.6: 151936)")
    parser.add_argument(
        "--rows",
        default="1,8,32,64,128,256,512",
        help="sampled logits rows to sweep: requests x (1 + draft positions)",
    )
    parser.add_argument("--spec-tokens", type=int, default=3, help="draft positions per request")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="bfloat16")
    parser.add_argument("--iters", type=int, default=100, help="launches per timed sample")
    parser.add_argument("--samples", type=int, default=5, help="timed samples per shape, order alternated")
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--frequency-penalty", type=float, default=0.1)
    parser.add_argument("--presence-penalty", type=float, default=0.1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", dest="check", action="store_false", help="skip the equality check")
    args = parser.parse_args(argv)
    args.rows = [int(value) for value in args.rows.split(",") if value.strip()]
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
