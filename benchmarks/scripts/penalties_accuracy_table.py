#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Operator accuracy table (before vs after) for the Model Runner V2 penalties kernel.

Emits a Markdown table, ready to paste into a PR description, comparing:

  * **before** - the baseline ``_penalties_kernel``, launched on a
    ``(num_tokens, vocab_grid_size)`` grid where
    ``vocab_grid_size = min(num_vocab_blocks, 65535 // num_tokens)``. Vendored
    verbatim into this script (see ``BASELINE_SOURCE``) so that both revisions
    can be measured in a single process, from a single checkout.
  * **after** - ``vllm_ascend.worker.v2.sample.penalties.apply_penalties`` as
    imported from the checkout this script runs in: one program per vector
    core, walking a contiguous run of the flattened ``(token, vocab_block)``
    tile space, with the per-row scalar loads hoisted out of the vocab loop.

Both are scored against the same float32 PyTorch reference
(``pytorch_apply_penalties``), and the two kernels are also compared directly
against each other. The optimization only changes how tiles are assigned to
programs, not the arithmetic applied to a tile, so the expected result is a
**bitwise identical** output: that is what the "before == after" column reports.

Scoring follows the reference accuracy standard's mixed tolerance,
``|actual_i - golden_i| <= atol + rtol * |golden_i|`` with the per-dtype
constants in ``TOLERANCES``, and requires both of its criteria at once:

  1. at least ``WITHIN_TOLERANCE_FLOOR`` (99%) of elements satisfy that bound, and
  2. the maximum absolute error over all elements is within
     ``max_abs_error_limit(dtype)``.

The second is not implied by the first - in bf16 at ``|golden| = 1000`` the
elementwise allowance is already ~15.6, far past the limit of 1 - so a case
only passes when both hold. The process exit status is non-zero if any case
misses either criterion.

Usage (on an Ascend NPU, from a checkout that has the optimization):

    python benchmarks/scripts/penalties_accuracy_table.py

    # the full parameter grid instead of the curated case list
    python benchmarks/scripts/penalties_accuracy_table.py --full

    # machine-readable output
    python benchmarks/scripts/penalties_accuracy_table.py --format json

The reference implementation and the data generator are adapted from
``tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_penality.py``
so that the numbers in the table and the numbers the test suite asserts on come
from the same source.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401  # registers the npu device and torch.npu
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton

# ---------------------------------------------------------------------------
# BEFORE: the baseline kernel, vendored verbatim.
# ---------------------------------------------------------------------------
# Copied unchanged from vllm_ascend/worker/v2/sample/penalties.py as of the
# commit named below, so a reviewer can diff the two by eye. Only the two
# symbol names carry a `_baseline` prefix, to keep them distinct from the
# imported current implementation. Do not "fix" anything in here: the point of
# this block is to reproduce the old numerics exactly.
# Verify with:
#   git show 6ba8bc2dd:vllm_ascend/worker/v2/sample/penalties.py
# whose blob is 118b7269291ef4509baebe52af71f6733fe740a1. That revision is this
# PR's merge base, and the file is unchanged on main, so the vendored copy is
# the "before" for both.
BASELINE_SOURCE = (
    "vllm_ascend/worker/v2/sample/penalties.py @ 6ba8bc2dd (blob 118b7269), identical on main"
)

# The baseline hard-codes its vocab block size in the launcher.
BASELINE_BLOCK_SIZE = 4096

# Triton-Ascend rejects a launch whose `coreDim` exceeds this, which is what
# forces the baseline's `65535 // num_tokens` grid split in the first place.
BASELINE_MAX_CORE_DIM = 65535


@triton.jit
def _baseline_penalties_kernel(
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


class BaselineUnlaunchable(RuntimeError):
    """The baseline launcher cannot express this shape as a valid grid."""


def baseline_apply_penalties(
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
    BLOCK_SIZE = BASELINE_BLOCK_SIZE
    num_vocab_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    vocab_grid_size = min(num_vocab_blocks, BASELINE_MAX_CORE_DIM // num_tokens)
    # Not in the vendored original: the original would launch a zero-sized grid
    # here and silently leave `logits` untouched. Surfacing it as an error keeps
    # a structural limitation of the baseline out of the accuracy numbers, where
    # it would masquerade as a precision failure.
    if vocab_grid_size < 1:
        raise BaselineUnlaunchable(
            f"num_tokens={num_tokens} exceeds the coreDim limit "
            f"({BASELINE_MAX_CORE_DIM}), so vocab_grid_size collapses to 0"
        )
    _baseline_penalties_kernel[(num_tokens, vocab_grid_size)](
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
# Reference and data generation.
# ---------------------------------------------------------------------------
# Adapted from tests/e2e/nightly/single_node/ops/singlecard_ops/triton/
# test_penality.py. The arithmetic is unchanged; the per-token scalar reads are
# hoisted to host lists so the reference does not pay a device sync per token,
# which matters at num_tokens=2048.

# (repetition, frequency, presence). The kernel branches on each independently.
PENALTY_CONFIGS: dict[str, tuple[float, float, float]] = {
    "none": (1.0, 0.0, 0.0),
    "rep_only": (1.1, 0.0, 0.0),
    "freq_only": (1.0, 0.1, 0.0),
    "pres_only": (1.0, 0.0, 0.1),
    "rep_pres": (1.1, 0.0, 0.1),
    "all": (1.1, 0.1, 0.1),
}

# Asks for a batch holding both penalised and unpenalised requests, which no
# single entry above produces.
MIXED_PENALTY_CONFIG = "mixed"
# Asks for per-request random penalties.
RANDOM_PENALTY_CONFIG = "random"

# Reference accuracy standard, as (atol, rtol) per output dtype:
#
#     |actual_i - golden_i| <= atol + rtol * |golden_i|
#
# Written as powers of two so the constants are exactly the ones the standard
# names. fp32 is the one row where atol and rtol differ.
TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (2**-9, 2**-9),
    torch.bfloat16: (2**-6, 2**-6),
    torch.float32: (2**-16, 2**-10),
}

# First criterion: at least this fraction of elements must satisfy the
# elementwise mixed tolerance above.
WITHIN_TOLERANCE_FLOOR = 0.99

# Second, independent criterion: a ceiling on the largest absolute error
# anywhere in the tensor. 100% elementwise pass does not imply it - in bf16 at
# |golden| = 1000 the elementwise allowance is already ~15.6, well past the
# limit of 1 - so both have to be checked.
#
# The standard sets the ceiling to max(fixed_limit, 32 * ULP(1.0)), where
# ULP(1.0) is the spacing above 1.0 for the dtype, i.e. torch.finfo(dtype).eps
# (bf16 2**-7, fp16 2**-10, fp32 2**-23). For these three dtypes the fixed limit
# dominates (32 * eps is 0.25, 0.03125 and 3.8e-6), but the max() is kept so a
# coarser dtype widens the ceiling as the rule intends.
FIXED_MAX_ABS_ERROR_LIMITS: dict[torch.dtype, float] = {
    torch.float16: 0.1,
    torch.bfloat16: 1.0,
    torch.float32: 0.01,
}


def max_abs_error_limit(dtype: torch.dtype) -> float:
    """The standard's maximum absolute error limit for `dtype`."""
    return max(FIXED_MAX_ABS_ERROR_LIMITS[dtype], 32 * torch.finfo(dtype).eps)


DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}
DTYPE_BY_NAME = {name: dtype for dtype, name in DTYPE_NAMES.items()}


def pack_prompt_bin_mask(
    prompt_token_ids: list[np.ndarray],
    num_status: int,
    vocab_size: int,
    device: str,
) -> torch.Tensor:
    """Bit-pack per-request prompt token sets into int32 words."""
    num_packed = (vocab_size + 31) // 32
    dense = np.zeros((num_status, num_packed * 32), dtype=np.uint8)
    for row, ids in enumerate(prompt_token_ids):
        dense[row, ids] = 1
    packed = np.packbits(dense, axis=-1, bitorder="little")
    packed = np.ascontiguousarray(packed).view(np.int32)
    return torch.from_numpy(np.ascontiguousarray(packed)).to(device)


def unpack_prompt_bin_mask(prompt_bin_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Inverse of pack_prompt_bin_mask: [R, P] int32 -> [R, vocab_size] bool."""
    arr = np.ascontiguousarray(prompt_bin_mask.detach().cpu().numpy())
    rows = arr.shape[0]
    bits = np.unpackbits(arr.view(np.uint8).reshape(rows, -1), axis=-1, bitorder="little")
    bits = np.ascontiguousarray(bits[:, :vocab_size].astype(bool))
    return torch.from_numpy(bits).to(prompt_bin_mask.device)


def pytorch_apply_penalties(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> torch.Tensor:
    """Float32 PyTorch reference for the penalties kernel."""
    num_tokens, vocab_size = logits.shape
    device = logits.device
    dtype = logits.dtype

    logits_float = logits.float()

    prompt_masks_unpacked = unpack_prompt_bin_mask(prompt_bin_mask, vocab_size)

    # Hoisted to host once instead of one .item() per token per scalar.
    idx_mapping_host = idx_mapping.detach().cpu().tolist()
    local_pos_host = expanded_local_pos.detach().cpu().tolist()
    token_ids_host = token_ids.detach().cpu().tolist()
    rep_host = repetition_penalty.detach().cpu().tolist()
    freq_host = frequency_penalty.detach().cpu().tolist()
    pres_host = presence_penalty.detach().cpu().tolist()

    for token_idx in range(num_tokens):
        req_state_idx = int(idx_mapping_host[token_idx])

        rep_penalty = float(rep_host[req_state_idx])
        freq_penalty = float(freq_host[req_state_idx])
        pres_penalty = float(pres_host[req_state_idx])

        use_rep_penalty = rep_penalty != 1.0
        use_freq_penalty = freq_penalty != 0.0
        use_pres_penalty = pres_penalty != 0.0
        use_penalty = use_rep_penalty or use_freq_penalty or use_pres_penalty

        if not use_penalty:
            continue

        current_prompt_mask = prompt_masks_unpacked[req_state_idx]
        base_counts = output_bin_counts[req_state_idx].clone()

        # Cumulative draft counts from the earlier positions of this request.
        pos = int(local_pos_host[token_idx])
        start_idx_in_batch = token_idx - pos
        draft_counts = torch.zeros(vocab_size, device=device, dtype=torch.int32)

        for prev_pos in range(pos):
            prev_token_idx = start_idx_in_batch + prev_pos + 1
            if 0 <= prev_token_idx < num_tokens:
                prev_token = int(token_ids_host[prev_token_idx])
                if 0 <= prev_token < vocab_size:
                    draft_counts[prev_token] += 1

        total_counts = base_counts + draft_counts
        output_bin_mask = total_counts > 0

        if use_rep_penalty:
            need_scale = current_prompt_mask | output_bin_mask
            scale = torch.where(need_scale, rep_penalty, 1.0)

            pos_mask = logits_float[token_idx] > 0
            scale_factor = torch.where(pos_mask, 1.0 / scale, scale)
            logits_float[token_idx] *= scale_factor

        if use_freq_penalty:
            logits_float[token_idx] -= freq_penalty * total_counts.float()

        if use_pres_penalty:
            logits_float[token_idx] -= pres_penalty * output_bin_mask.float()

    return logits_float.to(dtype)


def create_test_data(
    num_tokens: int,
    vocab_size: int,
    num_status: int,
    num_speculative_tokens: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
    penalty_config: tuple[float, float, float] | None = None,
    mixed_penalties: bool = False,
    spec_decode_groups: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Build one penalties batch.

    penalty_config: every request gets exactly this ``(repetition, frequency,
        presence)`` triple. The kernel branches on each penalty independently,
        so pinning them is what makes per-branch coverage possible. ``None``
        gives per-request random penalties.
    mixed_penalties: alternate requests between active and neutral, so one
        batch holds both. Takes precedence over penalty_config.
    spec_decode_groups: lay the rows out the way ``expand_idx_mapping``
        actually does - ``num_status`` runs of ``num_speculative_tokens + 1``
        rows, ``expanded_local_pos`` ramping 0..k, a constant ``req_state_idx``
        per run. With random per-token values instead, ``start_idx = token_idx
        - pos`` points at arbitrary rows, which no real batch produces.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    logits = torch.randn(num_tokens, vocab_size, device=device, dtype=dtype)

    if mixed_penalties:
        rep, freq, presence = PENALTY_CONFIGS["all"]
        active = torch.arange(num_status, device=device) % 2 == 0
        repetition_penalty = torch.where(
            active,
            torch.full((num_status,), rep, device=device),
            torch.ones(num_status, device=device),
        ).to(torch.float32)
        frequency_penalty = (active * freq).to(torch.float32)
        presence_penalty = (active * presence).to(torch.float32)
    elif penalty_config is None:
        repetition_penalty = torch.ones(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.3:
                repetition_penalty[i] = torch.rand(1, device=device).item() * 0.8 + 0.6

        frequency_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.5:
                frequency_penalty[i] = torch.rand(1, device=device).item() * 0.2

        presence_penalty = torch.zeros(num_status, device=device, dtype=torch.float32)
        for i in range(num_status):
            if torch.rand(1) > 0.5:
                presence_penalty[i] = torch.rand(1, device=device).item() * 0.2
    else:
        rep, freq, presence = penalty_config
        repetition_penalty = torch.full((num_status,), rep, device=device, dtype=torch.float32)
        frequency_penalty = torch.full((num_status,), freq, device=device, dtype=torch.float32)
        presence_penalty = torch.full((num_status,), presence, device=device, dtype=torch.float32)

    token_ids = torch.randint(0, vocab_size, (num_tokens,), device=device, dtype=torch.int32)

    if spec_decode_groups:
        rows_per_req = num_speculative_tokens + 1
        flat_req = np.minimum(np.arange(num_tokens) // rows_per_req, num_status - 1)
        flat_pos = np.arange(num_tokens) % rows_per_req
        idx_mapping = torch.from_numpy(flat_req.astype(np.int32)).to(device)
        expanded_local_pos = torch.from_numpy(flat_pos.astype(np.int32)).to(device)
    else:
        idx_mapping = torch.randint(0, num_status, (num_tokens,), device=device, dtype=torch.int32)
        expanded_local_pos = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        for i in range(num_tokens):
            expanded_local_pos[i] = torch.randint(0, num_speculative_tokens + 1, (1,)).item()

    num_in_prompt = max(1, vocab_size // 20)
    prompt_token_ids = [
        rng.choice(vocab_size, size=num_in_prompt, replace=False).astype(np.int64) for _ in range(num_status)
    ]
    prompt_bin_mask = pack_prompt_bin_mask(prompt_token_ids, num_status, vocab_size, device)

    output_bin_counts = torch.zeros(num_status, vocab_size, device=device, dtype=torch.int32)
    num_output_tokens = max(1, vocab_size // 20)
    out_ids = torch.from_numpy(rng.integers(0, vocab_size, size=(num_status, num_output_tokens), dtype=np.int64)).to(
        device
    )
    out_vals = torch.from_numpy(rng.integers(1, 10, size=(num_status, num_output_tokens), dtype=np.int32)).to(device)
    output_bin_counts.scatter_(1, out_ids, out_vals)

    return (
        logits,
        idx_mapping,
        token_ids,
        expanded_local_pos,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        prompt_bin_mask,
        output_bin_counts,
    )


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------


# `output_bin_counts` is [num_status, vocab_size] int32, so num_status drives
# the dominant allocation: 2048 request states at vocab 151936 is 1.24 GB on its
# own, before the logits copies. Production sizes this by max_num_seqs, so cap
# the derived value at a realistic batch rather than at num_tokens.
MAX_DERIVED_STATUS = 512


@dataclasses.dataclass(frozen=True)
class Case:
    """One row of the accuracy table."""

    name: str
    num_tokens: int
    vocab_size: int
    num_speculative_tokens: int
    penalty: str
    num_status: int | None = None

    def status_count(self) -> int:
        if self.num_status is not None:
            return self.num_status
        derived = self.num_tokens // (self.num_speculative_tokens + 1)
        return max(1, min(derived, MAX_DERIVED_STATUS))

    def describe(self) -> str:
        spec = "-" if self.num_speculative_tokens == 0 else str(self.num_speculative_tokens)
        return f"{self.num_tokens} x {self.vocab_size}, spec={spec}, {self.penalty}"


# Real serving shapes: vocab 151936 is Qwen-family, 155648 is the large-vocab
# target the kernel is sized for, 12800 and 16384 cover the multi-block and
# exact-multiple vocab block boundaries.
DEFAULT_CASES: tuple[Case, ...] = (
    Case("decode-bs1", 1, 151936, 0, "all"),
    Case("decode-bs32", 32, 151936, 0, "all"),
    Case("decode-bs256", 256, 151936, 0, "all"),
    Case("decode-bs2048", 2048, 151936, 0, "all"),
    Case("spec4-bs256", 256, 151936, 3, "all"),
    Case("spec4-bs2048", 2048, 155648, 3, "all"),
    Case("mixed-bs512", 512, 151936, 3, MIXED_PENALTY_CONFIG),
    Case("rep-only-bs256", 256, 151936, 0, "rep_only"),
    Case("freq-only-bs256", 256, 151936, 0, "freq_only"),
    Case("pres-only-bs256", 256, 151936, 0, "pres_only"),
    Case("no-penalty-bs256", 256, 151936, 0, "none"),
    Case("small-vocab-bs64", 64, 12800, 0, "all"),
    Case("exact-block-bs64", 64, 16384, 0, "all"),
    Case("random-bs128", 128, 151936, 3, RANDOM_PENALTY_CONFIG),
)

DEFAULT_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float32)

# --full sweeps the grid instead of the curated list.
FULL_TOKENS = (1, 5, 64, 256, 2048)
FULL_VOCABS = (12800, 16384, 151936)
FULL_SPEC = (0, 3)
FULL_PENALTIES = (*sorted(PENALTY_CONFIGS), MIXED_PENALTY_CONFIG, RANDOM_PENALTY_CONFIG)


def full_cases() -> list[Case]:
    cases = []
    for num_tokens in FULL_TOKENS:
        for vocab_size in FULL_VOCABS:
            for spec in FULL_SPEC:
                for penalty in FULL_PENALTIES:
                    cases.append(
                        Case(
                            name=f"t{num_tokens}-v{vocab_size}-s{spec}-{penalty}",
                            num_tokens=num_tokens,
                            vocab_size=vocab_size,
                            num_speculative_tokens=spec,
                            penalty=penalty,
                        )
                    )
    return cases


# ---------------------------------------------------------------------------
# Measurement.
# ---------------------------------------------------------------------------


def _json_safe(value: float) -> float | None:
    """json.dumps emits bare Infinity/NaN, which is not valid JSON."""
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


@dataclasses.dataclass
class Deviation:
    """How far one result sits from another, scored against the standard."""

    max_abs: float
    max_rel: float
    num_exceeding: int
    total: int
    tol_ratio: float
    bitwise_equal: bool
    # None for an exact (atol=rtol=0) comparison, where the standard's two
    # criteria do not apply and only `bitwise_equal` is meaningful.
    max_abs_limit: float | None = None

    @property
    def within_rate(self) -> float:
        """Fraction of elements satisfying the elementwise mixed tolerance."""
        if self.total == 0:
            return 1.0
        return (self.total - self.num_exceeding) / self.total

    @property
    def meets_within_rate(self) -> bool:
        """Criterion 1: the within-tolerance proportion clears the floor."""
        return self.within_rate >= WITHIN_TOLERANCE_FLOOR

    @property
    def meets_max_abs(self) -> bool:
        """Criterion 2: the largest absolute error is within the ceiling."""
        if self.max_abs_limit is None:
            return True
        return self.max_abs <= self.max_abs_limit

    @property
    def passed(self) -> bool:
        """The standard requires both criteria to hold simultaneously."""
        return self.meets_within_rate and self.meets_max_abs

    def to_dict(self) -> dict:
        return {
            "max_abs": _json_safe(self.max_abs),
            "max_rel": _json_safe(self.max_rel),
            "num_exceeding": self.num_exceeding,
            "total": self.total,
            "within_rate": self.within_rate,
            "within_rate_floor": WITHIN_TOLERANCE_FLOOR,
            "max_abs_limit": self.max_abs_limit,
            "meets_within_rate": self.meets_within_rate,
            "meets_max_abs": self.meets_max_abs,
            "tol_ratio": _json_safe(self.tol_ratio),
            "bitwise_equal": self.bitwise_equal,
            "passed": self.passed,
        }


# Rows per reduction chunk. The comparands are [num_tokens, vocab_size]; doing
# this in one shot would allocate several more tensors of that size, which at
# 2048 x 151936 fp32 is 1.24 GB each.
DEVIATION_CHUNK_ROWS = 64


def deviation(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    max_abs_limit: float | None = None,
) -> Deviation:
    """Score `actual` against `expected` against the mixed-tolerance standard.

    ``num_exceeding`` counts the elements violating
    ``|actual - expected| <= atol + rtol * |expected|``, which drives the
    within-tolerance proportion; ``max_abs`` is compared against
    ``max_abs_limit`` separately, since the standard requires both.

    With ``atol == rtol == 0`` this degenerates into an exact comparison, and
    ``num_exceeding`` counts every element that differs at all.
    """
    exact = atol == 0.0 and rtol == 0.0
    tiny = torch.finfo(torch.float32).tiny

    bitwise_equal = True
    max_abs = 0.0
    max_rel = 0.0
    max_ratio = 0.0
    num_exceeding = 0

    num_rows = actual.shape[0]
    for start in range(0, num_rows, DEVIATION_CHUNK_ROWS):
        stop = min(start + DEVIATION_CHUNK_ROWS, num_rows)
        actual_chunk = actual[start:stop]
        expected_chunk = expected[start:stop]

        if bitwise_equal and not torch.equal(actual_chunk, expected_chunk):
            bitwise_equal = False

        a = actual_chunk.float()
        e = expected_chunk.float()
        abs_diff = (a - e).abs()
        max_abs = max(max_abs, float(abs_diff.max().item()))

        # Relative error is only meaningful where the reference is non-zero;
        # elsewhere max_abs already carries the information.
        nonzero = e != 0
        rel = torch.where(nonzero, abs_diff / e.abs().clamp_min(tiny), torch.zeros_like(abs_diff))
        max_rel = max(max_rel, float(rel.max().item()))

        if exact:
            num_exceeding += int((abs_diff > 0).sum().item())
        else:
            # atol > 0 keeps this strictly positive, so the ratio is finite.
            allowed = atol + rtol * e.abs()
            num_exceeding += int((abs_diff > allowed).sum().item())
            max_ratio = max(max_ratio, float((abs_diff / allowed).max().item()))

    if exact:
        max_ratio = float("inf") if num_exceeding else 0.0

    return Deviation(
        max_abs=max_abs,
        max_rel=max_rel,
        num_exceeding=num_exceeding,
        total=int(actual.numel()),
        tol_ratio=max_ratio,
        bitwise_equal=bitwise_equal,
        max_abs_limit=max_abs_limit,
    )


@dataclasses.dataclass
class Row:
    case: Case
    dtype: torch.dtype
    atol: float
    rtol: float
    before: Deviation | None
    after: Deviation
    before_vs_after: Deviation | None
    baseline_error: str | None

    def to_dict(self) -> dict:
        return {
            "case": self.case.name,
            "num_tokens": self.case.num_tokens,
            "vocab_size": self.case.vocab_size,
            "num_speculative_tokens": self.case.num_speculative_tokens,
            "num_status": self.case.status_count(),
            "penalty": self.case.penalty,
            "dtype": DTYPE_NAMES[self.dtype],
            "atol": self.atol,
            "rtol": self.rtol,
            "max_abs_limit": max_abs_error_limit(self.dtype),
            "before_vs_reference": None if self.before is None else self.before.to_dict(),
            "after_vs_reference": self.after.to_dict(),
            "before_vs_after": None if self.before_vs_after is None else self.before_vs_after.to_dict(),
            "baseline_error": self.baseline_error,
        }


def _penalty_kwargs(penalty: str) -> dict:
    if penalty == MIXED_PENALTY_CONFIG:
        return {"mixed_penalties": True}
    if penalty == RANDOM_PENALTY_CONFIG:
        return {"penalty_config": None}
    return {"penalty_config": PENALTY_CONFIGS[penalty]}


def run_case(
    case: Case,
    dtype: torch.dtype,
    device: str,
    seed: int,
    apply_penalties_after,
) -> Row:
    atol, rtol = TOLERANCES[dtype]
    abs_limit = max_abs_error_limit(dtype)
    data = create_test_data(
        num_tokens=case.num_tokens,
        vocab_size=case.vocab_size,
        num_status=case.status_count(),
        num_speculative_tokens=case.num_speculative_tokens,
        device=device,
        dtype=dtype,
        seed=seed,
        spec_decode_groups=True,
        **_penalty_kwargs(case.penalty),
    )
    logits = data[0]
    pristine = logits.clone()
    operands = data[1:]

    # Reference first, from an untouched copy of the logits. Neither kernel nor
    # the reference mutates `operands`, so one batch serves all three runs.
    expected = pytorch_apply_penalties(pristine.clone(), *operands)

    # BEFORE: the vendored baseline. Kept in its own buffer because the
    # before-vs-after comparison needs both results at once.
    baseline_error: str | None = None
    before_result: torch.Tensor | None = None
    try:
        logits.copy_(pristine)
        baseline_apply_penalties(logits, *operands)
        torch.npu.synchronize()
        before_result = logits.clone()
    except BaselineUnlaunchable as exc:
        baseline_error = str(exc)
    except RuntimeError as exc:
        # A shape the baseline cannot run is a result worth reporting, not a
        # reason to lose the rest of the table.
        baseline_error = f"baseline launch failed: {exc}"

    # AFTER: the implementation this checkout provides. Left in `logits` rather
    # than cloned, so the peak holds three full-size buffers and not four.
    logits.copy_(pristine)
    apply_penalties_after(logits, *operands)
    torch.npu.synchronize()

    del pristine

    after_dev = deviation(logits, expected, atol, rtol, abs_limit)
    before_dev = None if before_result is None else deviation(before_result, expected, atol, rtol, abs_limit)
    # Compared with atol=rtol=0, so `num_exceeding` counts every element that
    # differs at all: a pure scheduling change should leave this at zero.
    pair_dev = None if before_result is None else deviation(logits, before_result, 0.0, 0.0)

    del data, logits, expected, before_result
    torch.npu.empty_cache()

    return Row(
        case=case,
        dtype=dtype,
        atol=atol,
        rtol=rtol,
        before=before_dev,
        after=after_dev,
        before_vs_after=pair_dev,
        baseline_error=baseline_error,
    )


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.3e}"


def _fmt_rate(rate: float) -> str:
    if rate == 1.0:
        return "100%"
    return f"{rate * 100:.4f}%"


def _failed(row: Row) -> bool:
    """Whether any measured result misses the standard."""
    if not row.after.passed:
        return True
    return row.before is not None and not row.before.passed


def _verdict(row: Row) -> str:
    # The standard comes first: two kernels can agree with each other and still
    # both be wrong, so bitwise equality is only reported once both pass.
    if _failed(row):
        which = []
        if row.before is not None and not row.before.passed:
            which.append("before")
        if not row.after.passed:
            which.append("after")
        return f"**FAILS STANDARD** ({', '.join(which)})"
    if row.baseline_error is not None:
        return "after only"
    assert row.before_vs_after is not None
    if row.before_vs_after.bitwise_equal:
        return "identical"
    return "both within standard"


def render_markdown(rows: list[Row], env: dict) -> str:
    out: list[str] = []
    out.append("### Operator accuracy: `_penalties_kernel` (before vs after)")
    out.append("")
    out.append(
        "`before` is the baseline kernel on a `(num_tokens, min(num_vocab_blocks, "
        "65535 // num_tokens))` grid; `after` is the vector-core tiled kernel in this PR. "
        "Both are scored against the same float32 PyTorch reference under the reference "
        "accuracy standard, and compared directly against each other. The change only "
        "reassigns tiles to programs, so the expected outcome is a bitwise identical result."
    )
    out.append("")
    out.append(
        "| Case | Shape (tokens x vocab) | spec | Penalties | dtype | "
        "max abs err (before) | max abs err (after) | max abs limit | "
        "within tol (before / after) | before == after | Result |"
    )
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        case = row.case
        spec = "-" if case.num_speculative_tokens == 0 else str(case.num_speculative_tokens)
        before_abs = "n/a" if row.before is None else _fmt(row.before.max_abs)
        before_rate = "n/a" if row.before is None else _fmt_rate(row.before.within_rate)
        if row.before_vs_after is None:
            equal = "n/a"
        else:
            equal = "yes" if row.before_vs_after.bitwise_equal else f"no ({row.before_vs_after.num_exceeding} elems)"
        out.append(
            f"| `{case.name}` "
            f"| {case.num_tokens} x {case.vocab_size} "
            f"| {spec} "
            f"| {case.penalty} "
            f"| {DTYPE_NAMES[row.dtype]} "
            f"| {before_abs} "
            f"| {_fmt(row.after.max_abs)} "
            f"| {max_abs_error_limit(row.dtype):g} "
            f"| {before_rate} / {_fmt_rate(row.after.within_rate)} "
            f"| {equal} "
            f"| {_verdict(row)} |"
        )
    out.append("")

    out.append("Scored against the reference accuracy standard, mixed tolerance")
    out.append("`|actual - golden| <= atol + rtol * |golden|`, both criteria required:")
    out.append("")
    out.append("| Output dtype | rtol | atol | max abs error limit |")
    out.append("| --- | --- | --- | --- |")
    for dtype in sorted(TOLERANCES, key=lambda d: DTYPE_NAMES[d]):
        atol, rtol = TOLERANCES[dtype]
        out.append(
            f"| {DTYPE_NAMES[dtype].upper()} "
            f"| {rtol:g} "
            f"| {atol:g} "
            f"| {max_abs_error_limit(dtype):g} |"
        )
    out.append("")
    out.append(
        "1. the proportion of elements meeting the elementwise mixed tolerance is at least "
        f"{WITHIN_TOLERANCE_FLOOR * 100:g}% (the `within tol` column), and"
    )
    out.append(
        "2. the maximum absolute error over all elements is within the limit above "
        "(the `max abs err` columns against `max abs limit`)."
    )
    out.append("")
    out.append(
        "The limit is `max(fixed_limit, 32 * ULP(1.0))`; for these dtypes the fixed limit "
        "dominates. `before == after` is an exact comparison, independent of the tolerance."
    )
    out.append("")

    unlaunchable = [r for r in rows if r.baseline_error is not None]
    if unlaunchable:
        out.append("Cases the baseline could not run, measured for the new kernel only:")
        out.append("")
        for row in unlaunchable:
            out.append(f"- `{row.case.name}` ({DTYPE_NAMES[row.dtype]}): {row.baseline_error}")
        out.append("")

    failures = [r for r in rows if _failed(r)]
    identical = sum(1 for r in rows if r.before_vs_after is not None and r.before_vs_after.bitwise_equal)
    out.append(
        f"**Summary**: {len(rows)} cases, {identical} bitwise identical before vs after, "
        f"{sum(1 for r in rows if r.after.passed)}/{len(rows)} meeting the standard, "
        f"{len(failures)} failing it."
    )
    out.append("")
    out.append("<details><summary>Environment</summary>")
    out.append("")
    for key, value in env.items():
        out.append(f"- {key}: {value}")
    out.append("")
    out.append("</details>")
    return "\n".join(out)


def render_csv(rows: list[Row]) -> str:
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "case",
            "num_tokens",
            "vocab_size",
            "num_speculative_tokens",
            "penalty",
            "dtype",
            "atol",
            "rtol",
            "max_abs_limit",
            "before_max_abs",
            "after_max_abs",
            "before_within_rate",
            "after_within_rate",
            "before_passed",
            "after_passed",
            "bitwise_equal",
            "verdict",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.case.name,
                row.case.num_tokens,
                row.case.vocab_size,
                row.case.num_speculative_tokens,
                row.case.penalty,
                DTYPE_NAMES[row.dtype],
                row.atol,
                row.rtol,
                max_abs_error_limit(row.dtype),
                "" if row.before is None else row.before.max_abs,
                row.after.max_abs,
                "" if row.before is None else row.before.within_rate,
                row.after.within_rate,
                "" if row.before is None else row.before.passed,
                row.after.passed,
                "" if row.before_vs_after is None else row.before_vs_after.bitwise_equal,
                _verdict(row),
            ]
        )
    return buf.getvalue()


def collect_env() -> dict:
    env: dict[str, object] = {}
    env["python"] = sys.version.split()[0]
    env["torch"] = torch.__version__
    env["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    try:
        env["device"] = torch.npu.get_device_name(torch.npu.current_device())
    except Exception:
        env["device"] = "unknown"
    try:
        env["vector cores"] = get_vectorcore_num()
    except Exception as exc:
        env["vector cores"] = f"unavailable ({exc})"
    try:
        rev = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if rev.returncode == 0:
            env["vllm-ascend commit"] = rev.stdout.strip()
    except Exception:
        pass
    env["baseline source"] = BASELINE_SOURCE
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true", help="sweep the whole grid instead of the curated case list")
    parser.add_argument(
        "--dtypes",
        default=",".join(DTYPE_NAMES[d] for d in DEFAULT_DTYPES),
        help="comma-separated subset of fp32,fp16,bf16",
    )
    parser.add_argument("--case", action="append", default=None, help="run only this case by name; repeatable")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--format", choices=("markdown", "json", "csv"), default="markdown")
    parser.add_argument("-o", "--output", default=None, help="write to this file instead of stdout")
    args = parser.parse_args()

    try:
        dtypes = [DTYPE_BY_NAME[name.strip()] for name in args.dtypes.split(",") if name.strip()]
    except KeyError as exc:
        parser.error(f"unknown dtype {exc}; choose from {', '.join(DTYPE_BY_NAME)}")

    cases = full_cases() if args.full else list(DEFAULT_CASES)
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            parser.error(f"no such case(s): {', '.join(sorted(missing))}")
    if not cases:
        parser.error("no cases selected")

    # Pin the device before anything allocates, so torch.npu.synchronize() and
    # empty_cache() act on the same card the tensors live on.
    torch.npu.set_device(int(args.device.split(":")[1]) if ":" in args.device else 0)

    # apply_penalties reads get_vectorcore_num(), which asserts unless device
    # properties were initialised. In production AscendWorker.init_device does
    # this once; a standalone script has to do it itself.
    init_device_properties_triton()

    from vllm_ascend.worker.v2.sample import penalties as current_penalties

    # The optimized revision introduces VOCAB_BLOCK_SIZE; the baseline keeps the
    # block size local to its launcher. Used only to warn, not to switch paths.
    is_optimized = hasattr(current_penalties, "VOCAB_BLOCK_SIZE")
    if not is_optimized:
        print(
            "warning: the imported vllm_ascend.worker.v2.sample.penalties looks like "
            "the pre-optimization revision, so the 'before' and 'after' columns will "
            "measure the same kernel. Run this from a checkout that has the change.",
            file=sys.stderr,
        )

    rows: list[Row] = []
    total = len(cases) * len(dtypes)
    done = 0
    for case in cases:
        for dtype in dtypes:
            done += 1
            print(f"[{done}/{total}] {case.name} ({DTYPE_NAMES[dtype]}) {case.describe()}", file=sys.stderr)
            rows.append(
                run_case(
                    case=case,
                    dtype=dtype,
                    device=args.device,
                    seed=args.seed,
                    apply_penalties_after=current_penalties.apply_penalties,
                )
            )

    if args.format == "markdown":
        text = render_markdown(rows, collect_env())
    elif args.format == "csv":
        text = render_csv(rows)
    else:
        text = json.dumps(
            {"environment": {k: str(v) for k, v in collect_env().items()}, "rows": [r.to_dict() for r in rows]},
            indent=2,
        )

    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)

    return 1 if any(_failed(r) for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
