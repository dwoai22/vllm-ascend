#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Mistral Large 4 SSMax log positions must come from attention metadata.

SSMax scales the query by ``s * log(pos + 1) + b``, one value per query row.
The model's eager forward publishes that tensor on the attention impl, but the
attention math runs inside the ``mla_forward`` custom op, which carries no log
positions in its signature. A captured decode graph therefore bakes the address
the attribute held at capture time, and at capture warmup the attribute still
holds the previous, longer run's tensor -- the row-count mismatch these tests
pin down. Deriving the value from the metadata positions (a slice of the
runner's fixed-address positions buffer) keeps it inside the captured region.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


class _SSMax:
    """Upstream SSMaxScaling.forward's shape contract, without vLLM imports."""

    def __init__(self, num_heads: int) -> None:
        self.s = torch.ones(num_heads)
        self.b = torch.zeros(num_heads)

    def __call__(self, q, log_positions, out_head_dim=None):
        if out_head_dim is not None:
            q = torch.nn.functional.pad(q, (0, out_head_dim - q.shape[-1]))
        scale = log_positions.to(q.dtype)[:, None, None] * self.s.to(q.dtype)[:, None] + self.b.to(q.dtype)[:, None]
        # The failure the decode graph hit: q rows and scale rows disagree.
        return q * scale


def _impl(*, num_tokens_published: int, num_heads: int = 4, head_dim: int = 8):
    """An impl carrying only what the Mistral query path reads."""
    impl = AscendSFAImpl.__new__(AscendSFAImpl)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.local_num_heads = num_heads
    impl.qk_head_dim = head_dim
    impl.qk_nope_head_dim = head_dim
    impl.qk_rope_head_dim = 0
    impl.mistral_use_rope = True
    impl.mistral_ssmax = _SSMax(num_heads)
    # A stale publish from an earlier, larger batch -- exactly the state the
    # capture warmup finds.
    impl.mistral_log_positions = torch.zeros(num_tokens_published)
    impl.q_proj = lambda x: (x.new_zeros(x.shape[0], num_heads * head_dim),)
    impl.W_UK_T = torch.zeros(num_heads, head_dim, head_dim)
    return impl


def _metadata(num_tokens: int):
    return SimpleNamespace(positions=torch.arange(num_tokens, dtype=torch.int64))


def test_stale_published_log_positions_reproduce_the_decode_graph_mismatch():
    """Without the metadata-derived value the published tensor still wins."""
    impl = _impl(num_tokens_published=512)
    hidden_states = torch.zeros(16, 32)

    with pytest.raises(RuntimeError, match="must match the size of tensor"):
        impl._mistral_q_proj_and_k_up_proj(hidden_states)


def test_log_positions_are_derived_from_metadata_positions():
    impl = _impl(num_tokens_published=512)
    num_tokens = 16

    log_positions = impl._mistral_log_positions(_metadata(num_tokens), num_tokens)

    assert log_positions.shape == (num_tokens,)
    assert log_positions.dtype == torch.float32
    expected = torch.log(torch.arange(num_tokens, dtype=torch.float32) + 1)
    assert torch.allclose(log_positions, expected)


def test_derived_log_positions_make_the_query_path_agree_with_the_batch():
    """The capture-warmup shape that used to raise now goes through."""
    impl = _impl(num_tokens_published=512)
    num_tokens = 16
    hidden_states = torch.zeros(num_tokens, 32)

    log_positions = impl._mistral_log_positions(_metadata(num_tokens), num_tokens)
    ql_nope, q_pe = impl._mistral_q_proj_and_k_up_proj(hidden_states, log_positions)

    assert ql_nope.shape[0] == num_tokens
    assert q_pe.shape[0] == num_tokens


def test_metadata_without_positions_keeps_the_published_fallback():
    """NoPE metadata and test doubles carry no positions."""
    impl = _impl(num_tokens_published=16)

    log_positions = impl._mistral_log_positions(SimpleNamespace(positions=None), 16)

    assert log_positions is impl.mistral_log_positions


def test_short_metadata_positions_are_rejected_rather_than_broadcast():
    """A silent slice would scale the wrong rows, so fail loudly instead."""
    impl = _impl(num_tokens_published=16)

    with pytest.raises(RuntimeError, match="one position per query row"):
        impl._mistral_log_positions(_metadata(8), 16)


def test_positions_are_sliced_to_the_query_rows():
    """Padded metadata must be narrowed, not handed over whole."""
    impl = _impl(num_tokens_published=16)

    log_positions = impl._mistral_log_positions(_metadata(64), 16)

    assert log_positions.shape == (16,)
