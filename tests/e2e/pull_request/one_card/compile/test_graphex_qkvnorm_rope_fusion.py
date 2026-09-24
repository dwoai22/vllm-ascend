import copy

import npugraph_ex as nge
import numpy as np
import pytest
import torch
import torch.nn as nn
import vllm.config
from vllm.config import ModelConfig, VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.system_utils import update_environment_variables

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.compilation.passes.qknorm_rope_fusion_pass import QKVNormRopeFusionPattern
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

MAX_POSITION_EMBEDDING = 262144

# Gemma4 alternates two attention shapes: sliding layers rotate a 256-dim head
# with 16 KV heads, full attention layers a 512-dim head with 4 KV heads. Both
# have to fuse, which is why the pass registers one pattern per shape.
GEMMA4_SLIDING_SHAPE = (256, 8, 4)
GEMMA4_FULL_SHAPE = (512, 8, 1)

# Sliding layers use plain RotaryEmbedding; full attention layers use
# proportional RoPE, which builds a Gemma4RotaryEmbedding.
GEMMA4_SLIDING_ROPE = {"rope_type": "default", "rope_theta": 10000.0}
GEMMA4_FULL_ROPE = {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0}


def find_op(gm, op_default):
    return any(node.op == "call_function" and node.target == op_default for node in gm.graph.nodes)


def create_pattern_wrapper(assert_func):
    original_func = nge.npu_fx_compiler._optimize_fx

    def wrapper(gm, example_inputs=None, config=None):
        ret = original_func(gm, example_inputs, config)
        graph_after = copy.deepcopy(gm)
        assert_func(graph_after)
        return ret

    return wrapper


@pytest.fixture(scope="module", autouse=True)
def init_triton():
    init_device_properties_triton()


class ModelQKVNormRope(nn.Module):
    """The Gemma4 pre-attention chain: q/k norm, RoPE and a weight-less v norm.

    `v_weight` is a buffer of ones rather than a parameter, matching
    `RMSNorm(head_dim, has_weight=False)`, which still passes a ones weight to
    `npu_rms_norm`.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dtype: torch.dtype = torch.bfloat16,
        eps: float = 1e-6,
        device="npu",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.eps = eps

        self.q_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.k_weight = nn.Parameter(torch.randn(head_dim, dtype=dtype, device=device))
        self.register_buffer("v_weight", torch.ones(head_dim, dtype=dtype, device=device))

    def forward(self, qkv, cos_sin_cache, positions):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_by_head = q.unflatten(-1, (self.num_heads, self.head_dim))
        q_norm_out, _ = torch.ops.npu.npu_rms_norm(q_by_head, self.q_weight, self.eps)

        k_by_head = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k_norm_out, _ = torch.ops.npu.npu_rms_norm(k_by_head, self.k_weight, self.eps)

        q_flat = q_norm_out.flatten(-2, -1)
        k_flat = k_norm_out.flatten(-2, -1)
        q_rope, k_rope = torch.ops.vllm.npu_rotary_embedding(
            positions, q_flat, k_flat, cos_sin_cache, self.head_dim, self.head_dim, True
        )

        v_by_head = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v_norm_out, _ = torch.ops.npu.npu_rms_norm(v_by_head, self.v_weight, self.eps)
        v_flat = v_norm_out.flatten(-2, -1)

        return q_rope, k_rope, v_flat


def assert_qkvnorm_rope_fusion(after_gm, expect_fused=True):
    check_rules = [
        (torch.ops.vllm.qkv_rmsnorm_rope_vnorm.default, expect_fused),
        (torch.ops.npu.npu_rms_norm.default, not expect_fused),
        (torch.ops.vllm.npu_rotary_embedding.default, not expect_fused),
    ]
    for torch_op, expect_exist in check_rules:
        found = find_op(after_gm, torch_op)
        if expect_exist:
            assert found, f"Expected operator '{torch_op}' but not find"
        else:
            assert not found, f"Not expected operator '{torch_op}' but find"


def test_pattern_key_is_distinct_per_shape():
    """Both Gemma4 shapes must register; a shared key would drop the second."""
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=torch.bfloat16))
    keys = {
        QKVNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=1e-6,
        ).pattern_key()
        for head_dim, num_heads, num_kv_heads in (GEMMA4_SLIDING_SHAPE, GEMMA4_FULL_SHAPE)
    }
    assert len(keys) == 2, f"expected one key per shape, got {keys}"


def run_fused_shape(vllm_config, shape, dtype, eps, num_tokens):
    head_dim, num_heads, num_kv_heads = shape
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size

    model = ModelQKVNormRope(head_dim, num_heads, num_kv_heads, dtype, eps, device="npu").to("npu")

    qkv = torch.randn(num_tokens, qkv_size, device="npu", dtype=dtype)
    cos_sin_cache = torch.from_numpy(np.random.uniform(0, 1, [MAX_POSITION_EMBEDDING, head_dim])).to(dtype).npu()
    positions = torch.randint(low=0, high=MAX_POSITION_EMBEDDING, size=(num_tokens,), dtype=torch.int64, device="npu")

    with torch.no_grad():
        original_optimize = nge.npu_fx_compiler._optimize_fx
        nge.npu_fx_compiler._optimize_fx = create_pattern_wrapper(
            lambda gm: assert_qkvnorm_rope_fusion(gm, expect_fused=True)
        )
        try:
            compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
            compiled_model(qkv, cos_sin_cache, positions)
        finally:
            nge.npu_fx_compiler._optimize_fx = original_optimize


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-6])
def test_qkvnorm_rope_fusion(dtype: torch.dtype, num_tokens: int, eps: float):
    """Both Gemma4 shapes fuse when both patterns are registered up front.

    Registration order matches QKNormRopeFusionPass.__init__, which registers
    every shape before any graph is compiled.
    """
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))
    with vllm.config.set_current_vllm_config(vllm_config):
        update_environment_variables(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)

    shapes = (GEMMA4_SLIDING_SHAPE, GEMMA4_FULL_SHAPE)
    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        from torch._inductor.pattern_matcher import PatternMatcherPass

        pm_pass = PatternMatcherPass()
        for head_dim, num_heads, num_kv_heads in shapes:
            QKVNormRopeFusionPattern(
                vllm_config=vllm_config,
                head_dim=head_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                eps=eps,
            ).register(pm_pass)

        for shape in shapes:
            run_fused_shape(vllm_config, shape, dtype, eps, num_tokens)


class ModelGemma4PreAttention(nn.Module):
    """Gemma4's pre-attention chain built from the real modules.

    Unlike ModelQKVNormRope this goes through RMSNorm and get_rope rather than
    calling the fused ops directly, so it exercises what the model actually
    lowers to: whether AscendRMSNorm emits npu_rms_norm for a weight-less norm,
    and whether Gemma4RotaryEmbedding reaches npu_rotary_embedding.
    """

    def __init__(self, head_dim, num_heads, num_kv_heads, rope_parameters, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)
        self.v_norm = RMSNorm(head_dim, eps=eps, has_weight=False)
        self.rotary_emb = get_rope(
            head_dim,
            max_position=MAX_POSITION_EMBEDDING,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

    def forward(self, qkv, positions):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)

        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        return q, k, v


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [257])
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize(
    "shape,rope_parameters",
    [(GEMMA4_SLIDING_SHAPE, GEMMA4_SLIDING_ROPE), (GEMMA4_FULL_SHAPE, GEMMA4_FULL_ROPE)],
    ids=["sliding", "full_attention"],
)
def test_qkvnorm_rope_fusion_real_modules(dtype, num_tokens, eps, shape, rope_parameters):
    head_dim, num_heads, num_kv_heads = shape
    vllm_config = VllmConfig(model_config=ModelConfig(dtype=dtype))
    with vllm.config.set_current_vllm_config(vllm_config):
        update_environment_variables(
            {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )
        init_distributed_environment()
        ensure_model_parallel_initialized(1, 1)

    with vllm.config.set_current_vllm_config(vllm_config), set_ascend_forward_context(None, vllm_config):
        from torch._inductor.pattern_matcher import PatternMatcherPass

        pm_pass = PatternMatcherPass()
        QKVNormRopeFusionPattern(
            vllm_config=vllm_config,
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            eps=eps,
        ).register(pm_pass)

        model = ModelGemma4PreAttention(head_dim, num_heads, num_kv_heads, rope_parameters, eps).to("npu")
        print(f"rotary_emb is {type(model.rotary_emb).__name__}")

        qkv_size = num_heads * head_dim + 2 * num_kv_heads * head_dim
        qkv = torch.randn(num_tokens, qkv_size, device="npu", dtype=dtype)
        positions = torch.randint(
            low=0, high=MAX_POSITION_EMBEDDING, size=(num_tokens,), dtype=torch.int64, device="npu"
        )

        with torch.no_grad():
            original_optimize = nge.npu_fx_compiler._optimize_fx
            nge.npu_fx_compiler._optimize_fx = create_pattern_wrapper(
                lambda gm: assert_qkvnorm_rope_fusion(gm, expect_fused=True)
            )
            try:
                compiled_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=True)
                compiled_model(qkv, positions)
            finally:
                nge.npu_fx_compiler._optimize_fx = original_optimize
