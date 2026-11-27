# SISO RoPE

Source: `vllm_ascend/ops/triton/rope.py` (wrapper: `rope_forward_triton_siso`).

## Description

- **Function**: Apply rotary position embeddings to one `[num_tokens, num_heads, head_dim]` tensor. The shared SFA indexer in `vllm_ascend/attention/indexer.py` uses the wrapper for Q and K. Its K projection has one head. Mistral Large 4 uses the interleaved layout when indexer RoPE is enabled.
- **Formula**: Compute `y0 = x0 * cos - x1 * sin` and `y1 = x1 * cos + x0 * sin` in FP32, then cast back to the input dtype. NeoX pairs the two contiguous halves of the rotary prefix. The interleaved layout pairs adjacent even and odd elements. The unrotated tail is preserved.
- **Algorithm flow**:
  1. Make the input contiguous as needed, select the rotation-table mode, and use the provided rotary dimension.
  2. For one head with preselected tables, tile across token rows. Each program reads separate cosine and sine coefficients for each row and masks tail rows and padded rotary lanes.
  3. Choose a power-of-two token tile from the ratio of tokens to vector cores, bounded by 16 and 128. These bounds were measured on a 64-vector-core Ascend950DT at rotary dimension 64 in BF16. The floor reduces per-program overhead for small batches.
  4. For multiple heads or indexed cache inputs, use the existing kernel. It tiles over heads and loops over rows with at most one program per vector core.
- **Supported modes**: The wrapper serves the existing Ascend SISO RoPE call sites. The token-tiled path has eager operator validation on Ascend950DT in FP16 and BF16. A2/A3 and ACL graph capture and replay for this path remain unverified.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data type | Data format |
| --- | --- | --- | --- | --- |
| `qk` | Input / Output | Tensor to rotate. The return value is the rotated tensor. | FP16 / BF16 / FP32 | `[num_tokens, num_heads, head_dim]` |
| `cos`, `sin` | Input | Position-selected coefficients with one row per token. | Floating point | `[num_tokens, rope_dim / 2]` after flattening each row |
| `cos_sin_cache` | Input | Alternative rotation table containing cosine followed by sine. | Floating point | `[num_positions, rope_dim]` |
| `positions` | Input | Indices into the alternative rotation table. | Integer indices | `[num_tokens]` |
| `rope_dim` | Attribute | Width of the rotary prefix. | Integer | Scalar |
| `is_neox_style` | Attribute | Selects half-split NeoX or adjacent-pair interleaved rotation. | Boolean | Scalar |

## Constraints

- Supply either `cos` and `sin`, or `cos_sin_cache` and `positions`. Provide a positive even `rope_dim` no larger than `head_dim`. Coefficients within a row are contiguous and preselected tables carry one copy of each coefficient.
- Contiguous input is modified in place. With a noncontiguous input the wrapper rotates a contiguous copy, so callers must use the returned tensor.
- The token-tiled path specializes on `BLOCK_M`, `rope_dim`, the padded half width derived from `rope_dim`, and layout. Pointer dtypes and alignment, and the Triton backend's specialization classes for token count and row strides, can create additional variants. A different grid alone does not require recompilation. Startup and serving compile counts are unmeasured.

## Origin and Differences

- **Origin**: The existing Ascend SISO kernel uses a single tensor and shares the RoPE arithmetic of the paired Q/K kernel.
- **Differences**: The new path widens the vector tile over tokens for a single head, keeping position-specific coefficients and the existing in-place semantics. The wrapper selects the kernel without changes to model call sites.

## Test Cases

The SISO test compares against a native PyTorch reference with `atol=rtol=1e-3`. Single-head cases cover both layouts and dtypes across decode and prefill sizes, partial and full rotary widths, a non-power-of-two width, and masked tails. The 2,048-token and 8,191-token cases exercise intermediate and large tile sizes.

```bash
python3 -m pytest --confcutdir=tests/e2e/nightly/single_node/ops -q \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_rope.py \
  -k test_rotary_embedding_triton_kernel_siso
```
