# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MoE with non-gated activations (*_no_mul).

These tests verify that MoE layers work correctly with activations like
silu_no_mul, gelu_no_mul, relu2_no_mul where the activation output dimension
equals N (not N // 2 like gated activations).
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts,
    fused_experts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExperts,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

# Test parameters
M_SIZES = [1, 16, 64]
N_SIZES = [128, 256]
K_SIZES = [64, 128]
TOPK_VALUES = [1, 2]
NUM_EXPERTS = 8
NO_MUL_ACTIVATIONS = [
    MoEActivation.SILU_NO_MUL,
    MoEActivation.GELU_NO_MUL,
    MoEActivation.RELU2_NO_MUL,
]


def make_test_tensors(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    """Create test tensors for MoE with non-gated activation.

    For non-gated activations (*_no_mul):
    - w1: (E, N, K) - projects from K to N
    - w2: (E, K, N) - projects from N back to K (note: N, not N//2)
    """
    hidden_states = torch.randn(m, k, dtype=dtype, device=device)

    # For non-gated: w1 projects K -> N, w2 projects N -> K
    w1 = torch.randn(num_experts, n, k, dtype=dtype, device=device) * 0.1
    w2 = torch.randn(num_experts, k, n, dtype=dtype, device=device) * 0.1

    topk_weights = torch.ones(m, topk, dtype=torch.float32, device=device) / topk
    topk_ids = torch.randint(0, num_experts, (m, topk), device=device)

    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@pytest.mark.parametrize("m", M_SIZES)
@pytest.mark.parametrize("n", N_SIZES)
@pytest.mark.parametrize("k", K_SIZES)
@pytest.mark.parametrize("topk", TOPK_VALUES)
@pytest.mark.parametrize("activation", NO_MUL_ACTIVATIONS)
@torch.inference_mode()
def test_triton_experts_no_mul_activation(
    m: int,
    n: int,
    k: int,
    topk: int,
    activation: MoEActivation,
):
    hidden_states, w1, w2, topk_weights, topk_ids = make_test_tensors(
        m, n, k, NUM_EXPERTS, topk
    )

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_shape, ws2_shape, out_shape = experts.workspace_shapes(
        M=m,
        N=n,
        K=k,
        topk=topk,
        global_num_experts=NUM_EXPERTS,
        local_num_experts=NUM_EXPERTS,
        expert_tokens_meta=None,
        activation=activation,
    )

    # Verify workspace shapes are correct for no_mul activation
    # workspace1 should handle activation_out_dim = N (not N//2)
    assert ws1_shape == (m, topk, max(n, k)), (
        f"workspace1 shape mismatch: expected {(m, topk, max(n, k))}, got {ws1_shape}"
    )
    # workspace2 should handle max(N, K) for intermediate_cache1/cache3
    assert ws2_shape == (m, topk, max(n, k)), (
        f"workspace2 shape mismatch: expected {(m, topk, max(n, k))}, got {ws2_shape}"
    )
    assert out_shape == (m, k), (
        f"output shape mismatch: expected {(m, k)}, got {out_shape}"
    )

    workspace1 = torch.empty(
        ws1_shape[0] * ws1_shape[1] * ws1_shape[2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    workspace2 = torch.empty(
        ws2_shape[0] * ws2_shape[1] * ws2_shape[2],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    output = torch.zeros(m, k, dtype=hidden_states.dtype, device=hidden_states.device)

    experts.apply(
        output=output,
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=activation,
        global_num_experts=NUM_EXPERTS,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=workspace1,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )

    assert output.shape == (m, k), f"Expected shape {(m, k)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    assert output.abs().sum() > 0, "Output is all zeros"


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_workspace_shapes_no_mul_vs_gated():
    """Test that workspace shapes differ correctly between gated and non-gated."""
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

    M, N, K, topk = 64, 256, 128, 2

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    ws1_no_mul, _, out_no_mul = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU_NO_MUL
    )

    ws1_gated, _, out_gated = experts.workspace_shapes(
        M, N, K, topk, 8, 8, None, MoEActivation.SILU
    )

    # For no_mul: activation_out_dim = N
    # For gated: activation_out_dim = N // 2
    # workspace1 should use max(activation_out_dim, K)
    activation_out_dim_no_mul = N
    activation_out_dim_gated = N // 2

    assert ws1_no_mul[2] == max(activation_out_dim_no_mul, K), (
        f"no_mul workspace1 last dim should be max({activation_out_dim_no_mul}, {K})"
    )
    assert ws1_gated[2] == max(activation_out_dim_gated, K), (
        f"gated workspace1 last dim should be max({activation_out_dim_gated}, {K})"
    )

    # Output shapes should be the same
    assert out_no_mul == out_gated == (M, K)


@pytest.mark.skipif(
    not current_platform.has_device_capability(80),
    reason="Requires compute capability >= 8.0",
)
@torch.inference_mode()
def test_adjust_n_for_activation():
    """Test the adjust_N_for_activation method."""
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

    experts = TritonExperts(
        moe_config=make_dummy_moe_config(),
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
    )

    N = 256

    # Gated activations should return N // 2
    assert experts.adjust_N_for_activation(N, MoEActivation.SILU) == N // 2
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU) == N // 2

    # Non-gated activations should return N
    assert experts.adjust_N_for_activation(N, MoEActivation.SILU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.GELU_NO_MUL) == N
    assert experts.adjust_N_for_activation(N, MoEActivation.RELU2_NO_MUL) == N


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 requires compute capability >= 8.9",
)
@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SILU_NO_MUL, MoEActivation.GELU_NO_MUL, MoEActivation.RELU2_NO_MUL],
)
def test_triton_experts_supports_fp8_no_mul(activation):
    """Test that TritonExperts reports support for FP8 non-gated configs."""
    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size_per_partition=256,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
        is_act_and_mul=False,
    )
    supported, reason = FusedMoEExperts.is_supported_config(
        TritonExperts,
        moe_config,
        weight_key=kFp8StaticChannelSym,
        activation_key=kFp8DynamicTokenSym,
        activation_format=FusedMoEActivationFormat.Standard,
    )
    assert supported, (
        f"TritonExperts should support FP8 non-gated {activation}: {reason}"
    )


TORCH_ACTIVATION_FN = {
    MoEActivation.SILU_NO_MUL: F.silu,
    MoEActivation.GELU_NO_MUL: F.gelu,
    MoEActivation.RELU2_NO_MUL: lambda x: F.relu(x).square(),
}


def native_w8a8_per_token_matmul(A, B, As, Bs, output_dtype):
    """Per-token input scale, per-channel weight scale matmul."""
    A = A.to(torch.float32)
    B = B.to(torch.float32).t()
    C = torch.matmul(A, B)
    C = As * C * Bs.view(1, -1)
    return C.to(output_dtype)


def fp8_mask(a, mask):
    dtype = a.dtype
    return a.view(torch.int8)[mask].view(dtype)


def torch_w8a8_per_column_moe_no_gate(
    a,
    w1,
    w2,
    w1_s,
    w2_s,
    topk_weights,
    topk_ids,
    activation_fn,
):
    """Reference non-gated FP8 MoE using native torch."""
    B, D = a.shape
    topk = topk_ids.shape[1]

    a_q, a_s = ops.scaled_fp8_quant(a, use_per_token_if_dynamic=True)
    a_q = a_q.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    a_s = a_s.view(B, -1, 1).repeat(1, topk, 1).reshape(-1, 1)

    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    flat_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = flat_ids == i
        if mask.sum():
            inter_out = native_w8a8_per_token_matmul(
                fp8_mask(a_q, mask),
                w1[i],
                fp8_mask(a_s, mask),
                w1_s[i],
                output_dtype=a.dtype,
            )
            act_out = activation_fn(inter_out)
            act_out_q, act_out_s = ops.scaled_fp8_quant(
                act_out,
                use_per_token_if_dynamic=True,
            )
            out[mask] = native_w8a8_per_token_matmul(
                act_out_q,
                w2[i],
                act_out_s,
                w2_s[i],
                output_dtype=a.dtype,
            )

    return (
        out.view(B, -1, w2.shape[1]) * topk_weights.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


@pytest.mark.skipif(
    not current_platform.has_device_capability(89),
    reason="FP8 requires compute capability >= 8.9",
)
@pytest.mark.parametrize("m", [1, 33])
@pytest.mark.parametrize("n", [128, 256])
@pytest.mark.parametrize("k", [128, 256])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize(
    "activation",
    [MoEActivation.SILU_NO_MUL, MoEActivation.GELU_NO_MUL, MoEActivation.RELU2_NO_MUL],
)
@torch.inference_mode()
def test_triton_experts_no_mul_fp8(m, n, k, topk, activation):
    dtype = torch.bfloat16
    e = NUM_EXPERTS
    finfo = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = finfo.max, finfo.min
    factor_for_scale = 1e-2

    a = torch.randn((m, k), dtype=dtype, device="cuda") / 10

    w1_fp32 = (torch.rand((e, n, k), dtype=torch.float32, device="cuda") - 0.5) * 0.1
    w1 = (w1_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w2_fp32 = (torch.rand((e, k, n), dtype=torch.float32, device="cuda") - 0.5) * 0.1
    w2 = (w2_fp32 * fp8_max).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    w1_s = torch.rand(e, n, device="cuda") * factor_for_scale
    w2_s = torch.rand(e, k, device="cuda") * factor_for_scale

    score = torch.randn((m, e), dtype=dtype, device="cuda")
    topk_weights = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(topk_weights, topk)

    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_module

    _real_kernel = fused_moe_module.invoke_fused_moe_triton_kernel
    kernel_call_count = []

    def spy_kernel(*args, **kwargs):
        kernel_call_count.append(1)
        return _real_kernel(*args, **kwargs)

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        ref_out = torch_w8a8_per_column_moe_no_gate(
            a,
            w1,
            w2,
            w1_s,
            w2_s,
            topk_weights,
            topk_ids,
            activation_fn=TORCH_ACTIVATION_FN[activation],
        )

        # use fused_experts and not TritonExperts since TritonExperts do not perform
        # the support check. use spy to make sure the triton kernel is actually called.
        with patch.object(
            fused_moe_module, "invoke_fused_moe_triton_kernel", spy_kernel
        ):
            out = fused_experts(
                a,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation=activation,
                quant_config=fp8_w8a8_moe_quant_config(
                    per_act_token_quant=True,
                    w1_scale=w1_s,
                    w2_scale=w2_s,
                    block_shape=None,
                ),
            )

    assert len(kernel_call_count) == 2, (
        f"Expected invoke_fused_moe_triton_kernel to be called twice "
        f"(w1 and w2 matmuls), got {len(kernel_call_count)}"
    )
    torch.testing.assert_close(out, ref_out, atol=0.1, rtol=0.05)
