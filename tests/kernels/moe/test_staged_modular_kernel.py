# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernelModularImpl,
)


class _AsyncPrepareFinalize:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def supports_async(self) -> bool:
        return True

    def prepare_async(self, hidden_states, *args, **kwargs):
        self.events.append("dispatch_launch")

        def hook():
            self.events.append("dispatch_hook")

        def receive():
            self.events.append("dispatch_receive")
            return hidden_states + 1, None, None, None, None

        return hook, receive

    def finalize_async(
        self,
        output,
        fused_out,
        *args,
        **kwargs,
    ):
        self.events.append("combine_launch")

        def hook():
            self.events.append("combine_hook")

        def receive():
            self.events.append("combine_receive")
            output.copy_(fused_out)

        return hook, receive


class _SyncPrepareFinalize:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def supports_async(self) -> bool:
        return False

    def prepare(self, hidden_states, *args, **kwargs):
        self.events.append("dispatch")
        return hidden_states + 1, None, None, None, None

    def finalize(self, output, fused_out, *args, **kwargs):
        self.events.append("combine")
        output.copy_(fused_out)


class _Experts:
    def __init__(self) -> None:
        self.moe_config = SimpleNamespace(moe_parallel_config=None)
        self.quant_config = None
        self.expects_unquantized_inputs = False

    def finalize_weight_and_reduce_impl(self):
        return None


def _make_impl(prepare_finalize, events):
    impl = FusedMoEKernelModularImpl(prepare_finalize, _Experts())

    def fused_experts(**kwargs):
        events.append("experts")
        events.append(f"persistent={kwargs['persistent_output']}")
        return kwargs["a1q"] * 2

    impl._fused_experts = fused_experts
    return impl


def _inputs():
    hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    weights = torch.empty(2, 4, 4)
    topk_ids = torch.zeros(2, 1, dtype=torch.int64)
    topk_weights = torch.ones(2, 1)
    return hidden_states, weights, topk_ids, topk_weights


def test_staged_execution_defers_dispatch_and_combine_receivers():
    events: list[str] = []
    impl = _make_impl(_AsyncPrepareFinalize(events), events)
    hidden_states, weights, topk_ids, topk_weights = _inputs()

    dispatch = impl.begin_staged(
        hidden_states,
        weights,
        weights,
        topk_ids,
        topk_weights,
        activation=MoEActivation.SILU,
    )
    assert events == ["dispatch_launch"]

    events.append("independent_before_experts")
    combine = impl.run_staged_experts(dispatch)
    assert events == [
        "dispatch_launch",
        "independent_before_experts",
        "dispatch_hook",
        "dispatch_receive",
        "experts",
        "persistent=True",
        "combine_launch",
    ]

    events.append("independent_before_finish")
    output = impl.finish_staged(combine)
    assert events[-3:] == [
        "independent_before_finish",
        "combine_hook",
        "combine_receive",
    ]
    torch.testing.assert_close(output, (hidden_states + 1) * 2)


def test_atomic_apply_reuses_workspace_output():
    events: list[str] = []
    impl = _make_impl(_AsyncPrepareFinalize(events), events)
    hidden_states, weights, topk_ids, topk_weights = _inputs()

    output = impl.apply(
        hidden_states,
        weights,
        weights,
        topk_ids,
        topk_weights,
        activation=MoEActivation.SILU,
    )

    assert "persistent=False" in events
    torch.testing.assert_close(output, (hidden_states + 1) * 2)


def test_staged_execution_falls_back_to_synchronous_prepare_finalize():
    events: list[str] = []
    impl = _make_impl(_SyncPrepareFinalize(events), events)
    hidden_states, weights, topk_ids, topk_weights = _inputs()

    dispatch = impl.begin_staged(
        hidden_states,
        weights,
        weights,
        topk_ids,
        topk_weights,
        activation=MoEActivation.SILU,
    )
    assert events == ["dispatch"]

    combine = impl.run_staged_experts(dispatch)
    assert events == ["dispatch", "experts", "persistent=False", "combine"]

    output = impl.finish_staged(combine)
    assert events == ["dispatch", "experts", "persistent=False", "combine"]
    torch.testing.assert_close(output, (hidden_states + 1) * 2)
