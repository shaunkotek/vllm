# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

logger = init_logger(__name__)


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.moe_kernel: mk.FusedMoEKernel | None = None

    @property
    def supports_internal_mk(self) -> bool:
        # NOTE(rob): temporary attribute to indicate support for
        # completed migration to the new internal MK interface.
        return self.moe_kernel is not None

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        # NOTE(rob): temporary attribute to indicate support for
        # completed migration to the new internal MK interface.
        return (
            self.moe_kernel is not None and self.moe_kernel.can_overlap_shared_experts
        )

    @property
    def supports_staged_execution(self) -> bool:
        return self.moe_kernel is not None and self.moe_kernel.supports_staged_execution

    @property
    def supports_async_staged_execution(self) -> bool:
        return (
            self.moe_kernel is not None
            and self.moe_kernel.supports_async_staged_execution
        )

    @abstractmethod
    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        """
        Given layer hidden size and intermediate size per partition and MoE
        configurations, round up hidden_size and intermediate_size_per_partition
        if necessary.

        Args:
            hidden_size: Layer hidden-size
            intermediate_size_per_partition: Intermediate size per partition for
                the layer.
            act_dtype: Data type of the layer activations.
            moe_parallel_config: Fused MoE parallelization strategy configuration.

        Return:
            A tuple of (rounded_hidden_size, rounded_intermediate_size_per_partition),
            where:
                - rounded_hidden_size is the possibly rounded up hidden size.
                - rounded_intermediate_size_per_partition is the possibly rounded
                  up intermediate size per partition.
        """
        from .all2all_utils import maybe_roundup_layer_hidden_size

        return maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype, moe_parallel_config
        ), intermediate_size_per_partition

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self.moe_kernel is not None:
            return self.moe_kernel.prepare_finalize.topk_indices_dtype()
        return None

    @property
    def skip_forward_padding(self) -> bool:
        """Whether to skip the padding in the forward before applying the moe method."""
        return False

    @property
    def has_unpadded_output(self) -> bool:
        """
        Indicates that the hidden_states output might be the unpadded
        hidden_states shape rather than the full padded shape.
        """
        return False

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    @property
    def is_monolithic(self) -> bool:
        if self.moe_kernel is None:
            if hasattr(self, "experts_cls"):
                return self.experts_cls.is_monolithic()
            else:
                return False
        return self.moe_kernel.is_monolithic

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Apply the MoE operation using modular kernels.

        Args:
            layer: RoutedExperts instance containing weight parameters
            x: Input tensor
            topk_weights: Expert weights from router
            topk_ids: Selected expert IDs from router
            shared_experts_input: Input for shared experts (if any)

        Returns:
            Output tensor from routed experts
        """
        raise NotImplementedError

    def apply_monolithic(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply the MoE operation using monolithic kernels.

        Args:
            layer: RoutedExperts instance containing weight parameters
            x: Input tensor
            router_logits: Router logits (routing done internally)

        Returns:
            Output tensor from routed experts
        """
        raise NotImplementedError

    def begin_staged(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> mk.FusedMoEDispatchHandle:
        """Launch dispatch for a staged modular MoE invocation.

        This API is intended for model adapters with independent computation
        that can hide dispatch latency. Callers without an overlap opportunity
        should use the regular atomic MoE forward path.
        """
        assert self.moe_kernel is not None
        return self.moe_kernel.begin_staged(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        )

    def run_staged_experts(
        self,
        handle: mk.FusedMoEDispatchHandle,
        shared_experts: "SharedExperts | None" = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> mk.FusedMoECombineHandle:
        """Complete dispatch, execute experts, and launch combine.

        Call this at the model-selected expert boundary, after computation that
        overlaps dispatch and before computation that overlaps combine.
        """
        assert self.moe_kernel is not None
        return self.moe_kernel.run_staged_experts(
            handle,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def finish_staged(
        self,
        handle: mk.FusedMoECombineHandle,
    ) -> torch.Tensor:
        """Wait for staged combine before consuming the routed output."""
        assert self.moe_kernel is not None
        return self.moe_kernel.finish_staged(handle)
