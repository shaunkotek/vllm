# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import final

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
    dbo_register_recv_hook,
    dbo_yield,
)
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

#
# This file defines a set of base classes used to make MoE kernels more modular.
# The goal is to be able to utilize different communication mechanisms with
# any fused MoE kernel without needing to have combinatoric implementations.
#
# The fused moe kernels are broken down into the following components:
#
# [Router] → [Quantize-Dispatch] → [Permute-Experts-Unpermute] → [Combine]
#
# Each component will be independent of (but may inform) the others except for
# [Quantize-Dispatch] and `[Combine] (see below). The components can then be
# mixed and matched with so that DP+EP can be supported easily for multiple
# MoE kernel implementations.
#
# The following main classes are defined:
# * FusedMoEPrepareAndFinalizeModular - an abstract base class for preparation of MoE
#   inputs (e.g. quantization, distribution) and finalization of Moe outputs.
#   The prepare method must take care of any needed quantization and the
#   finalize method, informed by the FusedMoEExpertsModular method,
#   may apply weights and/or do the final reduction of the output.
# * FusedMoEExpertsModular - an abstract base class for the main fused
#   MoE operation, i.e matmul + act_mul + optionally quant + matmul.
#   Some FusedMoEExpertsModular implementations may choose to do
#   the weight application and/or reduction. The class communicates this
#   to [Finalize] via a TopKWeightAndReduce object.
# * FusedMoEModularKernel - an interface class that combines a
#   FusedMoEPrepareAndFinalizeModular and a FusedMoEExpertsModular to
#   provide the standard fused MoE kernel interface.
# * TopKWeightAndReduce - A TopKWeightAndReduce implementation chosen
#   by the FusedMoEExpertsModular implementation that is passed
#   on to [Finalize].
#
# [Quantize-Prepare] and [Finalize] functionality are bundled into a single
# class `FusedMoEPrepareAndFinalizeModular` since they could use collective
# communication mechanisms that need to be consistent.
#


class FusedMoEActivationFormat(Enum):
    """
    The standard activation format (num_tokens, hidden dim).
    """

    Standard = ("standard",)
    """
    The batched experts format (num experts, max tokens per expert, hidden dim)
    """
    BatchedExperts = ("batched_experts",)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(
            expert_num_tokens_list, device="cpu", dtype=torch.int32
        )
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device, non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu,
        )


class TopKWeightAndReduce(ABC):
    """
    An abstract base class for weight application and reduction implementations.
    """

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """
        Apply topk_weights to the fused_experts_outputs and/or reduce.
        If an output tensor is not passed, it will be created in the
        function.
        """
        raise NotImplementedError


#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - Optional ExpertTokensMetadata containing gpu/cpu tensors
#   as big as the number of local experts with the information about the
#   number of tokens assigned to each local expert.
# - Optional dispatched expert topk IDs
# - Optional dispatched expert topk weight
#
# See `prepare` method below.
#
PrepareResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    ExpertTokensMetadata | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

#
# PrepareResultType is a tuple of:
# - quantized + dispatched a.
# - quantized + dispatched a1_scales.
# - dispatched router logits.
#
# See `prepare_monolithic` method below.
#
PrepareMonolithicResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]

ReceiverType = Callable[[], PrepareResultType]


@dataclass(frozen=True)
class FusedMoEDispatchHandle:
    """State produced after launching the dispatch stage of a modular MoE."""

    hidden_states: torch.Tensor
    output: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    activation: MoEActivation
    global_num_experts: int
    local_num_experts: int
    expert_map: torch.Tensor | None
    apply_router_weight_on_input: bool
    prepare_result: PrepareResultType | None
    completion_hook: Callable | None
    receiver: ReceiverType | None


@dataclass(frozen=True)
class FusedMoECombineHandle:
    """State produced after experts run and combine has been launched."""

    output: torch.Tensor
    fused_out: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    completion_hook: Callable | None
    receiver: Callable | None


################################################################################
# Prepare/Finalize
################################################################################


class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.

    There are two variants of this class:
    * FusedMoEPrepareAndFinalizeModular - this operates on topk ids and weights
    * FusedMoEPrepareAndFinalizeMonolithic - the operates on router_logits
    """

    def post_init_setup(self, fused_experts: "FusedMoEExperts"):
        """
        Initialize FusedMoEPrepareAndFinalizeModular settings that depend on
        FusedMoEExpertsModular experts object.
        The FusedMoEPrepareAndFinalizeModular implementations that have such
        dependencies may choose to override this function.
        """
        return

    @property
    @abstractmethod
    def activation_format(self) -> FusedMoEActivationFormat:
        """
        A property indicating the output format of the activations for the
        'prepare' method.
        """
        raise NotImplementedError

    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None:
        """
        The PrepareFinalize All2All implementations generally constrain the
        dtype of the topk_ids they support. This function returns the
        required topk indices dtype so it can be respected.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
        """
        Some PrepareFinalize All2All implementations are batched. Meaning,
        they can process only as set of tokens at a time. This
        function returns the batch size i.e the maximum number of tokens
        the implementation can process at a time.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of finalize is reduced across all
        ranks.
        """
        raise NotImplementedError

    def supports_async(self) -> bool:
        """
        Indicates whether or not this class implements prepare_async and
        finalize_async.
        """
        return False

    def on_commit(self) -> None:
        """
        Runs after this prepare/finalize has been committed to the active
        MoE kernel.
        """
        return


# TODO: pass FusedMoEParallelConfig in as ctor parameter?
class FusedMoEPrepareAndFinalizeModular(FusedMoEPrepareAndFinalize):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above for the Modular case.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> PrepareResultType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - quant_config: Quantization info provided by the fused experts.
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEExpertsModular
          in cases where the compute kernel expects unquantized inputs

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> tuple[Callable, ReceiverType] | ReceiverType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel
        but do not wait for results from other workers.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - defer_input_quant: Runtime parameter indicating whether or not to
          defer input quantization to the FusedMoEExpertsModular
          in cases where the compute kernel expects unquantized inputs

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `prepare`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        e.g.

        ret = obj.prepare_async(...)

        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()

        if hook is not None:
        a, a_scales, expert_meta, topk_ids, topk_weights = receiver()

        is equivalent to:

        a, a_scales, expert_meta, topk_ids, topk_weights = obj.prepare(...)
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.
        """
        raise NotImplementedError

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> tuple[Callable, Callable] | Callable:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output but do not wait for results from other workers.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `finalize`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        ret = obj.finalize_async(output, ...)
        ... output not valid yet ...
        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()
        receiver()
        ... output valid here ...

        is equivalent to:

        obj.finalize(output, ...)
        """
        raise NotImplementedError


class FusedMoEPrepareAndFinalizeMonolithic(FusedMoEPrepareAndFinalize):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above for the monolithic case.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType:
        """
        Optional method for subclasses compatible with monolithic
        FusedMoEExpertsModular kernels.

        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - quant_config: Quantization info provided by the fused experts.
        - defer_input_quant: Runtime parameter indicating whether or not to
            defer input quantization to the FusedMoEExpertsModular

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(self, fused_expert_output: torch.Tensor) -> torch.Tensor:
        """
        Optional method for subclasses compatible with monolithic
        FusedMoEExpertsModular kernels.

        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        """
        raise NotImplementedError


################################################################################
# Experts
################################################################################


# TODO: add supported activations method (return string)
class FusedMoEExperts(ABC):
    # ROCm AITER kernels consume a 0/1 local-expert mask (with a trailing
    # sentinel slot); every other backend consumes the canonical -1/local-slot
    # expert_map. RoutedExperts.expert_map reads this flag to pick which to hand
    # the active experts kernel.
    consumes_expert_mask: bool = False

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        """
        moe_config: MoE layer configuration.
        quant_config: Quantization parameters for this experts instance.
        """
        if self.activation_format() == FusedMoEActivationFormat.Standard and (
            max_num_tokens is not None or num_dispatchers is not None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers should only be set for "
                "BatchedExperts activation format."
            )
        elif self.activation_format() == FusedMoEActivationFormat.BatchedExperts and (
            max_num_tokens is None or num_dispatchers is None
        ):
            raise ValueError(
                "max_num_tokens and num_dispatchers must be set for "
                "BatchedExperts activation format."
            )

        self.moe_config = moe_config
        self.quant_config = quant_config
        self.activation_config = ApplyMoEActivationConfig.from_configs(
            moe_config, quant_config
        )
        self.max_num_tokens = max_num_tokens
        self.num_dispatchers = num_dispatchers

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:  # noqa: B027
        pass

    @staticmethod
    def is_monolithic() -> bool:
        raise NotImplementedError("Implemented by subclasses.")

    @property
    def expects_unquantized_inputs(self) -> bool:
        """
        Whether or not the PrepareFinalize should defer input quantization
        in the prepare step. If True, then the Experts kernel will
        execute the input quantization itself.

        Sample subclasses that override are AITER and FlashInfer CUTLASS.
        """
        return False

    @staticmethod
    @abstractmethod
    def activation_format() -> FusedMoEActivationFormat:
        """
        A property which is a tuple of the input and output activation formats
        for the 'apply' method.
        """
        raise NotImplementedError

    #
    # Various helpers for registering support for various features.
    # Used by the oracle to select a particular kernel for a deployment.
    #

    @staticmethod
    def is_supported_config(
        cls: type["FusedMoEExperts"],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        def _make_reason(reason: str) -> str:
            return f"kernel does not support {reason}"

        if not cls._supports_current_device():
            return False, _make_reason(f"current device {current_platform.device_name}")
        elif not (moe_config.is_act_and_mul or cls._supports_no_act_and_mul()):
            return False, _make_reason("no act_and_mul MLP layer")
        elif not cls._supports_activation(moe_config.activation):
            return False, _make_reason(f"{moe_config.activation} activation")
        elif not cls._supports_quant_scheme(weight_key, activation_key):
            return False, _make_reason(
                f"quantization scheme {weight_key}x{activation_key}"
            )
        elif not cls._supports_parallel_config(moe_config.moe_parallel_config):
            return False, _make_reason(
                f"parallel config {moe_config.moe_parallel_config}"
            )
        elif not cls._supports_routing_method(
            moe_config.routing_method, weight_key, activation_key
        ):
            return False, _make_reason(f"routing method {moe_config.routing_method}")
        elif not cls._supports_router_logits_dtype(
            moe_config.router_logits_dtype,
            moe_config.routing_method,
        ):
            return False, _make_reason(
                f"router logits dtype {moe_config.router_logits_dtype}"
            )
        elif not cls._supports_shape(moe_config.hidden_dim):
            return False, _make_reason(
                f"{moe_config.hidden_dim} hidden dim is not supported"
            )
        elif activation_format != cls.activation_format():
            return False, _make_reason(f"{activation_format.value} activation format")
        elif envs.VLLM_BATCH_INVARIANT and not cls._supports_batch_invariance():
            return False, _make_reason("batch invariance")
        elif moe_config.is_lora_enabled and not cls.supports_lora():
            return False, _make_reason("LoRA")
        return True, None

    @staticmethod
    @abstractmethod
    def _supports_current_device() -> bool:
        """
        Whether the kernel supports the current device type
        (compute cability and current platform).
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_no_act_and_mul() -> bool:
        """
        Whether the kernel supports act_and_mul=False, i.e.
        non-gated MoE models like Nemotron-Nano.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        """
        Whether the kernel supports a particular act function.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        """
        Whether the kernel supports deployment in particular parallel config.

        Can be overridden if a kernel does not support EP, SP or some other
        configuration.
        """
        raise NotImplementedError

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """
        Whether the kernel supports a routing method (e.g. GroupedTopK).

        Can be overridden by monolithic kernels that execute the router
        in addition to the experts if certain routers are not supported.
        """
        return True

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        """
        Whether a kernel supports a particular dtype for router logits input.

        Can be overridden by monolithic kernels that execute the router
        in addition to the experts if certain dtypes are not supported.
        """
        return True

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        """
        Whether a kernel supports a particular shape. Can be overridden if a kernel
        has specific shape requirements.
        """
        return True

    @staticmethod
    def _supports_batch_invariance() -> bool:
        """
        Whether the kernel supports batch invariance, i.e. the output does not
        depend on the order of the tokens in the input batch. This is useful
        for determining if the kernel can used with VLLM_BATCH_INVARIANT=1.
        """
        return False

    #
    # Various helpers for accessing quantization parameters from the
    # quant_config.
    #

    @property
    def quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.quant_dtype

    @property
    def weight_quant_dtype(self) -> torch.dtype | str | None:
        return self.quant_config.weight_quant_dtype

    @property
    def block_shape(self) -> list[int] | None:
        return self.quant_config.block_shape

    @property
    def per_act_token_quant(self) -> bool:
        return self.quant_config.per_act_token_quant

    @property
    def per_out_ch_quant(self) -> bool:
        return self.quant_config.per_out_ch_quant

    @property
    def a1_scale(self) -> torch.Tensor | None:
        return self.quant_config.a1_scale

    @property
    def a2_scale(self) -> torch.Tensor | None:
        return self.quant_config.a2_scale

    @property
    def a1_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a1_gscale

    @property
    def a2_gscale(self) -> torch.Tensor | None:
        return self.quant_config.a2_gscale

    @property
    def w1_scale(self) -> torch.Tensor | None:
        return self.quant_config.w1_scale

    @property
    def w2_scale(self) -> torch.Tensor | None:
        return self.quant_config.w2_scale

    @property
    def w1_zp(self) -> torch.Tensor | None:
        return self.quant_config.w1_zp

    @property
    def w2_zp(self) -> torch.Tensor | None:
        return self.quant_config.w2_zp

    @property
    def w1_bias(self) -> torch.Tensor | None:
        return self.quant_config.w1_bias

    @property
    def w2_bias(self) -> torch.Tensor | None:
        return self.quant_config.w2_bias

    @property
    def g1_alphas(self) -> torch.Tensor | None:
        return self.quant_config.g1_alphas

    @property
    def g2_alphas(self) -> torch.Tensor | None:
        return self.quant_config.g2_alphas

    @staticmethod
    def supports_lora() -> bool:
        """Return True if this expert impl natively handles LoRA.

        LoRA-aware experts should mix in LoRAExpertsMixin, which flips this
        to True and provides the per-forward LoRA state plumbing.
        """
        return False

    def supports_packed_ue8m0_act_scales(self) -> bool:
        """
        A flag indicating whether or not this class can process packed ue8m0
        activation scales.
        """
        return False


class FusedMoEExpertsModular(FusedMoEExperts):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above.
    """

    @staticmethod
    def is_monolithic() -> bool:
        return False

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        """
        Extract the MoE problem size from the given tensor arguments:
        - a: The hidden states, input to the MoE layer.
        - w1: The first set of expert weights.
        - w2: The second set of expert weights.
        - topk_ids: The topk ids.

        Note: extracting the problem shape from the weight and activation
        tensors is not obvious.  It needs to be done this way specifically
        due to subtle issues with particular kernels, e.g. the int4 kernels
        divide the trailing dimension by two, so it's not "correct" to
        extract N or K from the trailing dimension of w1 or w2.  Similarly,
        some kernels transpose the weights, so this needs to be kept in mind.

        Note: This implementation covers most cases. However, if experts
        require a specialized implementation, like MarlinExperts, they are free
        to override this function.
        """
        assert len(w1.shape) == 3 and len(w2.shape) == 3
        E, N, _ = w1.shape
        K = a1.size(-1)

        if a1.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{a1.size(0)} == {E}"
            M = a1.size(1)  # This is max_num_tokens

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        """
        Workspace type: The dtype to use for the workspace tensors.
        """
        return act_dtype

    @abstractmethod
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        Compute the shapes for the temporary and final outputs of the two gemms
        and activation in the fused expert function.  Since the gemms are
        independent, the workspace for the first gemm can be shared with the
        workspace for the last gemm.

        Inputs:
        - M: number of tokens.
        - N: Row (or column) dimension of expert weights.
        - K: hidden dimension
        - topk: The number of top-k experts to select.
        - global_num_experts: global number of experts.
        - local_num_experts: local number of experts due to DP/EP.
        - expert_tokens_meta: number of tokens per expert metadata for batched
                              format.

        Returns a tuple of:
        - workspace13 shape tuple: must be large enough to hold the
          result of either expert gemm.
        - workspace2 shape tuple: must be large enough to hold the
          result of the activation function.
        - output shape tuple: must be exact size of the final gemm output.
        - Note: workspace shapes can be 0 if the workspace is not needed.
          But in order for activation chunking to work, the first dimension
          of each tuple must be the number of tokens when the shape is
          not 0.
        """
        raise NotImplementedError

    @staticmethod
    def adjust_N_for_activation(N: int, activation: MoEActivation) -> int:
        """
        Calculate the output dimension for the activation function.

        For *_no_mul activations (e.g. relu2_no_mul),
        there's no gate/up split, so output size equals input size (N).

        For regular gated activations (e.g., silu, gelu, swigluoai),
        output size is N // 2 due to gate × activation(up) multiplication.

        Args:
            N: The intermediate size (width of w1/w3 weights).
            activation: The activation function enum.

        Returns:
            The output dimension after activation.
        """
        return N if not activation.is_gated else N // 2

    def activation(
        self,
        activation: MoEActivation,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        apply_moe_activation(
            activation,
            output,
            input,
            activation_config=self.activation_config,
            topk_ids=topk_ids,
            expert_map=expert_map,
        )

    @abstractmethod
    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        This function computes the intermediate result of a Mixture of Experts
        (MoE) layer using two sets of weights, w1 and w2.

        Parameters:
        - output: (torch.Tensor): The unweighted, unreduced output tensor.
        - hidden_states: (torch.Tensor): The (quantized) input tensor to the MoE
          layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights: A map of row to expert weights. Some implementations
          choose to do weight application.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - a1q_scale (Optional[torch.Tensor]): Optional quantized scale to be
          used for a1.  Result of quantization from prepare/finalize and not
          from the FusedMoEQuantConfig.
        - workspace13 (torch.Tensor): A scratch tensor used for gemm outputs
          must be large enough to hold output of either MoE gemm.
        - workspace2 (torch.Tensor): A scratch tensor used for the activation
          function.
        - expert_tokens_meta (Optional[ExpertTokensMetadata]) - An optional
          ExpertTokensMetadata object containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - apply_router_weight_on_input: True if router weights are already
          applied on the input. This is relevant if the implementation
          chooses to do weight application.
        """
        raise NotImplementedError


class FusedMoEExpertsMonolithic(FusedMoEExperts):
    """
    An abstract base class for the [Permute-Experts-Unpermute] step described
        above, but with the monolithic interface (accepts router logits
        rather than topk ids and weights).
    """

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        """
        Whether the kernel supports a routing method (e.g. GroupedTopK).

        Monolithic kernels should explicitly opt-in to support.
        """
        raise NotImplementedError

    @staticmethod
    def _supports_router_logits_dtype(
        router_logits_dtype: torch.dtype | None,
        routing_method: RoutingMethodType,
    ) -> bool:
        """
        Whether the kernel supports a dtype for router logits.

        Modular kernels should opt-in to support.
        """
        raise NotImplementedError

    @staticmethod
    def is_monolithic() -> bool:
        return True

    routing_replay_capture_fn: Callable[[torch.Tensor], None] | None = None
    _routing_replay_buffer: torch.Tensor | None = None

    def supports_routing_replay_capture(self) -> bool:
        """Whether this expert supports routing replay capture.

        Subclasses backed by a kernel that exposes routed expert IDs
        (e.g. FlashInfer's ``routing_replay_out``) should override.
        """
        return False

    def set_capture_fn(
        self,
        capture_fn: Callable[[torch.Tensor], None] | None,
    ) -> None:
        self.routing_replay_capture_fn = capture_fn
        if capture_fn is None:
            self._routing_replay_buffer = None
            return
        self._routing_replay_buffer = torch.empty(
            (self.moe_config.max_num_tokens, self.moe_config.experts_per_token),
            dtype=torch.int16,
            device=self.moe_config.device,
        )

    def _maybe_make_routing_replay_buffer(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.routing_replay_capture_fn is None:
            return None
        buf = self._routing_replay_buffer
        assert buf is not None
        if buf.shape[0] < num_tokens or buf.device != device:
            raise ValueError(
                "Routing replay buffer was initialized for "
                f"{buf.shape[0]} tokens on {buf.device}, but the kernel "
                f"received {num_tokens} tokens on {device}."
            )
        return buf

    def _maybe_dispatch_routing_replay(
        self,
        routing_replay_out: torch.Tensor | None,
        num_tokens: int,
    ) -> None:
        if routing_replay_out is None or self.routing_replay_capture_fn is None:
            return
        self.routing_replay_capture_fn(routing_replay_out[:num_tokens])

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """
        Same as apply(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is useful for kernels
        with fused router and fused_experts (e.g. FLASHINFER_TRTLLM).
        """
        raise NotImplementedError


################################################################################
# Kernel
################################################################################


@final
class FusedMoEKernelModularImpl:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        fused_experts: FusedMoEExpertsModular,
    ):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts
        self.shared_experts: SharedExperts | None = None
        moe_parallel_config = fused_experts.moe_config.moe_parallel_config
        self.moe_parallel_config = moe_parallel_config
        self.is_dp_ep = (
            moe_parallel_config is not None
            and moe_parallel_config.dp_size > 1
            and moe_parallel_config.use_ep
        )

    def _allocate_buffers(
        self,
        out_dtype: torch.dtype,
        device: torch.device,
        M_chunk: int,
        M_full: int,
        N: int,
        K: int,
        top_k: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: ExpertTokensMetadata | None,
        activation: MoEActivation,
        persistent_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Allocate temporary and output buffers for the fused experts op.
        Inputs:
        - out_dtype: output type of workspace and output tensors.
        - device: the device of the workspace and output tensors.
        See `workspace_shapes` for a description of the remainder of arguments.
        Returns a tuple of (workspace13, workspace2, output) tensors.
        """
        assert M_full > 0 and M_chunk > 0

        workspace_dtype = self.fused_experts.workspace_dtype(out_dtype)

        # Get intermediate workspace shapes based off the chunked M size.
        workspace13_shape, workspace2_shape, _ = self.fused_experts.workspace_shapes(
            M_chunk,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # Get final output shape based on the full M size.
        _, _, fused_out_shape = self.fused_experts.workspace_shapes(
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        if persistent_output:
            workspace13, workspace2 = current_workspace_manager().get_simultaneous(
                (workspace13_shape, workspace_dtype),
                (workspace2_shape, workspace_dtype),
            )
            fused_out = torch.empty(
                fused_out_shape,
                dtype=workspace_dtype,
                device=device,
            )
        else:
            # We can reuse the memory between cache1 and cache3 because by the
            # time we need cache3, we're done with cache1.
            # Reuse workspace13 for the output since there is only one chunk.
            max_shape_size = max(prod(workspace13_shape), prod(fused_out_shape))
            common_workspace, workspace2 = current_workspace_manager().get_simultaneous(
                ((max_shape_size,), workspace_dtype),
                (workspace2_shape, workspace_dtype),
            )
            workspace13 = _resize_cache(common_workspace, workspace13_shape)
            fused_out = _resize_cache(common_workspace, fused_out_shape)

        return workspace13, workspace2, fused_out

    def _maybe_apply_shared_experts(
        self,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ):
        if shared_experts is not None:
            assert self.prepare_finalize.supports_async()
            assert shared_experts_input is not None
            shared_experts(
                shared_experts_input,
                SharedExpertsOrder.MK_INTERNAL_OVERLAPPED,
            )

    @staticmethod
    def _unpack_async_result(
        result: tuple[Callable, Callable] | Callable,
    ) -> tuple[Callable | None, Callable]:
        return result if isinstance(result, tuple) else (None, result)

    @staticmethod
    def _complete_async(
        completion_hook: Callable | None,
        receiver: Callable,
    ):
        if completion_hook is not None:
            if dbo_enabled():
                dbo_register_recv_hook(completion_hook)
                dbo_yield()
            else:
                completion_hook()
        return receiver()

    def _launch_prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> tuple[
        PrepareResultType | None,
        Callable | None,
        ReceiverType | None,
    ]:
        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()
            prepare_result = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.fused_experts.quant_config,
                defer_input_quant=self.fused_experts.expects_unquantized_inputs,
            )
            return prepare_result, None, None

        dbo_maybe_run_recv_hook()
        prepare_ret = self.prepare_finalize.prepare_async(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )
        hook, receiver = self._unpack_async_result(prepare_ret)
        return None, hook, receiver

    def _complete_prepare(
        self,
        prepare_result: PrepareResultType | None,
        completion_hook: Callable | None,
        receiver: ReceiverType | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        if prepare_result is None:
            assert receiver is not None
            prepare_result = self._complete_async(completion_hook, receiver)

        (
            a1q,
            a1q_scale,
            expert_tokens_meta,
            expert_topk_ids,
            expert_topk_weights,
        ) = prepare_result

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if expert_topk_ids is None else expert_topk_ids
        topk_weights = (
            topk_weights if expert_topk_weights is None else expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run dispatch preparation to completion.

        This blocking helper preserves the original non-staged prepare
        contract. It wraps both synchronous prepare/finalize backends and
        asynchronous backends, including their DBO completion hooks, and
        returns only after dispatched expert inputs are available.

        Args:
            hidden_states: Local token activations before expert dispatch.
            topk_weights: Router weights for each selected expert.
            topk_ids: Selected global expert IDs.
            global_num_experts: Number of experts in the global expert space.
            expert_map: Optional global-to-local expert mapping.
            apply_router_weight_on_input: Whether routing weights are applied
                before expert computation.

        Returns:
            Dispatched activations, optional activation scales, optional expert
            token metadata, and the expert-local top-k IDs and weights.
        """
        prepare_result, hook, receiver = self._launch_prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )
        return self._complete_prepare(
            prepare_result,
            hook,
            receiver,
            topk_weights,
            topk_ids,
        )

    def _fused_experts(
        self,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: ExpertTokensMetadata | None,
        output_alias: torch.Tensor | None = None,
        persistent_output: bool = False,
    ) -> torch.Tensor:
        _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids
        )

        # This happens when none of the tokens from the all2all reach this
        # EP rank. Also, note that this is only relevant for CUDAGraph
        # incompatible all2all kernels like the DeepEP high-throughput
        # kernels. CUDAGraph compatible all2all kernels like the DeepEP
        # low-latency kernels are always batched and can never run into
        # the tensor.numel() == 0 case.
        if M_full == 0:
            return torch.empty_like(a1q, dtype=in_dtype)

        workspace13, workspace2, fused_out = self._allocate_buffers(
            in_dtype,
            a1q.device,
            M_full,
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
            persistent_output=persistent_output,
        )

        use_output_alias = (
            output_alias is not None
            and output_alias.shape == fused_out.shape
            and output_alias.dtype == fused_out.dtype
            and output_alias.device == fused_out.device
            and output_alias.is_contiguous()
        )

        # If caller's output buffer already matches fused_out shape/dtype, alias
        # to skip the redundant copy in TopKWeightAndReduceNoOP.apply downstream.
        # This eliminates ~94% of __amd_rocclr_copyBuffer events (Copy 2 of the
        # double-copy MoE write-back path).
        if current_platform.is_rocm():
            from vllm._aiter_ops import rocm_aiter_ops

            if use_output_alias and rocm_aiter_ops.is_fused_moe_enabled():
                fused_out = output_alias
        elif use_output_alias:
            fused_out = output_alias

        self.fused_experts.apply(
            output=fused_out,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=self.fused_experts.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        return fused_out

    def _launch_finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> tuple[Callable | None, Callable | None]:
        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()
            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            return None, None

        finalize_ret = self.prepare_finalize.finalize_async(
            output,
            fused_out,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            self.fused_experts.finalize_weight_and_reduce_impl(),
        )
        self._maybe_apply_shared_experts(shared_experts, shared_experts_input)
        return self._unpack_async_result(finalize_ret)

    def _complete_finalize(
        self,
        completion_hook: Callable | None,
        receiver: Callable | None,
    ) -> None:
        if receiver is not None:
            self._complete_async(completion_hook, receiver)

    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run output combine to completion.

        This blocking helper preserves the original non-staged finalize
        contract. It starts the backend's combine operation, handles DBO
        completion hooks, and waits until ``output`` is ready. The optional
        shared-expert arguments preserve the existing modular-kernel overlap
        used by atomic MoE execution; shortcut-connected model adapters instead
        compute their later shared-expert input outside this method.

        Args:
            output: Destination tensor for the combined routed-expert result.
            fused_out: Local routed-expert output before backend finalization.
            hidden_states: Original routed input retained for API compatibility.
            topk_weights: Top-k weights corresponding to ``fused_out``.
            topk_ids: Top-k expert IDs corresponding to ``fused_out``.
            apply_router_weight_on_input: Whether routing weights were applied
                before expert computation.
            shared_experts: Optional shared-expert runner for legacy internal
                overlap with asynchronous combine.
            shared_experts_input: Input for ``shared_experts`` when that overlap
                is requested.

        Returns:
            ``output`` after combine has completed.
        """
        del hidden_states
        hook, receiver = self._launch_finalize(
            output,
            fused_out,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            shared_experts,
            shared_experts_input,
        )
        self._complete_finalize(hook, receiver)

        return output

    def begin_staged(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation = MoEActivation.SILU,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ) -> FusedMoEDispatchHandle:
        """Start a staged MoE invocation by launching expert dispatch.

        Use this method only when the caller has independent model computation
        to execute between dispatch and expert computation, as in a
        shortcut-connected MoE. Callers without such work should use
        :meth:`apply`, which executes the same lifecycle atomically. An
        asynchronous prepare/finalize backend can overlap dispatch with the
        caller's computation; a synchronous backend still follows this API but
        completes dispatch before returning.

        The returned handle owns the invocation state and must be consumed once
        by :meth:`run_staged_experts`.

        Args:
            hidden_states: Routed or shortcut token activations.
            w1: First expert projection weights.
            w2: Second expert projection weights.
            topk_ids: Selected global expert IDs.
            topk_weights: Router weights for each selected expert.
            activation: Activation between expert projections.
            global_num_experts: Number of experts in the global expert space.
            expert_map: Optional global-to-local expert mapping.
            apply_router_weight_on_input: Whether routing weights are applied
                before expert computation.

        Returns:
            A handle representing an in-flight or completed dispatch.
        """
        output = torch.empty_like(hidden_states)
        local_num_experts = w1.shape[0]
        if global_num_experts == -1:
            global_num_experts = local_num_experts

        prepare_result, hook, receiver = self._launch_prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )
        return FusedMoEDispatchHandle(
            hidden_states=hidden_states,
            output=output,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            local_num_experts=local_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            prepare_result=prepare_result,
            completion_hook=hook,
            receiver=receiver,
        )

    def run_staged_experts(
        self,
        handle: FusedMoEDispatchHandle,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
        persistent_output: bool = True,
    ) -> FusedMoECombineHandle:
        """Cross the staged expert boundary and launch output combine.

        Call this after the independent computation intended to hide dispatch
        latency. It waits for dispatch if necessary, executes the configured
        optimized expert kernel, then starts the prepare/finalize backend's
        combine operation. The returned handle must be consumed once by
        :meth:`finish_staged` after any computation intended to hide combine
        latency.

        ``shared_experts`` is available for the atomic modular-kernel path's
        existing internal overlap. Shortcut-connected model adapters normally
        leave it unset because their shared-expert input becomes available only
        later in the model path.

        Args:
            handle: Handle returned by :meth:`begin_staged`.
            shared_experts: Optional shared-expert runner to overlap internally.
            shared_experts_input: Input for ``shared_experts``.
            persistent_output: Allocate the expert result outside reusable MoE
                workspace when asynchronous combine may outlive this call.

        Returns:
            A handle representing an in-flight or completed combine.
        """
        (
            a1q,
            a1q_scale,
            expert_tokens_meta,
            topk_ids,
            topk_weights,
        ) = self._complete_prepare(
            handle.prepare_result,
            handle.completion_hook,
            handle.receiver,
            handle.topk_weights,
            handle.topk_ids,
        )

        lora_ctx = getattr(self.fused_experts, "_lora_context", None)
        if lora_ctx is not None:
            lora_ctx.original_hidden_states = handle.hidden_states
        try:
            fused_out = self._fused_experts(
                in_dtype=handle.hidden_states.dtype,
                a1q=a1q,
                a1q_scale=a1q_scale,
                w1=handle.w1,
                w2=handle.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=handle.activation,
                global_num_experts=handle.global_num_experts,
                local_num_experts=handle.local_num_experts,
                expert_map=handle.expert_map,
                apply_router_weight_on_input=(handle.apply_router_weight_on_input),
                expert_tokens_meta=expert_tokens_meta,
                output_alias=handle.output,
                persistent_output=(
                    persistent_output and self.prepare_finalize.supports_async()
                ),
            )
        finally:
            if lora_ctx is not None:
                lora_ctx.original_hidden_states = None

        hook, receiver = self._launch_finalize(
            handle.output,
            fused_out,
            topk_weights,
            topk_ids,
            handle.apply_router_weight_on_input,
            shared_experts,
            shared_experts_input,
        )
        return FusedMoECombineHandle(
            output=handle.output,
            fused_out=fused_out,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            completion_hook=hook,
            receiver=receiver,
        )

    def finish_staged(self, handle: FusedMoECombineHandle) -> torch.Tensor:
        """Wait for staged combine and return the routed-expert output.

        Call this after all independent computation intended to overlap the
        combine operation. This is the synchronization point that makes the
        routed-expert result safe to consume. Each handle returned by
        :meth:`run_staged_experts` must be finished exactly once.

        Args:
            handle: Handle returned by :meth:`run_staged_experts`.

        Returns:
            The combined routed-expert output.
        """
        self._complete_finalize(handle.completion_hook, handle.receiver)
        return handle.output

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        activation: MoEActivation = MoEActivation.SILU,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states: (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights (torch.Tensor): The topk weights applied at the end of the layer.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - activation (MoEActivation): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - apply_router_weight_on_input (bool): When true, the topk weights are
          applied directly on the inputs. This is only applicable when topk is
          1.
        - shared_experts: SharedExperts | None. The shared experts if any.
        - shared_experts_input (Optional[torch.Tensor]): Optional separate
          input for shared experts. For latent MoE, this is the original
          hidden_states before latent projection.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """
        dispatch_handle = self.begin_staged(
            hidden_states,
            w1,
            w2,
            topk_ids,
            topk_weights=topk_weights,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
        combine_handle = self.run_staged_experts(
            dispatch_handle,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            persistent_output=False,
        )
        return self.finish_staged(combine_handle)


@final
class FusedMoEKernelMonolithicImpl:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeMonolithic,
        fused_experts: FusedMoEExpertsMonolithic,
    ):
        self.prepare_finalize = prepare_finalize
        self.fused_experts = fused_experts

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        """
        Same as forward(), except uses router_logits as opposed
        to the topk_ids and topk_weights. This is used for kernels
        that have fused router + experts (e.g. FLASHINFER_TRTLLM).
        """

        a1q, a1q_scale, router_logits = self.prepare_finalize.prepare(
            hidden_states,
            router_logits=router_logits,
            quant_config=self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )

        fused_out = self.fused_experts.apply(
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            a1q_scale=a1q_scale,
            # grouped topk + fused topk bias parameters
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )

        output = self.prepare_finalize.finalize(fused_out)

        return output


@final
class FusedMoEKernel:
    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        fused_experts: FusedMoEExperts,
    ):
        super().__init__()

        # Initialize the implementation (monolithic or modular).
        self.impl: FusedMoEKernelModularImpl | FusedMoEKernelMonolithicImpl
        if isinstance(
            prepare_finalize, FusedMoEPrepareAndFinalizeModular
        ) and isinstance(fused_experts, FusedMoEExpertsModular):
            self.impl = FusedMoEKernelModularImpl(
                prepare_finalize,
                fused_experts,
            )

        elif isinstance(
            prepare_finalize, FusedMoEPrepareAndFinalizeMonolithic
        ) and isinstance(fused_experts, FusedMoEExpertsMonolithic):
            self.impl = FusedMoEKernelMonolithicImpl(
                prepare_finalize,
                fused_experts,
            )

        else:
            raise ValueError(
                "prepare_finalize and fused_experts must both be either monolithic "
                f"or non-monolithic but got {prepare_finalize.__class__.__name__} "
                f"and {fused_experts.__class__.__name__}"
            )

        self._post_init_setup()

    @property
    def can_overlap_shared_experts(self) -> bool:
        if isinstance(self.impl, FusedMoEKernelModularImpl):
            return self.impl.prepare_finalize.supports_async()
        else:
            return False

    @property
    def supports_staged_execution(self) -> bool:
        return isinstance(self.impl, FusedMoEKernelModularImpl)

    @property
    def supports_async_staged_execution(self) -> bool:
        return self.supports_staged_execution and self.prepare_finalize.supports_async()

    @property
    def is_monolithic(self) -> bool:
        return isinstance(self.impl, FusedMoEKernelMonolithicImpl)

    @property
    def prepare_finalize(self) -> FusedMoEPrepareAndFinalize:
        return self.impl.prepare_finalize

    @property
    def fused_experts(self) -> FusedMoEExperts:
        return self.impl.fused_experts

    @property
    def moe_config(self) -> FusedMoEConfig:
        return self.fused_experts.moe_config

    def supports_lora(self) -> bool:
        return self.fused_experts.supports_lora()

    def _post_init_setup(self):
        """
        Resolve any leftover setup dependencies between self.prepare_finalize
        and self.fused_experts here.
        """
        self.prepare_finalize.post_init_setup(self.impl.fused_experts)
        assert (
            self.prepare_finalize.activation_format
            == self.fused_experts.activation_format()
        )

    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of fused MoE kernel
        is reduced across all ranks.
        """
        return self.prepare_finalize.output_is_reduced()

    def apply_monolithic(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_logits: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        # grouped topk + fused topk bias parameters
        num_expert_group: int | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        routed_scaling_factor: float | None = None,
        topk_group: int | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.impl, FusedMoEKernelMonolithicImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.apply(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )

    def begin_staged(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> FusedMoEDispatchHandle:
        """Launch dispatch for a staged modular MoE invocation.

        Use with independent model computation between this method,
        :meth:`run_staged_experts`, and :meth:`finish_staged`. Use
        :meth:`apply` when no such overlap opportunity exists.
        """
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.begin_staged(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

    def run_staged_experts(
        self,
        handle: FusedMoEDispatchHandle,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
        persistent_output: bool = True,
    ) -> FusedMoECombineHandle:
        """Complete staged dispatch, run experts, and launch combine."""
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.run_staged_experts(
            handle,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
            persistent_output=persistent_output,
        )

    def finish_staged(self, handle: FusedMoECombineHandle) -> torch.Tensor:
        """Complete staged combine and make its output safe to consume."""
        assert isinstance(self.impl, FusedMoEKernelModularImpl)
        return self.impl.finish_staged(handle)
