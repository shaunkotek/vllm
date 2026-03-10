# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KernelConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    set_current_vllm_config,
)
from vllm.config.vllm import VllmConfig
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_subset_from_hf,
    get_safetensors_weight_map,
    safetensors_subset_weights_iterator,
)
from vllm.model_executor.models.nemotron_h import NemotronHModel
from vllm.platforms import current_platform
from vllm.transformers_utils.configs import NemotronHConfig
from vllm.utils.torch_utils import set_default_torch_dtype

_MOE_LAYER_CONFIG_ATTRS = (
    "hidden_size",
    "layer_norm_epsilon",
    "mlp_bias",
    "mlp_hidden_act",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "moe_latent_size",
    "num_experts_per_tok",
    "routed_scaling_factor",
    "n_group",
    "topk_group",
    "norm_topk_prob",
)
_NON_EXPERT_LAYER_PREFIXES = (
    "norm.",
    "mixer.gate.",
    "mixer.shared_experts.",
    "mixer.fc1_latent_proj.",
    "mixer.fc2_latent_proj.",
)


@dataclass
class NemotronHSingleMoELayerModel:
    model: NemotronHModel
    vllm_config: VllmConfig
    source_layer_index: int
    source_config: NemotronHConfig
    reduced_config: NemotronHConfig
    downloaded_files: list[str]
    requested_source_weight_names: set[str]
    loaded_weight_names: set[str]


def get_first_nemotron_h_moe_layer_index(config: NemotronHConfig) -> int:
    try:
        return config.hybrid_override_pattern.index("E")
    except ValueError as exc:
        raise ValueError("Nemotron-H config does not contain any MoE layer.") from exc


def make_single_moe_layer_nemotron_h_config(
    source_config: NemotronHConfig,
    source_layer_index: int,
) -> NemotronHConfig:
    config_dict = deepcopy(source_config.to_dict())
    config_dict["num_hidden_layers"] = 1
    config_dict["hybrid_override_pattern"] = "E"

    get_layer_config = getattr(source_config, "get_nemotron_h_config_for_layer", None)
    layer_config = (
        get_layer_config(source_layer_index)
        if callable(get_layer_config)
        else source_config
    )
    for attr_name in _MOE_LAYER_CONFIG_ATTRS:
        if hasattr(layer_config, attr_name):
            config_dict[attr_name] = getattr(layer_config, attr_name)

    return NemotronHConfig(**config_dict)


def get_single_moe_layer_weight_name_map(
    model: NemotronHModel,
    source_layer_index: int,
    available_source_names: set[str] | None = None,
) -> dict[str, str]:
    weight_name_map: dict[str, str] = {}
    if available_source_names is None:
        raise ValueError("available_source_names must be provided.")

    model_param_names = {name for name, _ in model.named_parameters()}
    source_layer_marker = f"layers.{source_layer_index}."

    for source_name in sorted(available_source_names):
        marker_index = source_name.find(source_layer_marker)
        if marker_index < 0:
            continue
        source_suffix = source_name[marker_index:]
        remapped_name = "layers.0." + source_suffix.removeprefix(source_layer_marker)

        if source_suffix.removeprefix(source_layer_marker).startswith(
            _NON_EXPERT_LAYER_PREFIXES
        ):
            if remapped_name in model_param_names:
                weight_name_map[source_name] = remapped_name
        elif source_suffix.removeprefix(source_layer_marker).startswith(
            "mixer.experts."
        ):
            weight_name_map[source_name] = remapped_name

    if not weight_name_map:
        raise ValueError(
            f"Could not resolve any weights for Nemotron-H MoE layer "
            f"{source_layer_index}."
        )

    return weight_name_map


def load_first_nemotron_h_moe_layer_as_single_layer_model(
    checkpoint: str,
    *,
    trust_remote_code: bool = True,
    revision: str | None = None,
    cache_dir: str | None = None,
    use_tqdm_on_load: bool = False,
    safetensors_load_strategy: str = "lazy",
    moe_backend: str = "auto",
) -> NemotronHSingleMoELayerModel:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        with open(checkpoint_path / "config.json") as f:
            source_config = NemotronHConfig(**json.load(f))
        model_config = SimpleNamespace(dtype=torch.float16, quantization=None)
    else:
        model_config = ModelConfig(
            model=checkpoint,
            runner="generate",
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        source_config = model_config.hf_config
        if getattr(source_config, "model_type", None) == NemotronHConfig.model_type:
            source_config = NemotronHConfig(**source_config.to_dict())

    if not isinstance(source_config, NemotronHConfig):
        raise TypeError(
            f"Expected NemotronHConfig, got {type(source_config).__name__}."
        )

    source_layer_index = get_first_nemotron_h_moe_layer_index(source_config)
    reduced_config = make_single_moe_layer_nemotron_h_config(
        source_config, source_layer_index
    )

    if checkpoint_path.exists():
        model_config.hf_config = reduced_config
        model_config.hf_text_config = reduced_config
        vllm_config = VllmConfig(
            cache_config=CacheConfig(),
            parallel_config=ParallelConfig(),
            load_config=LoadConfig(
                download_dir=cache_dir,
                use_tqdm_on_load=use_tqdm_on_load,
            ),
            device_config=DeviceConfig(device=current_platform.device_type),
            kernel_config=KernelConfig(moe_backend=moe_backend),
        )
        vllm_config.model_config = model_config
    else:
        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=CacheConfig(),
            parallel_config=ParallelConfig(),
            load_config=LoadConfig(
                download_dir=cache_dir,
                use_tqdm_on_load=use_tqdm_on_load,
            ),
            device_config=DeviceConfig(device=current_platform.device_type),
            kernel_config=KernelConfig(moe_backend=moe_backend),
        ).with_hf_config(reduced_config)

    available_source_names = set(
        get_safetensors_weight_map(
            checkpoint,
            revision=revision,
            cache_dir=cache_dir,
        )
    )
    init_dtype = vllm_config.model_config.dtype
    with set_default_torch_dtype(init_dtype), set_current_vllm_config(vllm_config):
        model = NemotronHModel(vllm_config=vllm_config)
    weight_name_map = get_single_moe_layer_weight_name_map(
        model,
        source_layer_index,
        available_source_names=available_source_names,
    )

    _, downloaded_files, requested_source_weight_names = (
        download_safetensors_subset_from_hf(
            checkpoint,
            cache_dir=cache_dir,
            tensor_names=weight_name_map,
            revision=revision,
        )
    )

    weights_iter = (
        (weight_name_map[name], tensor)
        for name, tensor in safetensors_subset_weights_iterator(
            downloaded_files,
            requested_source_weight_names,
            use_tqdm_on_load=use_tqdm_on_load,
            safetensors_load_strategy=safetensors_load_strategy,
        )
    )
    loaded_weight_names = model.load_weights(weights_iter)
    process_weights_after_loading(
        model,
        vllm_config.model_config,
        torch.device(current_platform.device_type),
    )

    return NemotronHSingleMoELayerModel(
        model=model,
        vllm_config=vllm_config,
        source_layer_index=source_layer_index,
        source_config=source_config,
        reduced_config=reduced_config,
        downloaded_files=[str(Path(path)) for path in downloaded_files],
        requested_source_weight_names=requested_source_weight_names,
        loaded_weight_names=loaded_weight_names,
    )
