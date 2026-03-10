# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch
from safetensors.torch import save_file

from tests.kernels.moe.nemotron_h_test_utils import (
    get_first_nemotron_h_moe_layer_index,
    load_first_nemotron_h_moe_layer_as_single_layer_model,
    make_single_moe_layer_nemotron_h_config,
)
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ParallelConfig,
    set_current_vllm_config,
)
from vllm.config.vllm import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.nemotron_h import NemotronHModel
from vllm.platforms import current_platform
from vllm.transformers_utils.configs import NemotronHConfig

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Nemotron-H MoE layer construction requires distributed GPU setup.",
)


def _make_fake_source_checkpoint(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    source_config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        hybrid_override_pattern="*E-",
        num_attention_heads=4,
        head_dim=4,
        num_key_value_heads=4,
        n_routed_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        num_experts_per_tok=1,
        max_position_embeddings=32,
    )
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(source_config.to_dict(), f)

    reduced_config = make_single_moe_layer_nemotron_h_config(source_config, 1)
    reduced_vllm_config = VllmConfig(
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(),
        load_config=LoadConfig(use_tqdm_on_load=False),
        device_config=DeviceConfig(device=current_platform.device_type),
    )
    reduced_vllm_config.model_config = SimpleNamespace(
        hf_config=reduced_config,
        hf_text_config=reduced_config,
        dtype=torch.float16,
    )
    with set_current_vllm_config(reduced_vllm_config):
        reduced_model = NemotronHModel(vllm_config=reduced_vllm_config)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    with torch.no_grad():
        for _, parameter in reduced_model.named_parameters():
            parameter.copy_(torch.randn_like(parameter, generator=generator))

    moe_tensors = {}
    named_params = dict(reduced_model.named_parameters())
    for name, tensor in named_params.items():
        if not name.startswith("layers.0."):
            continue
        if "mixer.experts.w13_weight" in name or "mixer.experts.w2_weight" in name:
            continue
        source_name = "model.layers.1." + name.removeprefix("layers.0.")
        moe_tensors[source_name] = tensor.detach().cpu().clone()

    fused_w13 = named_params["layers.0.mixer.experts.w13_weight"].detach().cpu()
    fused_w2 = named_params["layers.0.mixer.experts.w2_weight"].detach().cpu()
    for expert_id in range(source_config.n_routed_experts):
        moe_tensors[f"model.layers.1.mixer.experts.{expert_id}.up_proj.weight"] = (
            fused_w13[expert_id].clone()
        )
        moe_tensors[f"model.layers.1.mixer.experts.{expert_id}.down_proj.weight"] = (
            fused_w2[expert_id].clone()
        )

    distractor_tensors = {
        "model.layers.0.mixer.qkv_proj.weight": torch.randn(16, 16),
    }

    shard1_name = "model-00001-of-00002.safetensors"
    shard2_name = "model-00002-of-00002.safetensors"
    save_file(moe_tensors, str(checkpoint_dir / shard1_name))
    save_file(distractor_tensors, str(checkpoint_dir / shard2_name))

    weight_map = {name: shard1_name for name in moe_tensors}
    weight_map.update({name: shard2_name for name in distractor_tensors})
    with open(checkpoint_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)

    return moe_tensors


def test_get_first_nemotron_h_moe_layer_index():
    config = NemotronHConfig(num_hidden_layers=3, hybrid_override_pattern="M*E")
    assert get_first_nemotron_h_moe_layer_index(config) == 2


def test_load_first_nemotron_h_moe_layer_as_single_layer_model(tmp_path, dist_init):
    expected_tensors = _make_fake_source_checkpoint(tmp_path)

    result = load_first_nemotron_h_moe_layer_as_single_layer_model(str(tmp_path))

    assert result.source_layer_index == 1
    assert result.reduced_config.num_hidden_layers == 1
    assert result.reduced_config.hybrid_override_pattern == "E"
    assert len(result.downloaded_files) == 1
    assert result.downloaded_files[0].endswith("model-00001-of-00002.safetensors")
    assert set(result.requested_source_weight_names) == set(expected_tensors)
    assert result.loaded_weight_names == {
        "layers.0.norm.weight",
        "layers.0.mixer.gate.weight",
        "layers.0.mixer.gate.e_score_correction_bias",
        "layers.0.mixer.shared_experts.up_proj.weight",
        "layers.0.mixer.shared_experts.down_proj.weight",
        "layers.0.mixer.experts.w13_weight",
        "layers.0.mixer.experts.w2_weight",
    }

    model_weights = dict(result.model.named_parameters())
    for source_name, expected in expected_tensors.items():
        if ".mixer.experts." not in source_name:
            dest_name = source_name.replace("model.layers.1.", "layers.0.")
            assert dest_name in model_weights
            torch.testing.assert_close(
                model_weights[dest_name].detach().cpu(),
                expected.to(model_weights[dest_name].dtype),
                equal_nan=True,
            )

    for expert_id in range(2):
        expected_up = expected_tensors[
            f"model.layers.1.mixer.experts.{expert_id}.up_proj.weight"
        ]
        expected_down = expected_tensors[
            f"model.layers.1.mixer.experts.{expert_id}.down_proj.weight"
        ]
        torch.testing.assert_close(
            model_weights["layers.0.mixer.experts.w13_weight"]
            .detach()
            .cpu()[expert_id],
            expected_up.to(model_weights["layers.0.mixer.experts.w13_weight"].dtype),
            equal_nan=True,
        )
        torch.testing.assert_close(
            model_weights["layers.0.mixer.experts.w2_weight"].detach().cpu()[expert_id],
            expected_down.to(model_weights["layers.0.mixer.experts.w2_weight"].dtype),
            equal_nan=True,
        )


@pytest.mark.parametrize(
    "model_checkpoint",
    [
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ],
)
@pytest.mark.parametrize(
    "backend", ["flashinfer-latency", "flashinfer-throughput", "triton", "marlin"]
)
def test_real_nemotron_first_moe_layer_forward(
    dist_init,
    workspace_init,
    model_checkpoint: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    if ("NVFP4" in model_checkpoint) and not current_platform.supports_nvfp4:
        pytest.skip("Skipping test because NVFP4 is not supported on this platform")

    _set_envs_for_backend(backend, monkeypatch)

    try:
        result = load_first_nemotron_h_moe_layer_as_single_layer_model(
            model_checkpoint,
            trust_remote_code=True,
        )
    except Exception as exc:
        if isinstance(
            exc,
            (
                OSError,
                RuntimeError,
                huggingface_hub.errors.HfHubHTTPError,
                huggingface_hub.errors.LocalEntryNotFoundError,
            ),
        ):
            pytest.skip(f"Failed to load test checkpoint from HF: {exc}")
        raise

    device = torch.device(current_platform.device_type)
    model = result.model.to(device)
    layer = model.layers[0]

    hidden_states = torch.randn(
        8,
        result.reduced_config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    with (
        set_current_vllm_config(result.vllm_config),
        set_forward_context(
            None,
            result.vllm_config,
            num_tokens=hidden_states.shape[0],
        ),
    ):
        output = layer.mixer(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)


def _set_envs_for_backend(backend, monkeypatch: pytest.MonkeyPatch) -> None:
    if backend == "flashinfer-latency":
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
        monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "latency")
    elif backend == "flashinfer-throughput":
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "1")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "1")
        monkeypatch.setenv("VLLM_FLASHINFER_MOE_BACKEND", "throughput")
    elif backend == "triton":
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP4", "0")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP8", "0")
        monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_FP16", "0")
    elif backend == "marlin":
        pytest.skip("Marlin backend is not supported in CI yet.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
