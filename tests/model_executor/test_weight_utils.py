# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile

import huggingface_hub.constants
import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_subset_from_hf,
    download_weights_from_hf,
    enable_hf_transfer,
    get_safetensors_weight_map,
    maybe_remap_kv_scale_name,
    resolve_safetensors_tensor_names,
    safetensors_subset_weights_iterator,
)


def test_hf_transfer_auto_activation():
    if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
        # in case it is already set, we can't test the auto activation
        pytest.skip("HF_HUB_ENABLE_HF_TRANSFER is set, can't test auto activation")
    enable_hf_transfer()
    try:
        # enable hf hub transfer if available
        import hf_transfer  # type: ignore # noqa

        HF_TRANSFER_ACTIVE = True
    except ImportError:
        HF_TRANSFER_ACTIVE = False
    assert huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER == HF_TRANSFER_ACTIVE


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


def test_resolve_safetensors_subset_from_local_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard1 = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        shard2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        save_file(
            {
                "model.layers.0.weight": torch.ones(2, 2),
                "model.layers.0.bias": torch.ones(2),
            },
            shard1,
        )
        save_file(
            {
                "model.layers.1.weight": torch.zeros(2, 2),
                "lm_head.weight": torch.zeros(2, 2),
            },
            shard2,
        )
        with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
            f.write(
                """{
  "metadata": {"total_size": 0},
  "weight_map": {
    "model.layers.0.weight": "model-00001-of-00002.safetensors",
    "model.layers.0.bias": "model-00001-of-00002.safetensors",
    "model.layers.1.weight": "model-00002-of-00002.safetensors",
    "lm_head.weight": "model-00002-of-00002.safetensors"
  }
}
"""
            )

        weight_map = get_safetensors_weight_map(tmpdir)
        assert weight_map["model.layers.0.weight"] == "model-00001-of-00002.safetensors"

        resolved_names, resolved_weight_map = resolve_safetensors_tensor_names(
            tmpdir,
            tensor_name_prefixes=["model.layers.0."],
        )
        assert resolved_names == {
            "model.layers.0.weight",
            "model.layers.0.bias",
        }
        assert resolved_weight_map == weight_map

        hf_folder, local_files, downloaded_names = download_safetensors_subset_from_hf(
            tmpdir,
            cache_dir=None,
            tensor_name_prefixes=["model.layers.0."],
        )
        assert hf_folder == tmpdir
        assert local_files == [shard1]
        assert downloaded_names == resolved_names

        loaded = dict(
            safetensors_subset_weights_iterator(
                local_files,
                downloaded_names,
                use_tqdm_on_load=False,
            )
        )
        assert set(loaded) == resolved_names
        assert torch.equal(loaded["model.layers.0.weight"], torch.ones(2, 2))


def test_resolve_safetensors_tensor_names_missing_inputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        shard = os.path.join(tmpdir, "model.safetensors")
        save_file({"model.layers.0.weight": torch.ones(1)}, shard)

        with pytest.raises(ValueError, match="tensor names"):
            resolve_safetensors_tensor_names(
                tmpdir,
                tensor_names=["missing.weight"],
            )

        with pytest.raises(ValueError, match="tensor prefixes"):
            resolve_safetensors_tensor_names(
                tmpdir,
                tensor_name_prefixes=["missing."],
            )


class TestMaybeRemapKvScaleName:
    """Tests for maybe_remap_kv_scale_name covering all checkpoint formats."""

    PARAMS_DICT = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.self_attn.attn.v_scale": None,
        "model.layers.0.self_attn.attn.q_scale": None,
        "model.layers.0.self_attn.qkv_proj.weight": None,
    }

    def test_qkv_proj_k_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.k_scale -> attn.k_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qkv_proj_v_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.v_scale -> attn.v_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_modelopt_k_proj_k_scale(self):
        """ModelOpt format: k_proj.k_scale -> attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_non_scale_name_unchanged(self):
        """Non-scale names should be returned unchanged."""
        name = "model.layers.0.self_attn.qkv_proj.weight"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_modelopt_k_proj_k_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        k_proj.k_scale -> attn.k_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_nvfp4_modelopt_v_proj_v_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        v_proj.v_scale -> attn.v_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_qwen3_vl_moe_qkv_proj_k_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qwen3_vl_moe_qkv_proj_v_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_nvfp4_weight_scale_not_remapped(self):
        """NVFP4 weight_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.weight_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_input_scale_not_remapped(self):
        """NVFP4 input_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.input_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_missing_target_returns_none(self):
        """If remapped name not in params_dict, return None."""
        empty_params: dict[str, None] = {}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", empty_params
        )
        assert result is None


if __name__ == "__main__":
    test_hf_transfer_auto_activation()
    test_download_weights_from_hf()
