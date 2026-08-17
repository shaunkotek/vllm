# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm.model_executor.models.nemotron_h import (
    NemotronHModel,
    NemotronHMoE,
    NemotronHMoEDecoderLayer,
    _build_scmoe_execution_plan,
    _is_scmoe_enabled,
)


class _FakeMoE(nn.Module):
    def __init__(self, name: str, events: list[tuple]) -> None:
        super().__init__()
        self.name = name
        self.events = events
        self.experts = SimpleNamespace(
            supports_staged_execution=True,
            supports_async_staged_execution=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append(("atomic", self.name, hidden_states.item()))
        return hidden_states + 100

    def begin_staged(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append(("begin", self.name, hidden_states.item()))
        return hidden_states.new_tensor(int(self.name[1:]))

    def run_staged_experts(
        self,
        ticket: torch.Tensor,
        dependency: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append(("experts", self.name, dependency.item(), ticket.item()))
        return ticket

    def finish_staged(
        self,
        ticket: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        self.events.append(("shared", self.name, hidden_states.item()))
        self.events.append(("finish", self.name, ticket.item()))
        return hidden_states + 200


class _FakeMoELayer(NemotronHMoEDecoderLayer):
    def __init__(self, name: str, events: list[tuple]) -> None:
        nn.Module.__init__(self)
        self.name = name
        self.events = events
        self.mixer = _FakeMoE(name, events)

    def normalize(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.append(("normalize", self.name, hidden_states.item()))
        if residual is None:
            residual = hidden_states
        return hidden_states + 10, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.normalize(hidden_states, residual)
        return self.mixer(hidden_states), residual


class _FakeLayer(nn.Module):
    def __init__(self, name: str, events: list[tuple]) -> None:
        super().__init__()
        self.name = name
        self.events = events

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.events.append(("layer", self.name, hidden_states.item()))
        return hidden_states + 1, residual


class _FakeEndNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, None


def _make_model(layers: list[nn.Module], pattern: str) -> NemotronHModel:
    model = object.__new__(NemotronHModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(layers)
    model.start_layer = 0
    model.end_layer = len(layers)
    model._scmoe_execution_plan = _build_scmoe_execution_plan(pattern)
    model.aux_hidden_state_layers = ()
    model.do_not_compile = True
    return model


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("1", True)],
)
def test_scmoe_env(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool):
    if value is None:
        monkeypatch.delenv("VLLM_NEMOTRON_H_SCMOE", raising=False)
    else:
        monkeypatch.setenv("VLLM_NEMOTRON_H_SCMOE", value)
    assert _is_scmoe_enabled() is expected


def test_scmoe_env_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_NEMOTRON_H_SCMOE", "true")
    with pytest.raises(ValueError, match="must be '0' or '1'"):
        _is_scmoe_enabled()


def test_disabled_path_uses_atomic_layer_forward():
    events: list[tuple] = []
    model = _make_model(
        [_FakeMoELayer("E0", events), _FakeLayer("M1", events)],
        "EM",
    )
    model.use_scmoe = False
    model.norm_f = _FakeEndNorm()

    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    with patch(
        "vllm.model_executor.models.nemotron_h.get_pp_group",
        return_value=pp_group,
    ):
        output = model(
            input_ids=None,
            positions=torch.empty(0),
            inputs_embeds=torch.tensor(0.0),
        )

    assert events == [
        ("normalize", "E0", 0.0),
        ("atomic", "E0", 10.0),
        ("layer", "M1", 110.0),
    ]
    assert output.item() == 111.0


def test_scmoe_plan_unrolls_moe_around_current_mamba_layer():
    plan = _build_scmoe_execution_plan("MEMEM")

    assert "".join(step.op for step in plan) == "MGEeMSM"
    gate = next(step for step in plan if step.op == "G")
    assert (gate.layer_idx, gate.target_layer_idx) == (1, 3)


def test_scmoe_plan_makes_moe_after_attention_atomic():
    plan = _build_scmoe_execution_plan("MEM*EMEM")

    assert "".join(step.op for step in plan) == "MEM*GEeMSM"
    assert all(not (step.op == "G" and step.target_layer_idx == 4) for step in plan)


def test_scmoe_executes_explicit_unrolled_plan():
    events: list[tuple] = []
    model = _make_model(
        [
            _FakeLayer("M0", events),
            _FakeMoELayer("E1", events),
            _FakeLayer("M2", events),
            _FakeMoELayer("E3", events),
            _FakeLayer("M4", events),
        ],
        "MEMEM",
    )

    hidden_states, residual = model._forward_scmoe_layers(
        positions=torch.empty(0),
        hidden_states=torch.tensor(0.0),
        residual=None,
        aux_hidden_states=[],
    )

    assert events == [
        ("layer", "M0", 0.0),
        ("normalize", "E1", 1.0),
        ("begin", "E3", 11.0),
        ("atomic", "E1", 11.0),
        ("experts", "E3", 111.0, 3.0),
        ("layer", "M2", 111.0),
        ("normalize", "E3", 112.0),
        ("shared", "E3", 122.0),
        ("finish", "E3", 3.0),
        ("layer", "M4", 322.0),
    ]
    assert hidden_states.item() == 323.0
    assert residual is not None and residual.item() == 1.0


def test_nemotron_moe_staged_wrapper_orders_shared_expert_between_custom_ops():
    events: list[tuple] = []
    moe = object.__new__(NemotronHMoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False

    class _Gate(nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            events.append(("gate", hidden_states.shape))
            return hidden_states.new_zeros((hidden_states.shape[0], 2)), None

    class _Experts:
        def begin_staged(self, **kwargs):
            events.append(("begin", kwargs["hidden_states"].shape))
            return kwargs["hidden_states"].new_empty(0)

        def run_staged_experts(self, ticket, dependency):
            events.append(("experts", dependency.shape))
            return ticket

        def run_staged_shared_experts(self, hidden_states):
            events.append(("shared", hidden_states.shape))
            return hidden_states + 1

        def finish_staged(self, ticket, output_template, shared_output):
            events.append(("finish", output_template.shape))
            return shared_output + 1

    moe.gate = _Gate()
    moe.experts = _Experts()
    hidden_states = torch.zeros((3, 4))

    ticket = moe.begin_staged(hidden_states)
    ticket = moe.run_staged_experts(ticket, hidden_states)
    output = moe.finish_staged(ticket, hidden_states)

    assert events == [
        ("gate", torch.Size([3, 4])),
        ("begin", torch.Size([3, 4])),
        ("experts", torch.Size([3, 4])),
        ("shared", torch.Size([3, 4])),
        ("finish", torch.Size([3, 4])),
    ]
    torch.testing.assert_close(output, torch.full((3, 4), 2.0))


def test_scmoe_rejects_unsupported_staged_layer():
    layers = [_FakeMoELayer("E0", []), _FakeMoELayer("E1", [])]
    layers[1].mixer.experts.supports_staged_execution = False
    model = _make_model(layers, "EE")
    model.use_scmoe = True

    with pytest.raises(RuntimeError, match=r"unsupported layers: \[1\]"):
        model.validate_scmoe_support()


def test_scmoe_warns_when_staged_layer_is_synchronous():
    layers = [_FakeMoELayer("E0", []), _FakeMoELayer("E1", [])]
    layers[1].mixer.experts.supports_async_staged_execution = False
    model = _make_model(layers, "EE")
    model.use_scmoe = True

    with patch(
        "vllm.model_executor.models.nemotron_h.logger.warning_once"
    ) as warning_once:
        model.validate_scmoe_support()

    warning_once.assert_called_once()
    assert warning_once.call_args.args[1:] == (
        "VLLM_NEMOTRON_H_SCMOE",
        [1],
    )
    assert "will not overlap" in warning_once.call_args.args[0]
