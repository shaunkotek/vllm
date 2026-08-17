# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class StagedMoESchedule:
    """Select the expert boundary for a shortcut-connected MoE (ScMoE).

    Shortcut-connected MoE architectures can launch routed-expert dispatch
    from an earlier activation, execute independent current-path stages, and
    choose one boundary at which to run the routed experts. Model adapters use
    this descriptor to keep that boundary explicit and configuration driven.
    """

    expert_checkpoint: int
    num_compute_stages: int

    def __post_init__(self) -> None:
        if self.num_compute_stages < 0:
            raise ValueError("num_compute_stages must be non-negative")
        if not 0 <= self.expert_checkpoint <= self.num_compute_stages:
            raise ValueError(
                "expert_checkpoint must identify a boundary between compute stages"
            )

    def should_run_experts(self, checkpoint: int) -> bool:
        """Return whether experts should run at the supplied stage boundary."""
        if not 0 <= checkpoint <= self.num_compute_stages:
            raise ValueError(f"checkpoint must be in [0, {self.num_compute_stages}]")
        return checkpoint == self.expert_checkpoint
