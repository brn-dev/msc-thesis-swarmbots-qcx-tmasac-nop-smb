from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

from swarmbots.mjw_env.mjw_torch_utils import masked_mean
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)


@dataclass(slots=True)
class ClimbSettledSnapshot:
    common: MJWCommonSettledSnapshot
    horizontal_progress: float
    height_progress: float


def _compute_climb_reward_kernel(
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    horizontal_progress: torch.Tensor,
    height_progress: torch.Tensor,
    horizontal_goal_radius: float,
    height_goal_radius: float,
    progress_reward_weight: float,
    horizontal_reward_weight: float,
    height_reward_weight: float,
    potential_reward_discount_factor: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    safe_unit_position = torch.where(stable_mask.view(-1, 1, 1), unit_position, torch.zeros_like(unit_position))
    new_horizontal_progress = _compute_climb_horizontal_progress_baseline_torch(
        unit_position=safe_unit_position,
        goal_position=goal_position,
        units_active_mask=units_active_mask,
        goal_radius=horizontal_goal_radius,
    )
    new_height_progress = _compute_climb_axis_progress_baseline_torch(
        unit_position=safe_unit_position,
        goal_position=goal_position,
        units_active_mask=units_active_mask,
        goal_radius=height_goal_radius,
        axis=2,
    )
    horizontal_reward = (
        (new_horizontal_progress * float(potential_reward_discount_factor)) - horizontal_progress
    ) * float(horizontal_reward_weight)
    height_reward = (
        (new_height_progress * float(potential_reward_discount_factor)) - height_progress
    ) * float(height_reward_weight)
    progress_reward = (horizontal_reward + height_reward) * float(progress_reward_weight)

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(horizontal_progress, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return (
        new_horizontal_progress,
        new_height_progress,
        progress_reward,
        horizontal_reward * float(progress_reward_weight),
        height_reward * float(progress_reward_weight),
        guidance_reward,
    )


class _ClimbCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWClimbScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario

    def settle_batch(self, *, specs: list[MJWCommonResetSpec]) -> list[ClimbSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: MJWCommonResetSpec) -> ClimbSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        unit_position = np.stack(
            (
                self.data.qpos[self.bindings.metadata.unit_qpos_adr],
                self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1],
                self.data.qpos[self.bindings.metadata.unit_qpos_adr + 2],
            ),
            axis=-1,
        )
        active_mask = self._pool_active_mask[spec.pool_idx]
        horizontal_progress = _compute_climb_horizontal_progress_baseline_np(
            unit_position=unit_position,
            goal_position=np.asarray(self.scenario.goal_position, dtype=float),
            active_mask=active_mask,
            goal_radius=self.scenario.horizontal_goal_radius,
        )
        height_progress = _compute_climb_axis_progress_baseline_np(
            unit_position=unit_position,
            goal_position=np.asarray(self.scenario.goal_position, dtype=float),
            active_mask=active_mask,
            goal_radius=self.scenario.height_goal_radius,
            axis=2,
        )
        return ClimbSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec),
            horizontal_progress=horizontal_progress,
            height_progress=height_progress,
        )


class ClimbMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWClimbScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: object,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self._goal_position = torch.tensor(
            scenario.goal_position,
            device=bindings.device,
            dtype=torch.float32,
        )
        self._global_obs = self._goal_position.expand(bindings.num_envs, 3).clone()
        self._hidden_local_obs = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, 0),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._hidden_global_obs = torch.zeros((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self.horizontal_progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self.height_progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._unit_qpos_adr = torch.as_tensor(bindings.metadata.unit_qpos_adr, device=bindings.device, dtype=torch.long)
        self._xyz_offsets = torch.tensor([0, 1, 2], device=bindings.device, dtype=torch.long)
        self._cpu_settler = _ClimbCPUResetSettler(scenario=scenario, bindings=bindings)
        self._reward_kernel: Callable[
            ...,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = _compute_climb_reward_kernel
        if scenario.compile_reward_kernel:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_reward_kernel=True requires torch.compile support.")
            if sys.platform == "win32" and shutil.which("cl") is None:
                raise RuntimeError(
                    "compile_reward_kernel=True on this Windows setup requires cl.exe on PATH for torch.compile."
                )
            self._reward_kernel = torch.compile(
                self._reward_kernel,
                mode=scenario.reward_kernel_compile_mode,
                fullgraph=False,
                dynamic=False,
            )

    @property
    def global_obs(self) -> torch.Tensor:
        return self._global_obs

    @property
    def hidden_local_obs(self) -> torch.Tensor:
        return self._hidden_local_obs

    @property
    def hidden_global_obs(self) -> torch.Tensor:
        return self._hidden_global_obs

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> MJWCommonResetBatch:
        return self._sample_common_reset_batch(n_reset=n_reset, rng=rng)

    def select_reset_batch(self, *, reset_batch: MJWCommonResetBatch, mask: torch.Tensor) -> MJWCommonResetBatch:
        return self._select_common_reset_batch(common_reset_batch=reset_batch, mask=mask)

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: MJWCommonResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch)
        mjw.forward(self.bindings.model, self.bindings.data)
        self._global_obs[world_idx] = self._goal_position
        unit_position = self._get_unit_position()[world_idx]
        units_active_mask = self.bindings.units_active_mask[world_idx]
        goal_position = self._global_obs[world_idx]
        self.horizontal_progress[world_idx] = _compute_climb_horizontal_progress_baseline_torch(
            unit_position=unit_position,
            goal_position=goal_position,
            units_active_mask=units_active_mask,
            goal_radius=self.scenario.horizontal_goal_radius,
        )
        self.height_progress[world_idx] = _compute_climb_axis_progress_baseline_torch(
            unit_position=unit_position,
            goal_position=goal_position,
            units_active_mask=units_active_mask,
            goal_radius=self.scenario.height_goal_radius,
            axis=2,
        )

    def build_cpu_reset_specs(self, *, reset_batch: MJWCommonResetBatch) -> list[MJWCommonResetSpec]:
        return self._build_common_reset_specs(common_reset_batch=reset_batch)

    def settle_cpu_reset_specs(self, *, specs: list[MJWCommonResetSpec]) -> list[ClimbSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[ClimbSettledSnapshot]) -> None:
        self._apply_common_settled_snapshot_batch(
            world_idx=world_idx,
            snapshots=[snapshot.common for snapshot in snapshots],
        )
        self.horizontal_progress[world_idx] = torch.as_tensor(
            [snapshot.horizontal_progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.horizontal_progress.dtype,
        )
        self.height_progress[world_idx] = torch.as_tensor(
            [snapshot.height_progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.height_progress.dtype,
        )
        self._global_obs[world_idx] = self._goal_position
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        unit_position = self._get_unit_position()
        (
            new_horizontal_progress,
            new_height_progress,
            progress_reward,
            horizontal_reward,
            height_reward,
            guidance_reward,
        ) = self._reward_kernel(
            unit_position,
            self._global_obs,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.horizontal_progress,
            self.height_progress,
            float(self.scenario.horizontal_goal_radius),
            float(self.scenario.height_goal_radius),
            float(self.scenario.progress_reward_weight),
            float(self.scenario.horizontal_reward_weight),
            float(self.scenario.height_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.horizontal_progress[stable_mask] = new_horizontal_progress[stable_mask]
        self.height_progress[stable_mask] = new_height_progress[stable_mask]
        success_terminations = _compute_climb_goal_success_terminations(
            unit_position=unit_position,
            goal_position=self._global_obs,
            stable_mask=stable_mask,
            units_active_mask=self.bindings.units_active_mask,
            goal_radius=float(self.scenario.goal_radius),
        )
        goal_success_reward = success_terminations.to(dtype=progress_reward.dtype) * (
            float(self.scenario.goal_success_reward) * float(self.scenario.progress_reward_weight)
        )
        progress_reward = progress_reward + goal_success_reward

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "success": success_terminations,
                "progress_reward": progress_reward,
                "horizontal_reward": horizontal_reward,
                "height_reward": height_reward,
                "goal_success_reward": goal_success_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "horizontal": horizontal_reward,
                    "height": height_reward,
                    "success": goal_success_reward,
                    "guidance": guidance_reward,
                },
            },
            terminations=success_terminations,
        )

    def _get_unit_position(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._unit_qpos_adr[:, None] + self._xyz_offsets[None, :]]


def _compute_climb_axis_progress_baseline_np(
    *,
    unit_position: np.ndarray,
    goal_position: np.ndarray,
    active_mask: np.ndarray,
    goal_radius: float,
    axis: int,
) -> float:
    distance_to_goal = np.abs(unit_position[:, axis] - goal_position[axis])
    remaining_distance = np.maximum(distance_to_goal - float(goal_radius), 0.0)
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    return -float(remaining_distance[active_units_mask].mean())


def _compute_climb_horizontal_progress_baseline_np(
    *,
    unit_position: np.ndarray,
    goal_position: np.ndarray,
    active_mask: np.ndarray,
    goal_radius: float,
) -> float:
    distance_to_goal = np.linalg.norm(unit_position[:, :2] - goal_position[np.newaxis, :2], axis=1)
    remaining_distance = np.maximum(distance_to_goal - float(goal_radius), 0.0)
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    return -float(remaining_distance[active_units_mask].mean())


def _compute_climb_axis_progress_baseline_torch(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    goal_radius: float,
    axis: int,
) -> torch.Tensor:
    distance_to_goal = torch.abs(unit_position[:, :, axis] - goal_position[:, axis].unsqueeze(1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    return -masked_mean(remaining_distance, units_active_mask, dim=1)


def _compute_climb_horizontal_progress_baseline_torch(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    goal_radius: float,
) -> torch.Tensor:
    delta = unit_position[:, :, :2] - goal_position[:, :2].unsqueeze(1)
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    return -masked_mean(remaining_distance, units_active_mask, dim=1)


def _compute_climb_goal_success_terminations(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    goal_radius: float,
) -> torch.Tensor:
    delta = unit_position - goal_position.unsqueeze(1)
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    units_inside_goal = distance_to_goal <= float(goal_radius)
    active_units_count = units_active_mask.sum(dim=-1)
    active_units_success = torch.where(units_active_mask, units_inside_goal, torch.ones_like(units_inside_goal))
    return stable_mask & (active_units_count > 0) & active_units_success.all(dim=-1)
