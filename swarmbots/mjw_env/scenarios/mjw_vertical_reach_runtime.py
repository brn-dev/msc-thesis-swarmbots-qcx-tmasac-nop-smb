from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

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
class VerticalReachSettledSnapshot:
    common: MJWCommonSettledSnapshot
    horizontal_progress: float
    height_progress: float
    reach_column_progress: float


def _compute_vertical_reach_reward_kernel(
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    horizontal_goal_position: torch.Tensor,
    goal_box_half_size: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    horizontal_progress: torch.Tensor,
    height_progress: torch.Tensor,
    reach_column_progress: torch.Tensor,
    progress_reward_weight: float,
    horizontal_reward_weight: float,
    height_reward_weight: float,
    reach_column_reward_weight: float,
    reach_column_half_width: float,
    reach_column_min_y: float,
    reach_column_max_y: float,
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
    torch.Tensor,
    torch.Tensor,
]:
    new_horizontal_progress, new_height_progress = _compute_vertical_reach_progress_baselines_torch(
        unit_position=unit_position,
        goal_position=goal_position,
        horizontal_goal_position=horizontal_goal_position,
        goal_box_half_size=goal_box_half_size,
        units_active_mask=units_active_mask,
        horizontal_reward_weight=horizontal_reward_weight,
        height_reward_weight=height_reward_weight,
    )
    new_horizontal_progress = torch.where(stable_mask, new_horizontal_progress, horizontal_progress)
    new_height_progress = torch.where(stable_mask, new_height_progress, height_progress)
    new_reach_column_progress = _compute_reach_column_progress_baseline_torch(
        unit_position=unit_position,
        goal_position=goal_position,
        units_active_mask=units_active_mask,
        reach_column_half_width=reach_column_half_width,
        reach_column_min_y=reach_column_min_y,
        reach_column_max_y=reach_column_max_y,
    )
    new_reach_column_progress = torch.where(stable_mask, new_reach_column_progress, reach_column_progress)

    horizontal_reward = (
        (new_horizontal_progress * float(potential_reward_discount_factor)) - horizontal_progress
    ) * float(horizontal_reward_weight)
    height_reward = (
        (new_height_progress * float(potential_reward_discount_factor)) - height_progress
    ) * float(height_reward_weight)
    reach_column_reward = (
        (new_reach_column_progress * float(potential_reward_discount_factor)) - reach_column_progress
    ) * float(reach_column_reward_weight)
    progress_reward = (horizontal_reward + height_reward + reach_column_reward) * float(progress_reward_weight)

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
        new_reach_column_progress,
        progress_reward,
        horizontal_reward * float(progress_reward_weight),
        height_reward * float(progress_reward_weight),
        reach_column_reward * float(progress_reward_weight),
        guidance_reward,
    )


class _VerticalReachCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWVerticalReachScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario

    def settle_batch(self, *, specs: list[MJWCommonResetSpec]) -> list[VerticalReachSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: MJWCommonResetSpec) -> VerticalReachSettledSnapshot:
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
        horizontal_progress, height_progress = _compute_vertical_reach_progress_baselines_np(
            unit_position=unit_position,
            goal_position=np.asarray(self.scenario.goal_position, dtype=float),
            horizontal_goal_position=np.asarray(self.scenario.horizontal_goal_position, dtype=float),
            goal_box_half_size=np.asarray(self.scenario.goal_box_half_size, dtype=float),
            active_mask=self._pool_active_mask[spec.pool_idx],
            horizontal_reward_weight=self.scenario.horizontal_reward_weight,
            height_reward_weight=self.scenario.height_reward_weight,
        )
        reach_column_progress = _compute_reach_column_progress_baseline_np(
            unit_position=unit_position,
            goal_position=np.asarray(self.scenario.goal_position, dtype=float),
            active_mask=self._pool_active_mask[spec.pool_idx],
            reach_column_half_width=self.scenario.reach_column_half_width,
            reach_column_min_y=self.scenario.reach_column_min_y,
            reach_column_max_y=self.scenario.reach_column_max_y,
        )
        return VerticalReachSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec),
            horizontal_progress=horizontal_progress,
            height_progress=height_progress,
            reach_column_progress=reach_column_progress,
        )


class VerticalReachMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWVerticalReachScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: object,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self._goal_position = torch.tensor(
            scenario.goal_position,
            device=bindings.device,
            dtype=torch.float32,
        )
        self._goal_box_half_size = torch.tensor(
            scenario.goal_box_half_size,
            device=bindings.device,
            dtype=torch.float32,
        )
        self._horizontal_goal_position = torch.tensor(
            scenario.horizontal_goal_position,
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
        self.reach_column_progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._unit_qpos_adr = torch.as_tensor(bindings.metadata.unit_qpos_adr, device=bindings.device, dtype=torch.long)
        self._xyz_offsets = torch.tensor([0, 1, 2], device=bindings.device, dtype=torch.long)
        self._cpu_settler = _VerticalReachCPUResetSettler(scenario=scenario, bindings=bindings)
        self._reward_kernel: Callable[
            ...,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = _compute_vertical_reach_reward_kernel
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
        horizontal_progress, height_progress = _compute_vertical_reach_progress_baselines_torch(
            unit_position=unit_position,
            goal_position=goal_position,
            horizontal_goal_position=self._horizontal_goal_position.expand(world_idx.numel(), 3),
            goal_box_half_size=self._goal_box_half_size,
            units_active_mask=units_active_mask,
            horizontal_reward_weight=self.scenario.horizontal_reward_weight,
            height_reward_weight=self.scenario.height_reward_weight,
        )
        self.horizontal_progress[world_idx] = horizontal_progress
        self.height_progress[world_idx] = height_progress
        self.reach_column_progress[world_idx] = _compute_reach_column_progress_baseline_torch(
            unit_position=unit_position,
            goal_position=goal_position,
            units_active_mask=units_active_mask,
            reach_column_half_width=self.scenario.reach_column_half_width,
            reach_column_min_y=self.scenario.reach_column_min_y,
            reach_column_max_y=self.scenario.reach_column_max_y,
        )

    def build_cpu_reset_specs(self, *, reset_batch: MJWCommonResetBatch) -> list[MJWCommonResetSpec]:
        return self._build_common_reset_specs(common_reset_batch=reset_batch)

    def settle_cpu_reset_specs(self, *, specs: list[MJWCommonResetSpec]) -> list[VerticalReachSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[VerticalReachSettledSnapshot]) -> None:
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
        self.reach_column_progress[world_idx] = torch.as_tensor(
            [snapshot.reach_column_progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.reach_column_progress.dtype,
        )
        self._global_obs[world_idx] = self._goal_position
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        unit_position = self._get_unit_position()
        (
            new_horizontal_progress,
            new_height_progress,
            new_reach_column_progress,
            progress_reward,
            horizontal_reward,
            height_reward,
            reach_column_reward,
            guidance_reward,
        ) = self._reward_kernel(
            unit_position,
            self._global_obs,
            self._horizontal_goal_position.expand(unit_position.shape[0], 3),
            self._goal_box_half_size,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.horizontal_progress,
            self.height_progress,
            self.reach_column_progress,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.horizontal_reward_weight),
            float(self.scenario.height_reward_weight),
            float(self.scenario.reach_column_reward_weight),
            float(self.scenario.reach_column_half_width),
            float(self.scenario.reach_column_min_y),
            float(self.scenario.reach_column_max_y),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.horizontal_progress[stable_mask] = new_horizontal_progress[stable_mask]
        self.height_progress[stable_mask] = new_height_progress[stable_mask]
        self.reach_column_progress[stable_mask] = new_reach_column_progress[stable_mask]
        success_terminations = _compute_vertical_reach_goal_success_terminations(
            unit_position=unit_position,
            goal_position=self._global_obs,
            goal_box_half_size=self._goal_box_half_size,
            stable_mask=stable_mask,
            units_active_mask=self.bindings.units_active_mask,
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
                "reach_column_reward": reach_column_reward,
                "goal_success_reward": goal_success_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "horizontal": horizontal_reward,
                    "height": height_reward,
                    "reach_column": reach_column_reward,
                    "success": goal_success_reward,
                    "guidance": guidance_reward,
                },
            },
            terminations=success_terminations,
        )

    def _get_unit_position(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._unit_qpos_adr[:, None] + self._xyz_offsets[None, :]]


def _compute_vertical_reach_progress_baselines_np(
    *,
    unit_position: np.ndarray,
    goal_position: np.ndarray,
    horizontal_goal_position: np.ndarray,
    goal_box_half_size: np.ndarray,
    active_mask: np.ndarray,
    horizontal_reward_weight: float,
    height_reward_weight: float,
) -> tuple[float, float]:
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0, 0.0

    horizontal_abs_delta = np.abs(unit_position[:, :2] - horizontal_goal_position[np.newaxis, :2])
    horizontal_outside_distance = np.maximum(horizontal_abs_delta - goal_box_half_size[np.newaxis, :2], 0.0)
    horizontal_distance = np.linalg.norm(horizontal_outside_distance, axis=1)
    height_distance = np.maximum(np.abs(unit_position[:, 2] - goal_position[2]) - goal_box_half_size[2], 0.0)
    weighted_distance = (
        horizontal_distance * float(horizontal_reward_weight)
        + height_distance * float(height_reward_weight)
    )
    weighted_distance = np.where(active_units_mask, weighted_distance, np.inf)
    best_unit_idx = int(np.argmin(weighted_distance))
    return -float(horizontal_distance[best_unit_idx]), -float(height_distance[best_unit_idx])


def _compute_vertical_reach_progress_baselines_torch(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    horizontal_goal_position: torch.Tensor,
    goal_box_half_size: torch.Tensor,
    units_active_mask: torch.Tensor,
    horizontal_reward_weight: float,
    height_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    horizontal_abs_delta = torch.abs(unit_position[:, :, :2] - horizontal_goal_position[:, :2].unsqueeze(1))
    horizontal_outside_distance = torch.clamp(horizontal_abs_delta - goal_box_half_size[:2].view(1, 1, 2), min=0.0)
    horizontal_distance = torch.sqrt((horizontal_outside_distance * horizontal_outside_distance).sum(dim=-1))
    height_distance = torch.clamp(
        torch.abs(unit_position[:, :, 2] - goal_position[:, 2].unsqueeze(1)) - goal_box_half_size[2],
        min=0.0,
    )
    weighted_distance = (
        horizontal_distance * float(horizontal_reward_weight)
        + height_distance * float(height_reward_weight)
    )
    inf_distance = torch.full_like(weighted_distance, float("inf"))
    weighted_distance = torch.where(units_active_mask, weighted_distance, inf_distance)
    best_unit_idx = weighted_distance.argmin(dim=1)
    horizontal_progress = -horizontal_distance.gather(1, best_unit_idx.view(-1, 1)).squeeze(1)
    height_progress = -height_distance.gather(1, best_unit_idx.view(-1, 1)).squeeze(1)
    active_units_count = units_active_mask.sum(dim=-1)
    horizontal_progress = torch.where(active_units_count > 0, horizontal_progress, torch.zeros_like(horizontal_progress))
    height_progress = torch.where(active_units_count > 0, height_progress, torch.zeros_like(height_progress))
    return horizontal_progress, height_progress


def _compute_reach_column_progress_baseline_np(
    *,
    unit_position: np.ndarray,
    goal_position: np.ndarray,
    active_mask: np.ndarray,
    reach_column_half_width: float,
    reach_column_min_y: float,
    reach_column_max_y: float,
) -> float:
    active_units_mask = np.asarray(active_mask, dtype=bool)
    in_column = (
        active_units_mask
        & (np.abs(unit_position[:, 0] - goal_position[0]) <= float(reach_column_half_width))
        & (unit_position[:, 1] >= float(reach_column_min_y))
        & (unit_position[:, 1] <= float(reach_column_max_y))
    )
    max_z = float(unit_position[in_column, 2].max()) if in_column.any() else 0.0
    return -max(float(goal_position[2]) - max_z, 0.0)


def _compute_reach_column_progress_baseline_torch(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    reach_column_half_width: float,
    reach_column_min_y: float,
    reach_column_max_y: float,
) -> torch.Tensor:
    in_column = (
        units_active_mask
        & (torch.abs(unit_position[:, :, 0] - goal_position[:, 0].unsqueeze(1)) <= float(reach_column_half_width))
        & (unit_position[:, :, 1] >= float(reach_column_min_y))
        & (unit_position[:, :, 1] <= float(reach_column_max_y))
    )
    neg_inf = torch.full_like(unit_position[:, :, 2], -float("inf"))
    max_z = torch.where(in_column, unit_position[:, :, 2], neg_inf).max(dim=1).values
    max_z = torch.where(in_column.any(dim=1), max_z, torch.zeros_like(max_z))
    return -torch.clamp(goal_position[:, 2] - max_z, min=0.0)


def _compute_vertical_reach_goal_success_terminations(
    *,
    unit_position: torch.Tensor,
    goal_position: torch.Tensor,
    goal_box_half_size: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
) -> torch.Tensor:
    units_inside_goal = (torch.abs(unit_position - goal_position.unsqueeze(1)) <= goal_box_half_size.view(1, 1, 3)).all(
        dim=-1
    )
    active_units_inside_goal = units_inside_goal & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    return stable_mask & (active_units_count > 0) & active_units_inside_goal.any(dim=-1)
