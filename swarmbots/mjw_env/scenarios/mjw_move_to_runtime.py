from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

from swarmbots.mjw_env.mjw_torch_utils import masked_mean, sample_float_or_dist
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.scenarios.mjw_move_to_scenario import MJWMoveToRuntimeMetadata
from swarmbots.scenario_presets.move_to_goal_config import AbsoluteGoalConfig, RelativePolarGoalConfig


@dataclass(slots=True)
class MoveToResetBatch:
    common: MJWCommonResetBatch
    goal_position: torch.Tensor


@dataclass(slots=True)
class MoveToResetSpec:
    common: MJWCommonResetSpec
    goal_position: np.ndarray


@dataclass(slots=True)
class MoveToSettledSnapshot:
    common: MJWCommonSettledSnapshot
    goal_position: np.ndarray
    progress: float


def _compute_move_to_reward_kernel(
    unit_xy: torch.Tensor,
    goal_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    progress: torch.Tensor,
    goal_radius: float,
    progress_reward_weight: float,
    forward_reward_weight: float,
    potential_reward_discount_factor: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_unit_xy = torch.where(stable_mask.view(-1, 1, 1), unit_xy, torch.zeros_like(unit_xy))
    new_progress = _compute_goal_progress_baseline_torch(
        unit_xy=safe_unit_xy,
        goal_position=goal_position,
        units_active_mask=units_active_mask,
        goal_radius=goal_radius,
    )
    forward_component_reward = (
        (new_progress * float(potential_reward_discount_factor)) - progress
    ) * float(forward_reward_weight)
    progress_reward = forward_component_reward * float(progress_reward_weight)

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(progress, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return new_progress, progress_reward, progress_reward, guidance_reward


class _MoveToCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWMoveToScenario", bindings: MJWRuntimeBindings, goal_mocap_id: int) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self.goal_mocap_id = int(goal_mocap_id)

    def settle_batch(self, *, specs: list[MoveToResetSpec]) -> list[MoveToSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: MoveToResetSpec) -> MoveToSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_goal_marker(goal_position=spec.goal_position)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        unit_xy = np.stack(
            (
                self.data.qpos[self.bindings.metadata.unit_qpos_adr],
                self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1],
            ),
            axis=-1,
        )
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        progress = _compute_goal_progress_baseline_np(
            unit_xy=unit_xy,
            goal_position=spec.goal_position,
            active_mask=active_mask,
            goal_radius=self.scenario.goal_radius,
        )
        return MoveToSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            goal_position=spec.goal_position.copy(),
            progress=progress,
        )

    def _apply_goal_marker(self, *, goal_position: np.ndarray) -> None:
        if self.data.mocap_pos.size == 0:
            return
        self.data.mocap_pos[self.goal_mocap_id] = [float(goal_position[0]), float(goal_position[1]), 0.0]


class MoveToMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWMoveToScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWMoveToRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self.goal_position = torch.zeros((bindings.num_envs, 2), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros((bindings.num_envs, scenario.swarm.num_units, 0), device=bindings.device, dtype=torch.float32)
        self._hidden_global_obs = torch.zeros((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self.progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._unit_qpos_adr = torch.as_tensor(bindings.metadata.unit_qpos_adr, device=bindings.device, dtype=torch.long)
        self._xy_offsets = torch.tensor([0, 1], device=bindings.device, dtype=torch.long)
        self._goal_mocap_id = int(runtime_metadata.goal_mocap_id)
        self._cpu_settler = _MoveToCPUResetSettler(
            scenario=scenario,
            bindings=bindings,
            goal_mocap_id=self._goal_mocap_id,
        )
        self._reward_kernel: Callable[
            ...,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = _compute_move_to_reward_kernel
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
        return self.goal_position

    @property
    def hidden_local_obs(self) -> torch.Tensor:
        return self._hidden_local_obs

    @property
    def hidden_global_obs(self) -> torch.Tensor:
        return self._hidden_global_obs

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> MoveToResetBatch:
        common = self._sample_common_reset_batch(n_reset=n_reset, rng=rng)
        return MoveToResetBatch(
            common=common,
            goal_position=self._sample_goal_positions(
                n_reset=n_reset,
                rng=rng,
                swarm_start=common.swarm_start,
            ),
        )

    def select_reset_batch(self, *, reset_batch: MoveToResetBatch, mask: torch.Tensor) -> MoveToResetBatch:
        return MoveToResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            goal_position=reset_batch.goal_position[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: MoveToResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_goal_marker(world_idx=world_idx, goal_position=reset_batch.goal_position)
        mjw.forward(self.bindings.model, self.bindings.data)

        self.goal_position[world_idx] = reset_batch.goal_position
        self.progress[world_idx] = _compute_goal_progress_baseline_torch(
            unit_xy=self._get_unit_xy()[world_idx],
            goal_position=reset_batch.goal_position,
            units_active_mask=self.bindings.units_active_mask[world_idx],
            goal_radius=self.scenario.goal_radius,
        )

    def build_cpu_reset_specs(self, *, reset_batch: MoveToResetBatch) -> list[MoveToResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        goal_position = reset_batch.goal_position.detach().cpu().numpy()
        return [
            MoveToResetSpec(
                common=common_specs[i],
                goal_position=goal_position[i].copy(),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(self, *, specs: list[MoveToResetSpec]) -> list[MoveToSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[MoveToSettledSnapshot]) -> None:
        self._apply_common_settled_snapshot_batch(world_idx=world_idx, snapshots=[snapshot.common for snapshot in snapshots])
        self.goal_position[world_idx] = torch.as_tensor(
            np.stack([snapshot.goal_position for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.goal_position.dtype,
        )
        self._apply_goal_marker(world_idx=world_idx, goal_position=self.goal_position[world_idx])
        self.progress[world_idx] = torch.as_tensor(
            [snapshot.progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.progress.dtype,
        )
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        (
            new_progress,
            progress_reward,
            forward_reward,
            guidance_reward,
        ) = self._reward_kernel(
            self._get_unit_xy(),
            self.goal_position,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.progress,
            float(self.scenario.goal_radius),
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.progress[stable_mask] = new_progress[stable_mask]

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "guidance": guidance_reward,
                },
            },
        )

    def _sample_goal_positions(
        self,
        *,
        n_reset: int,
        rng: torch.Generator,
        swarm_start: torch.Tensor,
    ) -> torch.Tensor:
        goal = self.scenario.goal
        if isinstance(goal, RelativePolarGoalConfig):
            distance = sample_float_or_dist(
                goal.distance,
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            )
            angle = sample_float_or_dist(
                goal.angle,
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            )
            offset = torch.stack((distance * torch.cos(angle), distance * torch.sin(angle)), dim=-1)
            return swarm_start[:, :2] + offset
        if not isinstance(goal, AbsoluteGoalConfig):
            raise TypeError(f"Unsupported move-to goal config: {type(goal).__name__}")

        goal_x = sample_float_or_dist(
            goal.x,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        goal_y = sample_float_or_dist(
            goal.y,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        return torch.stack((goal_x, goal_y), dim=-1)

    def _apply_goal_marker(self, *, world_idx: torch.Tensor, goal_position: torch.Tensor) -> None:
        if self.bindings.mocap_pos.numel() == 0:
            return
        marker_position = torch.zeros((int(world_idx.numel()), 3), device=self.bindings.device, dtype=self.bindings.mocap_pos.dtype)
        marker_position[:, :2] = goal_position.to(dtype=self.bindings.mocap_pos.dtype)
        self.bindings.mocap_pos[world_idx, self._goal_mocap_id] = marker_position

    def _get_unit_xy(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._unit_qpos_adr[:, None] + self._xy_offsets[None, :]]


def _compute_goal_progress_baseline_np(
    *,
    unit_xy: np.ndarray,
    goal_position: np.ndarray,
    active_mask: np.ndarray,
    goal_radius: float,
) -> float:
    distance_to_goal = np.linalg.norm(unit_xy - goal_position[np.newaxis, :], axis=1)
    remaining_distance = np.maximum(distance_to_goal - float(goal_radius), 0.0)
    active_units_count = int(np.asarray(active_mask, dtype=bool).sum())
    if active_units_count == 0:
        return 0.0
    return -float(remaining_distance[np.asarray(active_mask, dtype=bool)].mean())


def _compute_goal_progress_baseline_torch(
    *,
    unit_xy: torch.Tensor,
    goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    goal_radius: float,
) -> torch.Tensor:
    delta = unit_xy - goal_position.unsqueeze(1)
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    return -masked_mean(remaining_distance, units_active_mask, dim=1)
