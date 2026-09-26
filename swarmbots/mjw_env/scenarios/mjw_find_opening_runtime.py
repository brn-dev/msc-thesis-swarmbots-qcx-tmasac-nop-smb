from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

from swarmbots.mjw_env.mjw_torch_utils import masked_mean, sample_float_or_bounded_dist
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.scenarios.mjw_find_opening_scenario import (
    MJWFindOpeningRuntimeMetadata,
)


@dataclass(slots=True)
class FindOpeningResetBatch:
    common: MJWCommonResetBatch
    opening_x: torch.Tensor


@dataclass(slots=True)
class FindOpeningResetSpec:
    common: MJWCommonResetSpec
    opening_x: float


@dataclass(slots=True)
class FindOpeningSettledSnapshot:
    common: MJWCommonSettledSnapshot
    opening_x: float
    opening_potential: float
    wall_exploration_visited_cells: np.ndarray


def _compute_wall_exploration_visited_cells_torch(
    *,
    unit_x: torch.Tensor,
    unit_y: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    side_wall_x: float,
    wall_front_y: float,
    cell_depth: float,
    cell_count: int,
) -> torch.Tensor:
    if cell_count == 0:
        return torch.zeros((unit_x.shape[0], 0), device=unit_x.device, dtype=torch.bool)

    valid = (
        stable_mask.unsqueeze(1)
        & units_active_mask
        & (unit_x >= -float(side_wall_x))
        & (unit_x <= float(side_wall_x))
        & (unit_y >= float(wall_front_y) - float(cell_depth))
        & (unit_y <= float(wall_front_y))
    )
    normalized_x = (unit_x + float(side_wall_x)) / (2.0 * float(side_wall_x))
    cell_indices = torch.floor(normalized_x * int(cell_count)).to(dtype=torch.long)
    cell_indices = torch.clamp(cell_indices, min=0, max=int(cell_count) - 1)
    one_hot_cells = torch.nn.functional.one_hot(cell_indices, num_classes=int(cell_count)).to(dtype=torch.bool)
    return (one_hot_cells & valid.unsqueeze(-1)).any(dim=1)


def _compute_find_opening_reward_kernel(
    unit_x: torch.Tensor,
    unit_y: torch.Tensor,
    opening_x: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    opening_potential: torch.Tensor,
    opening_waypoint_y: float,
    opening_distance_reward_weight: float,
    opening_distance_reward_falloff_distance: float | None,
    success_reward_value: float,
    progress_reward_weight: float,
    potential_reward_discount_factor: float,
    wall_exploration_visited_cells: torch.Tensor,
    wall_exploration_cell_count: int,
    wall_exploration_cell_reward: float,
    wall_exploration_cell_depth: float,
    side_wall_x: float,
    wall_front_y: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance_to_waypoint = torch.sqrt(
        torch.square(unit_x - opening_x.unsqueeze(1))
        + torch.square(unit_y - float(opening_waypoint_y))
    )
    shaped_distance = _shape_opening_reward_distance_torch(
        distance_to_waypoint,
        opening_distance_reward_falloff_distance,
    )
    candidate_potential = -masked_mean(shaped_distance, units_active_mask, dim=1)
    new_opening_potential = torch.where(stable_mask, candidate_potential, opening_potential)
    opening_reward = (
        (new_opening_potential * float(potential_reward_discount_factor)) - opening_potential
    ) * float(opening_distance_reward_weight) * float(progress_reward_weight)

    unit_past_barrier = unit_y > float(opening_waypoint_y)
    active_units_count = units_active_mask.sum(dim=-1)
    active_units_success = torch.where(
        units_active_mask,
        unit_past_barrier,
        torch.ones_like(unit_past_barrier),
    )
    success = stable_mask & (active_units_count > 0) & active_units_success.all(dim=-1)
    success_reward = success.to(dtype=torch.float32) * (
        float(success_reward_value) * float(progress_reward_weight)
    )

    current_exploration_cells = _compute_wall_exploration_visited_cells_torch(
        unit_x=unit_x,
        unit_y=unit_y,
        stable_mask=stable_mask,
        units_active_mask=units_active_mask,
        side_wall_x=side_wall_x,
        wall_front_y=wall_front_y,
        cell_depth=wall_exploration_cell_depth,
        cell_count=int(wall_exploration_cell_count),
    )
    newly_visited_cells = current_exploration_cells & ~wall_exploration_visited_cells
    new_wall_exploration_visited_cells = wall_exploration_visited_cells | current_exploration_cells
    exploration_reward = newly_visited_cells.sum(dim=-1).to(dtype=torch.float32) * (
        float(wall_exploration_cell_reward) * float(progress_reward_weight)
    )

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(opening_potential, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)
    return (
        new_opening_potential,
        opening_reward,
        exploration_reward,
        success_reward,
        guidance_reward,
        success,
        new_wall_exploration_visited_cells,
    )


class _FindOpeningCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(
        self,
        *,
        scenario: "MJWFindOpeningScenario",
        bindings: MJWRuntimeBindings,
        barrier_mocap_id: int,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self.barrier_mocap_id = int(barrier_mocap_id)

    def settle_batch(
        self,
        *,
        specs: list[FindOpeningResetSpec],
    ) -> list[FindOpeningSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: FindOpeningResetSpec) -> FindOpeningSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_opening_position(opening_x=spec.opening_x)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        unit_x = self.data.qpos[self.bindings.metadata.unit_qpos_adr]
        unit_y = self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1]
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        return FindOpeningSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            opening_x=float(spec.opening_x),
            opening_potential=_compute_opening_potential_np(
                unit_x=unit_x,
                unit_y=unit_y,
                active_mask=active_mask,
                opening_x=spec.opening_x,
                opening_waypoint_y=self.scenario.wall_y + self.scenario.opening_y_margin,
                opening_distance_reward_falloff_distance=self.scenario.opening_distance_reward_falloff_distance,
            ),
            wall_exploration_visited_cells=_compute_wall_exploration_visited_cells_np(
                unit_x=unit_x,
                unit_y=unit_y,
                active_mask=active_mask,
                side_wall_x=float(self.scenario.side_wall_x),
                wall_front_y=float(self.scenario.wall_y - (self.scenario.wall_thickness / 2.0)),
                cell_depth=float(self.scenario.wall_exploration_cell_depth),
                cell_count=int(self.scenario.wall_exploration_cell_count),
            ),
        )

    def _apply_opening_position(self, *, opening_x: float) -> None:
        self.data.mocap_pos[self.barrier_mocap_id] = [
            float(opening_x),
            float(self.scenario.wall_y),
            0.0,
        ]


class FindOpeningMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWFindOpeningScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWFindOpeningRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self._global_obs = torch.empty((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, 0),
            device=bindings.device,
            dtype=torch.float32,
        )
        hidden_global_dim = 1 + int(scenario.wall_exploration_cell_count)
        self._hidden_global_obs = torch.zeros(
            (bindings.num_envs, hidden_global_dim),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.opening_x = self._hidden_global_obs[:, :1]
        self.opening_potential = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self.wall_exploration_visited_cells = torch.zeros(
            (bindings.num_envs, int(scenario.wall_exploration_cell_count)),
            device=bindings.device,
            dtype=torch.bool,
        )
        self._barrier_mocap_id = int(runtime_metadata.barrier_mocap_id)
        self._cpu_settler = _FindOpeningCPUResetSettler(
            scenario=scenario,
            bindings=bindings,
            barrier_mocap_id=self._barrier_mocap_id,
        )
        self._reward_kernel: Callable[
            ...,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = _compute_find_opening_reward_kernel
        if scenario.compile_reward_kernel:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_reward_kernel=True requires torch.compile support")
            if sys.platform == "win32" and shutil.which("cl") is None:
                raise RuntimeError(
                    "compile_reward_kernel=True on this Windows setup requires cl.exe on PATH for torch.compile"
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

    def sample_reset_batch(
        self,
        *,
        n_reset: int,
        rng: torch.Generator,
    ) -> FindOpeningResetBatch:
        return FindOpeningResetBatch(
            common=self._sample_common_reset_batch(n_reset=n_reset, rng=rng),
            opening_x=sample_float_or_bounded_dist(
                self.scenario.opening_x,
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            ),
        )

    def select_reset_batch(
        self,
        *,
        reset_batch: FindOpeningResetBatch,
        mask: torch.Tensor,
    ) -> FindOpeningResetBatch:
        return FindOpeningResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            opening_x=reset_batch.opening_x[mask],
        )

    def apply_reset_batch(
        self,
        *,
        world_idx: torch.Tensor,
        reset_batch: FindOpeningResetBatch,
    ) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_opening_position(world_idx=world_idx, opening_x=reset_batch.opening_x)
        mjw.forward(self.bindings.model, self.bindings.data)

        self.opening_x[world_idx, 0] = reset_batch.opening_x
        self.opening_potential[world_idx] = _compute_opening_potential_torch(
            unit_x=self._get_unit_x()[world_idx],
            unit_y=self._get_unit_y()[world_idx],
            active_mask=self.bindings.units_active_mask[world_idx],
            opening_x=reset_batch.opening_x,
            opening_waypoint_y=self.scenario.wall_y + self.scenario.opening_y_margin,
            opening_distance_reward_falloff_distance=self.scenario.opening_distance_reward_falloff_distance,
        )
        self.wall_exploration_visited_cells[world_idx] = _compute_wall_exploration_visited_cells_torch(
            unit_x=self._get_unit_x()[world_idx],
            unit_y=self._get_unit_y()[world_idx],
            stable_mask=torch.ones((world_idx.shape[0],), device=self.bindings.device, dtype=torch.bool),
            units_active_mask=self.bindings.units_active_mask[world_idx],
            side_wall_x=float(self.scenario.side_wall_x),
            wall_front_y=float(self.scenario.wall_y - (self.scenario.wall_thickness / 2.0)),
            cell_depth=float(self.scenario.wall_exploration_cell_depth),
            cell_count=int(self.scenario.wall_exploration_cell_count),
        )
        self._sync_wall_exploration_hidden_obs(world_idx=world_idx)

    def build_cpu_reset_specs(
        self,
        *,
        reset_batch: FindOpeningResetBatch,
    ) -> list[FindOpeningResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        opening_x = reset_batch.opening_x.detach().cpu().numpy()
        return [
            FindOpeningResetSpec(
                common=common_specs[i],
                opening_x=float(opening_x[i]),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(
        self,
        *,
        specs: list[FindOpeningResetSpec],
    ) -> list[FindOpeningSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(
        self,
        *,
        world_idx: torch.Tensor,
        snapshots: list[FindOpeningSettledSnapshot],
    ) -> None:
        self._apply_common_settled_snapshot_batch(
            world_idx=world_idx,
            snapshots=[snapshot.common for snapshot in snapshots],
        )
        self.opening_x[world_idx, 0] = torch.as_tensor(
            [snapshot.opening_x for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.opening_x.dtype,
        )
        self.opening_potential[world_idx] = torch.as_tensor(
            [snapshot.opening_potential for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.opening_potential.dtype,
        )
        self.wall_exploration_visited_cells[world_idx] = torch.as_tensor(
            np.stack([snapshot.wall_exploration_visited_cells for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.wall_exploration_visited_cells.dtype,
        )
        self._sync_wall_exploration_hidden_obs(world_idx=world_idx)
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        (
            new_opening_potential,
            opening_reward,
            exploration_reward,
            success_reward,
            guidance_reward,
            success,
            new_wall_exploration_visited_cells,
        ) = self._reward_kernel(
            self._get_unit_x(),
            self._get_unit_y(),
            self.opening_x[:, 0],
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.opening_potential,
            float(self.scenario.wall_y + self.scenario.opening_y_margin),
            float(self.scenario.opening_distance_reward_weight),
            self.scenario.opening_distance_reward_falloff_distance,
            float(self.scenario.success_reward),
            float(self.scenario.progress_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
            self.wall_exploration_visited_cells,
            int(self.scenario.wall_exploration_cell_count),
            float(self.scenario.wall_exploration_cell_reward),
            float(self.scenario.wall_exploration_cell_depth),
            float(self.scenario.side_wall_x),
            float(self.scenario.wall_y - (self.scenario.wall_thickness / 2.0)),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.opening_potential[stable_mask] = new_opening_potential[stable_mask]
        self.wall_exploration_visited_cells[stable_mask] = new_wall_exploration_visited_cells[stable_mask]
        self._sync_wall_exploration_hidden_obs(world_idx=stable_mask)
        progress_reward = opening_reward + exploration_reward + success_reward
        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "success": success,
                "progress_reward": progress_reward,
                "opening_reward": opening_reward,
                "wall_exploration_reward": exploration_reward,
                "success_reward": success_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "opening": opening_reward,
                    "exploration": exploration_reward,
                    "success": success_reward,
                    "guidance": guidance_reward,
                },
            },
            terminations=success,
        )

    def _apply_opening_position(
        self,
        *,
        world_idx: torch.Tensor,
        opening_x: torch.Tensor,
    ) -> None:
        self.bindings.mocap_pos[world_idx, self._barrier_mocap_id, 0] = opening_x
        self.bindings.mocap_pos[world_idx, self._barrier_mocap_id, 1] = float(self.scenario.wall_y)
        self.bindings.mocap_pos[world_idx, self._barrier_mocap_id, 2] = 0.0

    def _get_unit_x(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr]

    def _get_unit_y(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 1]

    def _sync_wall_exploration_hidden_obs(self, *, world_idx: torch.Tensor) -> None:
        if self.wall_exploration_visited_cells.shape[1] == 0:
            return
        self._hidden_global_obs[world_idx, 1:] = self.wall_exploration_visited_cells[world_idx].to(
            dtype=self._hidden_global_obs.dtype
        )


def _compute_opening_potential_np(
    *,
    unit_x: np.ndarray,
    unit_y: np.ndarray,
    active_mask: np.ndarray,
    opening_x: float,
    opening_waypoint_y: float,
    opening_distance_reward_falloff_distance: float | None,
) -> float:
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    distance_to_waypoint = np.hypot(
        np.asarray(unit_x, dtype=float) - float(opening_x),
        np.asarray(unit_y, dtype=float) - float(opening_waypoint_y),
    )
    shaped_distance = _shape_opening_reward_distance_np(
        distance_to_waypoint,
        opening_distance_reward_falloff_distance,
    )
    return -float(shaped_distance[active_units_mask].mean())


def _shape_opening_reward_distance_np(
    distance_to_waypoint: np.ndarray,
    falloff_distance: float | None,
) -> np.ndarray:
    if falloff_distance is None:
        return distance_to_waypoint
    falloff_distance = float(falloff_distance)
    return falloff_distance * np.log1p(distance_to_waypoint / falloff_distance)


def _compute_wall_exploration_visited_cells_np(
    *,
    unit_x: np.ndarray,
    unit_y: np.ndarray,
    active_mask: np.ndarray,
    side_wall_x: float,
    wall_front_y: float,
    cell_depth: float,
    cell_count: int,
) -> np.ndarray:
    if cell_count == 0:
        return np.zeros((0,), dtype=bool)

    active_units_mask = np.asarray(active_mask, dtype=bool)
    unit_x = np.asarray(unit_x, dtype=float)
    unit_y = np.asarray(unit_y, dtype=float)
    in_exploration_strip = (
        active_units_mask
        & (unit_x >= -float(side_wall_x))
        & (unit_x <= float(side_wall_x))
        & (unit_y >= float(wall_front_y) - float(cell_depth))
        & (unit_y <= float(wall_front_y))
    )
    visited_cells = np.zeros((cell_count,), dtype=bool)
    if not in_exploration_strip.any():
        return visited_cells

    cell_width = (2.0 * float(side_wall_x)) / float(cell_count)
    cell_indices = np.floor((unit_x[in_exploration_strip] + float(side_wall_x)) / cell_width).astype(int)
    cell_indices = np.clip(cell_indices, 0, cell_count - 1)
    visited_cells[cell_indices] = True
    return visited_cells


def _compute_opening_potential_torch(
    *,
    unit_x: torch.Tensor,
    unit_y: torch.Tensor,
    active_mask: torch.Tensor,
    opening_x: torch.Tensor,
    opening_waypoint_y: float,
    opening_distance_reward_falloff_distance: float | None,
) -> torch.Tensor:
    distance_to_waypoint = torch.sqrt(
        torch.square(unit_x - opening_x.unsqueeze(1))
        + torch.square(unit_y - float(opening_waypoint_y))
    )
    shaped_distance = _shape_opening_reward_distance_torch(
        distance_to_waypoint,
        opening_distance_reward_falloff_distance,
    )
    return -masked_mean(shaped_distance, active_mask, dim=1)


def _shape_opening_reward_distance_torch(
    distance_to_waypoint: torch.Tensor,
    falloff_distance: float | None,
) -> torch.Tensor:
    if falloff_distance is None:
        return distance_to_waypoint
    falloff_distance = float(falloff_distance)
    return falloff_distance * torch.log1p(distance_to_waypoint / falloff_distance)
