from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mjw_env.mjw_torch_quat import quat_to_rot6d_torch
from swarmbots.mjw_env.mjw_torch_utils import sample_float_or_dist
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneRuntimeMetadata


@dataclass(slots=True)
class DualPayloadPlaneResetBatch:
    common: MJWCommonResetBatch
    payload_position: torch.Tensor


@dataclass(slots=True)
class DualPayloadPlaneResetSpec:
    common: MJWCommonResetSpec
    payload_position: np.ndarray


@dataclass(slots=True)
class DualPayloadPlaneSettledSnapshot:
    common: MJWCommonSettledSnapshot
    payload_position: np.ndarray
    payload_progress: np.ndarray
    back_payload_progress: float
    towards_payload_progress: np.ndarray


def _compute_dual_payload_plane_reward_kernel(
    unit_xy: torch.Tensor,
    payload_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    payload_progress: torch.Tensor,
    back_payload_progress: torch.Tensor,
    towards_payload_progress: torch.Tensor,
    progress_reward_weight: float,
    forward_reward_weight: float,
    forward_reward_max_y: float,
    potential_reward_discount_factor: float,
    lagging_payload_weight: float,
    payload_radius: float,
    towards_payload_reward_weight: float,
    towards_payload_goal_radius: float,
    towards_payload_units_per_payload: int,
    payload_centering_penalty_weight: float,
    payload_centering_penalty_power: float,
    payload_centering_tolerance: float,
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
    safe_unit_xy = torch.where(stable_mask.view(-1, 1, 1), unit_xy, torch.zeros_like(unit_xy))
    safe_payload_position = torch.where(stable_mask.view(-1, 1, 1), payload_position, torch.zeros_like(payload_position))
    safe_payload_x = safe_payload_position[:, :, 0]
    safe_payload_y = safe_payload_position[:, :, 1]
    new_payload_progress = torch.clamp(safe_payload_y, max=float(forward_reward_max_y))
    new_back_payload_progress = new_payload_progress.min(dim=1).values
    payload_progress_delta = (new_payload_progress * float(potential_reward_discount_factor)) - payload_progress
    back_progress_delta = (new_back_payload_progress * float(potential_reward_discount_factor)) - back_payload_progress
    num_payloads = float(payload_progress_delta.shape[1])
    blended_forward_delta = (
        ((1.0 - float(lagging_payload_weight)) * payload_progress_delta.sum(dim=1))
        + (float(lagging_payload_weight) * num_payloads * back_progress_delta)
    )

    forward_component_reward = blended_forward_delta * float(forward_reward_weight)
    virtual_goal_position = torch.stack(
        (safe_payload_x, safe_payload_y - float(payload_radius)),
        dim=-1,
    )
    new_towards_payload_progress = _compute_towards_payload_progress_baseline_torch(
        unit_xy=safe_unit_xy,
        virtual_goal_position=virtual_goal_position,
        units_active_mask=units_active_mask,
        goal_radius=towards_payload_goal_radius,
        units_per_payload=towards_payload_units_per_payload,
    )
    towards_payload_reward = (
        (new_towards_payload_progress * float(potential_reward_discount_factor)) - towards_payload_progress
    ).sum(dim=1)
    towards_payload_reward *= float(towards_payload_reward_weight)

    centered_payload_x = torch.clamp(
        torch.abs(safe_payload_x) - float(payload_centering_tolerance),
        min=0.0,
    )
    if float(payload_centering_penalty_power) == 1.0:
        penalty_magnitude = centered_payload_x
    else:
        penalty_magnitude = centered_payload_x.pow(float(payload_centering_penalty_power))
    payload_x_penalty = -penalty_magnitude.sum(dim=1) * float(payload_centering_penalty_weight)

    progress_reward = (forward_component_reward + towards_payload_reward + payload_x_penalty) * float(
        progress_reward_weight
    )
    forward_reward = forward_component_reward * float(progress_reward_weight)
    towards_payload_reward = towards_payload_reward * float(progress_reward_weight)
    payload_x_penalty = payload_x_penalty * float(progress_reward_weight)

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(back_payload_progress, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return (
        new_payload_progress,
        new_back_payload_progress,
        new_towards_payload_progress,
        progress_reward,
        forward_reward,
        towards_payload_reward,
        payload_x_penalty,
        guidance_reward,
    )


class _DualPayloadPlaneCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWDualPayloadPlaneScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self._payload_qpos_indices = np.stack(
            [
                np.asarray(mj_utils.qpos_indices_for_body(self.model, payload_name), dtype=np.int64)
                for payload_name in ("Payload0", "Payload1")
            ],
            axis=0,
        )
        if self._payload_qpos_indices.shape != (2, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")

    def settle_batch(self, *, specs: list[DualPayloadPlaneResetSpec]) -> list[DualPayloadPlaneSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: DualPayloadPlaneResetSpec) -> DualPayloadPlaneSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_payload_position(payload_position=spec.payload_position)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        payload_position = np.asarray(self.data.qpos[self._payload_qpos_indices[:, :3]], dtype=np.float64).copy()
        unit_xy = np.stack(
            (
                self.data.qpos[self.bindings.metadata.unit_qpos_adr],
                self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1],
            ),
            axis=-1,
        )
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        payload_progress = _compute_dual_payload_progress_baseline_np(
            payload_y=payload_position[:, 1],
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )
        back_payload_progress = float(payload_progress.min())
        towards_payload_progress = _compute_dual_towards_payload_progress_baseline_np(
            unit_xy=unit_xy,
            payload_position=payload_position,
            active_mask=active_mask,
            payload_radius=self.scenario.payload_radius,
            goal_radius=self.scenario.towards_payload_goal_radius,
            units_per_payload=self.scenario.towards_payload_units_per_payload,
        )
        return DualPayloadPlaneSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            payload_position=payload_position,
            payload_progress=payload_progress,
            back_payload_progress=back_payload_progress,
            towards_payload_progress=towards_payload_progress,
        )

    def _apply_payload_position(self, *, payload_position: np.ndarray) -> None:
        for payload_idx in range(2):
            self.data.qpos[self._payload_qpos_indices[payload_idx, :3]] = payload_position[payload_idx]
            self.data.qpos[self._payload_qpos_indices[payload_idx, 3:7]] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )


class DualPayloadPlaneMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWDualPayloadPlaneScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWPayloadPlaneRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        if runtime_metadata.payload_qpos_indices.shape != (2, 7):
            raise ValueError("DualPayloadPlaneMJWScenarioRuntime requires two free-joint payload qpos index rows.")

        self.payload_position = torch.zeros((bindings.num_envs, 2, 3), device=bindings.device, dtype=torch.float32)
        global_obs_dim = 18 if scenario.payload_pos_observable else 0
        hidden_global_obs_dim = 0 if scenario.payload_pos_observable else 18
        self._global_obs = torch.zeros((bindings.num_envs, global_obs_dim), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, 0),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._hidden_global_obs = torch.zeros(
            (bindings.num_envs, hidden_global_obs_dim),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.payload_progress = torch.zeros((bindings.num_envs, 2), device=bindings.device, dtype=torch.float32)
        self.back_payload_progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self.towards_payload_progress = torch.zeros(
            (bindings.num_envs, 2),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._payload_qpos_indices = torch.as_tensor(
            runtime_metadata.payload_qpos_indices,
            device=bindings.device,
            dtype=torch.long,
        )
        self._unit_qpos_adr = torch.as_tensor(bindings.metadata.unit_qpos_adr, device=bindings.device, dtype=torch.long)
        self._xy_offsets = torch.tensor([0, 1], device=bindings.device, dtype=torch.long)
        self._cpu_settler = _DualPayloadPlaneCPUResetSettler(scenario=scenario, bindings=bindings)
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
        ] = _compute_dual_payload_plane_reward_kernel
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

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> DualPayloadPlaneResetBatch:
        common = self._sample_common_reset_batch(n_reset=n_reset, rng=rng)
        return DualPayloadPlaneResetBatch(
            common=common,
            payload_position=self._sample_payload_positions(common_reset_batch=common, rng=rng),
        )

    def select_reset_batch(
        self,
        *,
        reset_batch: DualPayloadPlaneResetBatch,
        mask: torch.Tensor,
    ) -> DualPayloadPlaneResetBatch:
        return DualPayloadPlaneResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            payload_position=reset_batch.payload_position[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: DualPayloadPlaneResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_payload_position(world_idx=world_idx, payload_position=reset_batch.payload_position)
        mjw.forward(self.bindings.model, self.bindings.data)

        self._update_payload_obs(world_idx=world_idx)
        payload_progress = _compute_dual_payload_progress_baseline_torch(
            payload_y=reset_batch.payload_position[:, :, 1],
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )
        self.payload_progress[world_idx] = payload_progress
        self.back_payload_progress[world_idx] = payload_progress.min(dim=1).values
        self.towards_payload_progress[world_idx] = _compute_towards_payload_progress_baseline_torch(
            unit_xy=self._get_unit_xy()[world_idx],
            virtual_goal_position=self._virtual_goal_position(reset_batch.payload_position),
            units_active_mask=self.bindings.units_active_mask[world_idx],
            goal_radius=self.scenario.towards_payload_goal_radius,
            units_per_payload=self.scenario.towards_payload_units_per_payload,
        )

    def build_cpu_reset_specs(self, *, reset_batch: DualPayloadPlaneResetBatch) -> list[DualPayloadPlaneResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        payload_position = reset_batch.payload_position.detach().cpu().numpy()
        return [
            DualPayloadPlaneResetSpec(
                common=common_specs[i],
                payload_position=payload_position[i].copy(),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(
        self,
        *,
        specs: list[DualPayloadPlaneResetSpec],
    ) -> list[DualPayloadPlaneSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(
        self,
        *,
        world_idx: torch.Tensor,
        snapshots: list[DualPayloadPlaneSettledSnapshot],
    ) -> None:
        self._apply_common_settled_snapshot_batch(
            world_idx=world_idx,
            snapshots=[snapshot.common for snapshot in snapshots],
        )
        self.payload_position[world_idx] = torch.as_tensor(
            np.stack([snapshot.payload_position for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.payload_position.dtype,
        )
        self._update_payload_obs(world_idx=world_idx)
        self.payload_progress[world_idx] = torch.as_tensor(
            np.stack([snapshot.payload_progress for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.payload_progress.dtype,
        )
        self.back_payload_progress[world_idx] = torch.as_tensor(
            [snapshot.back_payload_progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.back_payload_progress.dtype,
        )
        self.towards_payload_progress[world_idx] = torch.as_tensor(
            np.stack([snapshot.towards_payload_progress for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.towards_payload_progress.dtype,
        )
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        current_payload_position = self._get_payload_position()
        self.payload_position[:] = current_payload_position
        self._update_payload_obs()
        (
            new_payload_progress,
            new_back_payload_progress,
            new_towards_payload_progress,
            progress_reward,
            forward_reward,
            towards_payload_reward,
            payload_x_penalty,
            guidance_reward,
        ) = self._reward_kernel(
            self._get_unit_xy(),
            current_payload_position,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.payload_progress,
            self.back_payload_progress,
            self.towards_payload_progress,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float("inf") if self.scenario.forward_reward_max_y is None else float(self.scenario.forward_reward_max_y),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.lagging_payload_weight),
            float(self.scenario.payload_radius),
            float(self.scenario.towards_payload_reward_weight),
            float(self.scenario.towards_payload_goal_radius),
            int(self.scenario.towards_payload_units_per_payload),
            float(self.scenario.payload_centering_penalty_weight),
            float(self.scenario.payload_centering_penalty_power),
            float(self.scenario.payload_centering_tolerance),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.payload_progress[stable_mask] = new_payload_progress[stable_mask]
        self.back_payload_progress[stable_mask] = new_back_payload_progress[stable_mask]
        self.towards_payload_progress[stable_mask] = new_towards_payload_progress[stable_mask]

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "towards_payload_reward": towards_payload_reward,
                "payload_x_penalty": payload_x_penalty,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "towards_payload": towards_payload_reward,
                    "payload_x": payload_x_penalty,
                    "guidance": guidance_reward,
                },
            },
        )

    def _sample_payload_positions(
        self,
        *,
        common_reset_batch: MJWCommonResetBatch,
        rng: torch.Generator,
    ) -> torch.Tensor:
        n_reset = int(common_reset_batch.swarm_start.shape[0])
        payload_positions = []
        for payload_idx in range(2):
            payload_offset_x = sample_float_or_dist(
                self.scenario.payload_offset_x[payload_idx],
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            )
            payload_offset_y = sample_float_or_dist(
                self.scenario.payload_offset_y[payload_idx],
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            )
            payload_positions.append(
                torch.stack(
                    (
                        common_reset_batch.swarm_start[:, 0] + payload_offset_x,
                        common_reset_batch.swarm_start[:, 1] + payload_offset_y,
                        torch.full(
                            (n_reset,),
                            float(self.scenario.payload_radius),
                            device=self.bindings.device,
                            dtype=torch.float32,
                        ),
                    ),
                    dim=-1,
                )
            )
        return torch.stack(payload_positions, dim=1)

    def _apply_payload_position(self, *, world_idx: torch.Tensor, payload_position: torch.Tensor) -> None:
        identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.bindings.device,
            dtype=self.bindings.qpos.dtype,
        )
        for payload_idx in range(2):
            self.bindings.qpos[
                world_idx.unsqueeze(1),
                self._payload_qpos_indices[payload_idx, :3].unsqueeze(0),
            ] = payload_position[:, payload_idx]
            self.bindings.qpos[
                world_idx.unsqueeze(1),
                self._payload_qpos_indices[payload_idx, 3:7].unsqueeze(0),
            ] = identity_quat

    def _get_payload_position(self) -> torch.Tensor:
        return torch.stack(
            [
                self.bindings.qpos[:, self._payload_qpos_indices[payload_idx, :3]]
                for payload_idx in range(2)
            ],
            dim=1,
        )

    def _get_unit_xy(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._unit_qpos_adr[:, None] + self._xy_offsets[None, :]]

    def _virtual_goal_position(self, payload_position: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                payload_position[:, :, 0],
                payload_position[:, :, 1] - float(self.scenario.payload_radius),
            ),
            dim=-1,
        )

    def _get_payload_orientation_rot6d(self) -> torch.Tensor:
        return torch.stack(
            [
                quat_to_rot6d_torch(self.bindings.qpos[:, self._payload_qpos_indices[payload_idx, 3:7]])
                for payload_idx in range(2)
            ],
            dim=1,
        )

    def _payload_obs_target(self) -> torch.Tensor:
        if self.scenario.payload_pos_observable:
            return self._global_obs
        return self._hidden_global_obs

    def _update_payload_obs(self, *, world_idx: torch.Tensor | None = None) -> None:
        payload_obs = self._payload_obs_target()
        if world_idx is None:
            payload_position = self._get_payload_position()
            payload_rot6d = self._get_payload_orientation_rot6d()
            self.payload_position[:] = payload_position
            payload_obs[:, :9] = torch.cat((payload_position[:, 0], payload_rot6d[:, 0]), dim=-1)
            payload_obs[:, 9:] = torch.cat((payload_position[:, 1], payload_rot6d[:, 1]), dim=-1)
            return

        payload_position = torch.stack(
            [
                self.bindings.qpos[world_idx.unsqueeze(1), self._payload_qpos_indices[payload_idx, :3].unsqueeze(0)]
                for payload_idx in range(2)
            ],
            dim=1,
        )
        payload_quat = torch.stack(
            [
                self.bindings.qpos[world_idx.unsqueeze(1), self._payload_qpos_indices[payload_idx, 3:7].unsqueeze(0)]
                for payload_idx in range(2)
            ],
            dim=1,
        )
        payload_rot6d = torch.stack(
            [quat_to_rot6d_torch(payload_quat[:, payload_idx]) for payload_idx in range(2)],
            dim=1,
        )
        self.payload_position[world_idx] = payload_position
        payload_obs[world_idx, :9] = torch.cat((payload_position[:, 0], payload_rot6d[:, 0]), dim=-1)
        payload_obs[world_idx, 9:] = torch.cat((payload_position[:, 1], payload_rot6d[:, 1]), dim=-1)


def _compute_dual_payload_progress_baseline_np(
    *,
    payload_y: np.ndarray,
    forward_reward_max_y: float | None,
) -> np.ndarray:
    payload_y = np.asarray(payload_y, dtype=np.float64)
    if forward_reward_max_y is not None:
        return np.minimum(payload_y, float(forward_reward_max_y))
    return payload_y


def _compute_dual_payload_progress_baseline_torch(
    *,
    payload_y: torch.Tensor,
    forward_reward_max_y: float | None,
) -> torch.Tensor:
    if forward_reward_max_y is not None:
        return torch.clamp(payload_y, max=float(forward_reward_max_y))
    return payload_y


def _compute_dual_towards_payload_progress_baseline_np(
    *,
    unit_xy: np.ndarray,
    payload_position: np.ndarray,
    active_mask: np.ndarray,
    payload_radius: float,
    goal_radius: float,
    units_per_payload: int,
) -> np.ndarray:
    virtual_goal_position = np.asarray(payload_position[:, :2], dtype=np.float64).copy()
    virtual_goal_position[:, 1] -= float(payload_radius)
    delta = unit_xy[:, np.newaxis, :] - virtual_goal_position[np.newaxis, :, :]
    distance_to_goal = np.linalg.norm(delta, axis=-1)
    remaining_distance = np.maximum(distance_to_goal - float(goal_radius), 0.0)
    active_distances = remaining_distance[np.asarray(active_mask, dtype=bool)]
    active_units_count = int(active_distances.shape[0])
    if active_units_count == 0:
        return np.zeros((2,), dtype=np.float64)
    nearest_count = min(int(units_per_payload), active_units_count)
    nearest_distances = np.sort(active_distances, axis=0)[:nearest_count]
    return -nearest_distances.mean(axis=0)


def _compute_towards_payload_progress_baseline_torch(
    *,
    unit_xy: torch.Tensor,
    virtual_goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    goal_radius: float,
    units_per_payload: int,
) -> torch.Tensor:
    delta = unit_xy.unsqueeze(2) - virtual_goal_position.unsqueeze(1)
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    masked_distances = torch.where(
        units_active_mask.unsqueeze(-1),
        remaining_distance,
        torch.full_like(remaining_distance, float("inf")),
    )
    nearest_count = min(int(units_per_payload), int(unit_xy.shape[1]))
    nearest_distances = torch.topk(masked_distances, k=nearest_count, dim=1, largest=False).values
    finite_nearest_distances = torch.where(
        torch.isfinite(nearest_distances),
        nearest_distances,
        torch.zeros_like(nearest_distances),
    )
    active_units_count = units_active_mask.sum(dim=1)
    denominator = torch.clamp(
        torch.minimum(
            active_units_count,
            torch.full_like(active_units_count, nearest_count),
        ).to(dtype=unit_xy.dtype),
        min=1.0,
    )
    return -(finite_nearest_distances.sum(dim=1) / denominator.unsqueeze(-1))
