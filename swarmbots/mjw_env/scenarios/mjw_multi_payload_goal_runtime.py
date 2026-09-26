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
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import (
    MJWMultiPayloadGoalRuntimeMetadata,
)
from swarmbots.scenario_presets.multi_payload_goal import PAYLOAD_GOAL_IDENTITY_ROT6D


@dataclass(slots=True)
class MultiPayloadGoalResetBatch:
    common: MJWCommonResetBatch
    payload_position: torch.Tensor
    active_payload_mask: torch.Tensor
    goal_position: torch.Tensor


@dataclass(slots=True)
class MultiPayloadGoalResetSpec:
    common: MJWCommonResetSpec
    payload_position: np.ndarray
    active_payload_mask: np.ndarray
    goal_position: np.ndarray


@dataclass(slots=True)
class MultiPayloadGoalSettledSnapshot:
    common: MJWCommonSettledSnapshot
    payload_position: np.ndarray
    active_payload_mask: np.ndarray
    goal_position: np.ndarray
    payload_progress: np.ndarray


def _compute_multi_payload_goal_reward_kernel(
    payload_position: torch.Tensor,
    goal_position: torch.Tensor,
    active_payload_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    payload_progress: torch.Tensor,
    progress_reward_weight: float,
    forward_reward_weight: float,
    potential_reward_discount_factor: float,
    payload_radius: float,
    goal_radius: float,
    success_reward: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    stable_payload_position = torch.where(
        stable_mask.view(-1, 1, 1),
        payload_position,
        torch.zeros_like(payload_position),
    )
    active_mask = active_payload_mask & stable_mask.view(-1, 1)
    goal_position_xyz = torch.cat(
        (
            goal_position,
            torch.full_like(goal_position[..., :1], float(payload_radius)),
        ),
        dim=-1,
    )
    delta = stable_payload_position - goal_position_xyz
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    new_payload_progress = torch.where(active_mask, -remaining_distance, torch.zeros_like(remaining_distance))

    progress_delta = (new_payload_progress * float(potential_reward_discount_factor)) - payload_progress
    forward_reward = (
        progress_delta.sum(dim=1)
        * float(forward_reward_weight)
        * float(progress_reward_weight)
    )
    active_payload_count = active_mask.sum(dim=1)
    payloads_inside_goal = distance_to_goal <= float(goal_radius)
    success = (active_payload_count > 0) & (payloads_inside_goal | ~active_mask).all(dim=1)
    goal_success_reward = success.to(dtype=forward_reward.dtype) * (
        float(success_reward) * float(progress_reward_weight)
    )
    progress_reward = forward_reward + goal_success_reward

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(forward_reward, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return new_payload_progress, progress_reward, forward_reward, goal_success_reward, guidance_reward, success


class _MultiPayloadGoalCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWMultiPayloadGoalScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self._payload_qpos_indices = np.stack(
            [
                np.asarray(mj_utils.qpos_indices_for_body(self.model, f"Payload{payload_idx}"), dtype=np.int64)
                for payload_idx in range(self.scenario.max_payloads)
            ],
            axis=0,
        )
        if self._payload_qpos_indices.shape != (self.scenario.max_payloads, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")
        self._goal_mocap_ids = np.zeros((self.scenario.max_payloads,), dtype=np.int64)
        for payload_idx in range(self.scenario.max_payloads):
            goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"PayloadGoal{payload_idx}")
            self._goal_mocap_ids[payload_idx] = int(self.model.body_mocapid[goal_body_id])

    def settle_batch(
        self,
        *,
        specs: list[MultiPayloadGoalResetSpec],
    ) -> list[MultiPayloadGoalSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: MultiPayloadGoalResetSpec) -> MultiPayloadGoalSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_payload_position(payload_position=spec.payload_position)
        self._apply_goal_markers(
            goal_position=spec.goal_position,
            active_payload_mask=spec.active_payload_mask,
        )
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        payload_position = np.asarray(
            self.data.qpos[self._payload_qpos_indices[:, :3]],
            dtype=np.float64,
        ).copy()
        payload_progress = _compute_payload_progress_np(
            payload_position=payload_position,
            goal_position=spec.goal_position,
            active_payload_mask=spec.active_payload_mask,
            payload_radius=self.scenario.payload_radius,
            goal_radius=self.scenario.goal_radius,
        )
        return MultiPayloadGoalSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            payload_position=payload_position,
            active_payload_mask=spec.active_payload_mask.copy(),
            goal_position=spec.goal_position.copy(),
            payload_progress=payload_progress,
        )

    def _apply_payload_position(self, *, payload_position: np.ndarray) -> None:
        for payload_idx in range(self.scenario.max_payloads):
            self.data.qpos[self._payload_qpos_indices[payload_idx, :3]] = payload_position[payload_idx]
            self.data.qpos[self._payload_qpos_indices[payload_idx, 3:7]] = np.array(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )

    def _apply_goal_markers(self, *, goal_position: np.ndarray, active_payload_mask: np.ndarray) -> None:
        inactive_base = np.asarray(self.scenario.inactive_area_location, dtype=np.float64)
        for payload_idx in range(self.scenario.max_payloads):
            if active_payload_mask[payload_idx]:
                self.data.mocap_pos[self._goal_mocap_ids[payload_idx]] = [
                    float(goal_position[payload_idx, 0]),
                    float(goal_position[payload_idx, 1]),
                    0.0,
                ]
            else:
                self.data.mocap_pos[self._goal_mocap_ids[payload_idx]] = inactive_base


class MultiPayloadGoalMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWMultiPayloadGoalScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWMultiPayloadGoalRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        if runtime_metadata.payload_qpos_indices.shape != (scenario.max_payloads, 7):
            raise ValueError("MultiPayloadGoalMJWScenarioRuntime requires one free-joint row per payload.")

        self.payload_position = torch.zeros(
            (bindings.num_envs, scenario.max_payloads, 3),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.active_payload_mask = torch.zeros(
            (bindings.num_envs, scenario.max_payloads),
            device=bindings.device,
            dtype=torch.bool,
        )
        self.goal_position = torch.zeros(
            (bindings.num_envs, scenario.max_payloads, 2),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.payload_progress = torch.zeros(
            (bindings.num_envs, scenario.max_payloads),
            device=bindings.device,
            dtype=torch.float32,
        )
        global_obs_dim = scenario.max_payloads * 12
        self._global_obs = torch.zeros((bindings.num_envs, global_obs_dim), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, 0),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._hidden_global_obs = torch.zeros((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self._payload_qpos_indices = torch.as_tensor(
            runtime_metadata.payload_qpos_indices,
            device=bindings.device,
            dtype=torch.long,
        )
        self._payload_position_qpos_indices = self._payload_qpos_indices[:, :3]
        self._payload_quat_qpos_indices = self._payload_qpos_indices[:, 3:7]
        self._goal_mocap_ids = torch.as_tensor(
            runtime_metadata.goal_mocap_ids,
            device=bindings.device,
            dtype=torch.long,
        )
        self._cpu_settler = _MultiPayloadGoalCPUResetSettler(scenario=scenario, bindings=bindings)
        self._reward_kernel: Callable[
            ...,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = _compute_multi_payload_goal_reward_kernel
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

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> MultiPayloadGoalResetBatch:
        common = self._sample_common_reset_batch(n_reset=n_reset, rng=rng)
        active_payload_mask = self._sample_active_payload_mask(n_reset=n_reset, rng=rng)
        goal_position = self._sample_goal_positions(n_reset=n_reset, rng=rng)
        return MultiPayloadGoalResetBatch(
            common=common,
            payload_position=self._sample_payload_positions(
                common_reset_batch=common,
                active_payload_mask=active_payload_mask,
                rng=rng,
            ),
            active_payload_mask=active_payload_mask,
            goal_position=goal_position,
        )

    def select_reset_batch(
        self,
        *,
        reset_batch: MultiPayloadGoalResetBatch,
        mask: torch.Tensor,
    ) -> MultiPayloadGoalResetBatch:
        return MultiPayloadGoalResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            payload_position=reset_batch.payload_position[mask],
            active_payload_mask=reset_batch.active_payload_mask[mask],
            goal_position=reset_batch.goal_position[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: MultiPayloadGoalResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_payload_position(world_idx=world_idx, payload_position=reset_batch.payload_position)
        self._apply_goal_markers(
            world_idx=world_idx,
            goal_position=reset_batch.goal_position,
            active_payload_mask=reset_batch.active_payload_mask,
        )
        mjw.forward(self.bindings.model, self.bindings.data)

        self.payload_position[world_idx] = reset_batch.payload_position
        self.active_payload_mask[world_idx] = reset_batch.active_payload_mask
        self.goal_position[world_idx] = reset_batch.goal_position
        self._update_payload_obs(world_idx=world_idx)
        self.payload_progress[world_idx] = _compute_payload_progress_torch(
            payload_position=reset_batch.payload_position,
            goal_position=reset_batch.goal_position,
            active_payload_mask=reset_batch.active_payload_mask,
            payload_radius=float(self.scenario.payload_radius),
            goal_radius=float(self.scenario.goal_radius),
        )

    def build_cpu_reset_specs(
        self,
        *,
        reset_batch: MultiPayloadGoalResetBatch,
    ) -> list[MultiPayloadGoalResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        payload_position = reset_batch.payload_position.detach().cpu().numpy()
        active_payload_mask = reset_batch.active_payload_mask.detach().cpu().numpy()
        goal_position = reset_batch.goal_position.detach().cpu().numpy()
        return [
            MultiPayloadGoalResetSpec(
                common=common_specs[i],
                payload_position=payload_position[i].copy(),
                active_payload_mask=active_payload_mask[i].copy(),
                goal_position=goal_position[i].copy(),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(
        self,
        *,
        specs: list[MultiPayloadGoalResetSpec],
    ) -> list[MultiPayloadGoalSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(
        self,
        *,
        world_idx: torch.Tensor,
        snapshots: list[MultiPayloadGoalSettledSnapshot],
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
        self.active_payload_mask[world_idx] = torch.as_tensor(
            np.stack([snapshot.active_payload_mask for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.active_payload_mask.dtype,
        )
        self.goal_position[world_idx] = torch.as_tensor(
            np.stack([snapshot.goal_position for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.goal_position.dtype,
        )
        self.payload_progress[world_idx] = torch.as_tensor(
            np.stack([snapshot.payload_progress for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.payload_progress.dtype,
        )
        self._update_payload_obs(world_idx=world_idx)
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        current_payload_position = self._get_payload_position()
        self.payload_position[:] = current_payload_position
        self._update_payload_obs()
        (
            new_payload_progress,
            progress_reward,
            forward_reward,
            goal_success_reward,
            guidance_reward,
            success_terminations,
        ) = self._reward_kernel(
            current_payload_position,
            self.goal_position,
            self.active_payload_mask,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.payload_progress,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.payload_radius),
            float(self.scenario.goal_radius),
            float(self.scenario.success_reward),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.payload_progress[stable_mask] = new_payload_progress[stable_mask]

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "success": success_terminations,
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "goal_success_reward": goal_success_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "success": goal_success_reward,
                    "guidance": guidance_reward,
                },
            },
            terminations=success_terminations,
        )

    def _sample_active_payload_mask(self, *, n_reset: int, rng: torch.Generator) -> torch.Tensor:
        counts = torch.as_tensor(
            list(self.scenario.active_payload_count_probs.keys()),
            device=self.bindings.device,
            dtype=torch.long,
        )
        probabilities = torch.as_tensor(
            list(self.scenario.active_payload_count_probs.values()),
            device=self.bindings.device,
            dtype=torch.float32,
        )
        sampled_count_indices = torch.multinomial(probabilities, n_reset, replacement=True, generator=rng)
        active_counts = counts[sampled_count_indices]
        random_scores = torch.rand(
            (n_reset, self.scenario.max_payloads),
            device=self.bindings.device,
            dtype=torch.float32,
            generator=rng,
        )
        payload_rank = torch.empty_like(random_scores, dtype=torch.long)
        payload_rank.scatter_(
            dim=1,
            index=torch.argsort(random_scores, dim=1),
            src=torch.arange(self.scenario.max_payloads, device=self.bindings.device).view(1, -1).expand(n_reset, -1),
        )
        return payload_rank < active_counts.view(-1, 1)

    def _sample_goal_positions(self, *, n_reset: int, rng: torch.Generator) -> torch.Tensor:
        x_min, x_max, y_min, y_max = self.scenario.goal_rect
        unit_samples = torch.rand(
            (n_reset, self.scenario.max_payloads, 2),
            device=self.bindings.device,
            dtype=torch.float32,
            generator=rng,
        )
        goal_position = torch.empty_like(unit_samples)
        goal_position[..., 0] = float(x_min) + unit_samples[..., 0] * (float(x_max) - float(x_min))
        goal_position[..., 1] = float(y_min) + unit_samples[..., 1] * (float(y_max) - float(y_min))
        return goal_position

    def _sample_payload_positions(
        self,
        *,
        common_reset_batch: MJWCommonResetBatch,
        active_payload_mask: torch.Tensor,
        rng: torch.Generator,
    ) -> torch.Tensor:
        n_reset = int(common_reset_batch.swarm_start.shape[0])
        max_payloads = int(self.scenario.max_payloads)
        inactive_base = torch.as_tensor(
            self.scenario.inactive_area_location,
            device=self.bindings.device,
            dtype=torch.float32,
        )
        payload_position = inactive_base.view(1, 1, 3).repeat(n_reset, max_payloads, 1)
        payload_position[:, :, 0] += torch.arange(
            max_payloads,
            device=self.bindings.device,
            dtype=torch.float32,
        ).view(1, -1) * float(self.scenario.payload_spawn_margin)
        payload_position[:, :, 2] = float(self.scenario.payload_radius)

        for reset_idx in range(n_reset):
            active_payload_indices = torch.nonzero(active_payload_mask[reset_idx], as_tuple=False).flatten()
            active_count = int(active_payload_indices.numel())
            if active_count == 0:
                continue
            centered_slots = torch.arange(active_count, device=self.bindings.device, dtype=torch.float32)
            centered_slots -= float(active_count - 1) / 2.0
            slot_x = centered_slots * float(self.scenario.payload_spawn_margin)
            slot_x = slot_x[torch.randperm(active_count, device=self.bindings.device, generator=rng)]
            payload_position[reset_idx, active_payload_indices, 0] = (
                common_reset_batch.swarm_start[reset_idx, 0] + slot_x
            )
            payload_position[reset_idx, active_payload_indices, 1] = (
                common_reset_batch.swarm_start[reset_idx, 1] + float(self.scenario.payload_spawn_y)
            )
            payload_position[reset_idx, active_payload_indices, 2] = float(self.scenario.payload_radius)

        return payload_position

    def _apply_payload_position(self, *, world_idx: torch.Tensor, payload_position: torch.Tensor) -> None:
        identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.bindings.device,
            dtype=self.bindings.qpos.dtype,
        )
        world_indices = world_idx.view(-1, 1, 1)
        self.bindings.qpos[world_indices, self._payload_position_qpos_indices.unsqueeze(0)] = payload_position
        self.bindings.qpos[world_indices, self._payload_quat_qpos_indices.unsqueeze(0)] = identity_quat.view(1, 1, 4)

    def _apply_goal_markers(
        self,
        *,
        world_idx: torch.Tensor,
        goal_position: torch.Tensor,
        active_payload_mask: torch.Tensor,
    ) -> None:
        goal_mocap_position = torch.zeros(
            (*goal_position.shape[:2], 3),
            device=self.bindings.device,
            dtype=self.bindings.mocap_pos.dtype,
        )
        goal_mocap_position[..., :2] = goal_position.to(dtype=self.bindings.mocap_pos.dtype)
        inactive_base = torch.as_tensor(
            self.scenario.inactive_area_location,
            device=self.bindings.device,
            dtype=self.bindings.mocap_pos.dtype,
        )
        goal_mocap_position = torch.where(
            active_payload_mask.unsqueeze(-1),
            goal_mocap_position,
            inactive_base.view(1, 1, 3),
        )
        self.bindings.mocap_pos[world_idx.unsqueeze(1), self._goal_mocap_ids.unsqueeze(0)] = goal_mocap_position

    def _get_payload_position(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._payload_position_qpos_indices]

    def _get_payload_orientation_rot6d(self) -> torch.Tensor:
        payload_quat = self.bindings.qpos[:, self._payload_quat_qpos_indices]
        return quat_to_rot6d_torch(payload_quat)

    def _update_payload_obs(self, *, world_idx: torch.Tensor | None = None) -> None:
        if world_idx is None:
            payload_position = self._get_payload_position()
            payload_rot6d = self._get_payload_orientation_rot6d()
            self.payload_position[:] = payload_position
            self._write_payload_obs(
                target=self._global_obs,
                payload_position=payload_position,
                payload_rot6d=payload_rot6d,
                active_payload_mask=self.active_payload_mask,
                goal_position=self.goal_position,
            )
            return

        reset_qpos = self.bindings.qpos[world_idx]
        payload_position = reset_qpos[:, self._payload_position_qpos_indices]
        payload_quat = reset_qpos[:, self._payload_quat_qpos_indices]
        payload_rot6d = quat_to_rot6d_torch(payload_quat)
        self.payload_position[world_idx] = payload_position
        self._write_payload_obs(
            target=self._global_obs[world_idx],
            payload_position=payload_position,
            payload_rot6d=payload_rot6d,
            active_payload_mask=self.active_payload_mask[world_idx],
            goal_position=self.goal_position[world_idx],
        )

    @staticmethod
    def _write_payload_obs(
        *,
        target: torch.Tensor,
        payload_position: torch.Tensor,
        payload_rot6d: torch.Tensor,
        active_payload_mask: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> None:
        records = target.view(*target.shape[:-1], -1, 12)
        records.zero_()
        identity_rot6d = torch.as_tensor(
            PAYLOAD_GOAL_IDENTITY_ROT6D,
            device=records.device,
            dtype=records.dtype,
        )
        records[..., 4:10] = identity_rot6d
        active = active_payload_mask.to(dtype=records.dtype)
        active_expanded = active_payload_mask.unsqueeze(-1)
        records[..., 0] = active
        records[..., 1:4] = payload_position * active_expanded
        records[..., 4:10] = torch.where(active_expanded, payload_rot6d, identity_rot6d)
        records[..., 10:12] = goal_position * active_expanded


def _compute_payload_progress_np(
    *,
    payload_position: np.ndarray,
    goal_position: np.ndarray,
    active_payload_mask: np.ndarray,
    payload_radius: float,
    goal_radius: float,
) -> np.ndarray:
    goal_position_xyz = np.zeros((goal_position.shape[0], 3), dtype=np.float64)
    goal_position_xyz[:, :2] = goal_position
    goal_position_xyz[:, 2] = float(payload_radius)
    distance_to_goal = np.linalg.norm(payload_position - goal_position_xyz, axis=1)
    remaining_distance = np.maximum(distance_to_goal - float(goal_radius), 0.0)
    progress = np.zeros((goal_position.shape[0],), dtype=np.float64)
    progress[np.asarray(active_payload_mask, dtype=bool)] = -remaining_distance[
        np.asarray(active_payload_mask, dtype=bool)
    ]
    return progress


def _compute_payload_progress_torch(
    *,
    payload_position: torch.Tensor,
    goal_position: torch.Tensor,
    active_payload_mask: torch.Tensor,
    payload_radius: float,
    goal_radius: float,
) -> torch.Tensor:
    goal_position_xyz = torch.cat(
        (
            goal_position,
            torch.full_like(goal_position[..., :1], float(payload_radius)),
        ),
        dim=-1,
    )
    delta = payload_position - goal_position_xyz
    distance_to_goal = torch.sqrt((delta * delta).sum(dim=-1))
    remaining_distance = torch.clamp(distance_to_goal - float(goal_radius), min=0.0)
    return torch.where(active_payload_mask, -remaining_distance, torch.zeros_like(remaining_distance))
