from __future__ import annotations

import shutil
import sys
from typing import Callable

import torch

from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRuntimeBindings, MJWStepResult
from swarmbots.mjw_env.scenarios.mjw_payload_plane_runtime import (
    PayloadPlaneMJWScenarioRuntime,
    PayloadPlaneResetBatch,
    PayloadPlaneSettledSnapshot,
)
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneRuntimeMetadata


def _compute_payload_step_height_potential_torch(
    *,
    payload_position: torch.Tensor,
    step_start_y: float,
    payload_ground_z: float,
    step_height: float,
    payload_step_height_reward_distance: float,
) -> torch.Tensor:
    distance_to_step = float(step_start_y) - payload_position[:, 1]
    approach = torch.clamp(1.0 - (distance_to_step / float(payload_step_height_reward_distance)), min=0.0, max=1.0)
    approach = torch.where(
        (distance_to_step >= 0.0) & (distance_to_step <= float(payload_step_height_reward_distance)),
        torch.sqrt(approach),
        torch.zeros_like(approach),
    )
    height = torch.clamp(
        (payload_position[:, 2] - float(payload_ground_z)) / float(step_height),
        min=0.0,
        max=1.0,
    )
    return approach * height


def _compute_payload_step_reward_extras_kernel(
    payload_position: torch.Tensor,
    stable_mask: torch.Tensor,
    payload_step_height_potential: torch.Tensor,
    payload_step_height_done: torch.Tensor,
    step_start_y: float,
    step_height: float,
    payload_ground_z: float,
    payload_step_height_reward_weight: float,
    payload_step_height_reward_distance: float,
    payload_success_y: float,
    payload_success_min_z: float,
    payload_success_reward_value: float,
    progress_reward_weight: float,
    potential_reward_discount_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_payload_position = torch.where(
        stable_mask.view(-1, 1),
        payload_position,
        torch.zeros_like(payload_position),
    )
    current_potential = _compute_payload_step_height_potential_torch(
        payload_position=safe_payload_position,
        step_start_y=step_start_y,
        payload_ground_z=payload_ground_z,
        step_height=step_height,
        payload_step_height_reward_distance=payload_step_height_reward_distance,
    )
    crossed_step = safe_payload_position[:, 1] >= float(step_start_y)
    newly_done = crossed_step & ~payload_step_height_done
    done = payload_step_height_done | crossed_step
    current_potential = torch.where(done, torch.zeros_like(current_potential), current_potential)
    height_delta = (current_potential * float(potential_reward_discount_factor)) - payload_step_height_potential
    height_delta = torch.where(newly_done & (height_delta < 0.0), torch.zeros_like(height_delta), height_delta)
    payload_step_height_reward = (
        height_delta * float(payload_step_height_reward_weight) * float(progress_reward_weight)
    )

    success = (
        stable_mask
        & (safe_payload_position[:, 1] >= float(payload_success_y))
        & (safe_payload_position[:, 2] >= float(payload_success_min_z))
    )
    payload_success_reward = (
        success.to(dtype=payload_position.dtype)
        * float(payload_success_reward_value)
        * float(progress_reward_weight)
    )
    return current_potential, done, payload_step_height_reward, payload_success_reward, success


class PayloadStepMJWScenarioRuntime(PayloadPlaneMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWPayloadStepScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWPayloadPlaneRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self.payload_step_height_potential = torch.zeros(
            (bindings.num_envs,),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.payload_step_height_done = torch.zeros(
            (bindings.num_envs,),
            device=bindings.device,
            dtype=torch.bool,
        )
        self._payload_step_reward_extras_kernel: Callable[
            ...,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = _compute_payload_step_reward_extras_kernel
        if scenario.compile_reward_kernel:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_reward_kernel=True requires torch.compile support.")
            if sys.platform == "win32" and shutil.which("cl") is None:
                raise RuntimeError(
                    "compile_reward_kernel=True on this Windows setup requires cl.exe on PATH for torch.compile."
                )
            self._payload_step_reward_extras_kernel = torch.compile(
                self._payload_step_reward_extras_kernel,
                mode=scenario.reward_kernel_compile_mode,
                fullgraph=False,
                dynamic=False,
            )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: PayloadPlaneResetBatch) -> None:
        super().apply_reset_batch(world_idx=world_idx, reset_batch=reset_batch)
        self._reset_payload_step_state(world_idx=world_idx)

    def apply_settled_reset_batch(
        self,
        *,
        world_idx: torch.Tensor,
        snapshots: list[PayloadPlaneSettledSnapshot],
    ) -> None:
        super().apply_settled_reset_batch(world_idx=world_idx, snapshots=snapshots)
        self._reset_payload_step_state(world_idx=world_idx)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        current_payload_position = self._get_payload_position()
        self.payload_position[:] = current_payload_position
        self._update_payload_obs()
        (
            new_progress,
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
            self.progress,
            self.towards_payload_progress,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float("inf") if self.scenario.forward_reward_max_y is None else float(self.scenario.forward_reward_max_y),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.payload_radius),
            float(self.scenario.towards_payload_reward_weight),
            float(self.scenario.towards_payload_goal_radius),
            float(self.scenario.payload_centering_penalty_weight),
            float(self.scenario.payload_centering_penalty_power),
            float(self.scenario.payload_centering_tolerance),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        (
            payload_step_height_potential,
            payload_step_height_done,
            payload_step_height_reward,
            payload_success_reward,
            success_terminations,
        ) = self._payload_step_reward_extras_kernel(
            current_payload_position,
            stable_mask,
            self.payload_step_height_potential,
            self.payload_step_height_done,
            float(self.scenario.step_start_y),
            float(self.scenario.step_height),
            float(self.scenario.payload_radius),
            float(self.scenario.payload_step_height_reward_weight),
            float(self.scenario.payload_step_height_reward_distance),
            float("inf") if self.scenario.payload_success_y is None else float(self.scenario.payload_success_y),
            float(self._payload_success_min_z()),
            float(self.scenario.payload_success_reward),
            float(self.scenario.progress_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
        )

        self.progress[stable_mask] = new_progress[stable_mask]
        self.towards_payload_progress[stable_mask] = new_towards_payload_progress[stable_mask]
        self.payload_step_height_potential[stable_mask] = payload_step_height_potential[stable_mask]
        self.payload_step_height_done[stable_mask] = payload_step_height_done[stable_mask]
        progress_reward = progress_reward + payload_step_height_reward + payload_success_reward

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "success": success_terminations,
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "towards_payload_reward": towards_payload_reward,
                "payload_x_penalty": payload_x_penalty,
                "payload_step_height_reward": payload_step_height_reward,
                "payload_success_reward": payload_success_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "towards_payload": towards_payload_reward,
                    "payload_x": payload_x_penalty,
                    "payload_step_height": payload_step_height_reward,
                    "success": payload_success_reward,
                    "guidance": guidance_reward,
                },
            },
            terminations=success_terminations,
        )

    def _reset_payload_step_state(self, *, world_idx: torch.Tensor) -> None:
        payload_position = self.payload_position[world_idx]
        crossed_step = payload_position[:, 1] >= float(self.scenario.step_start_y)
        potential = _compute_payload_step_height_potential_torch(
            payload_position=payload_position,
            step_start_y=float(self.scenario.step_start_y),
            payload_ground_z=float(self.scenario.payload_radius),
            step_height=float(self.scenario.step_height),
            payload_step_height_reward_distance=float(self.scenario.payload_step_height_reward_distance),
        )
        self.payload_step_height_done[world_idx] = crossed_step
        self.payload_step_height_potential[world_idx] = torch.where(
            crossed_step,
            torch.zeros_like(potential),
            potential,
        )

    def _payload_success_min_z(self) -> float:
        return max(
            float(self.scenario.payload_radius),
            float(self.scenario.payload_radius)
            + float(self.scenario.step_height)
            - float(self.scenario.payload_success_height_tolerance),
        )
