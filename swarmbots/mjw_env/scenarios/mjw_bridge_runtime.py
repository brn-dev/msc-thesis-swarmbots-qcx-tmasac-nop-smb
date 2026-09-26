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
from swarmbots.mjw_env.scenarios.mjw_bridge_scenario import (
    MJWBridgeRuntimeMetadata,
    MJWBridgeScenario,
)


@dataclass(slots=True)
class BridgeResetBatch:
    common: MJWCommonResetBatch
    bridge_x: torch.Tensor


@dataclass(slots=True)
class BridgeResetSpec:
    common: MJWCommonResetSpec
    bridge_x: float


@dataclass(slots=True)
class BridgeSettledSnapshot:
    common: MJWCommonSettledSnapshot
    bridge_x: float
    progress: float


def _compute_bridge_reward_kernel(
    unit_y: torch.Tensor,
    unit_z: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    progress: torch.Tensor,
    fall_z_threshold: float,
    fell_off_bridge_reward_value: float,
    success_y: float,
    success_reward_value: float,
    progress_reward_weight: float,
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
    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    new_progress = masked_mean(safe_unit_y, units_active_mask, dim=1)
    progress_reward = (
        (new_progress * float(potential_reward_discount_factor)) - progress
    ) * float(progress_reward_weight)

    active_below_threshold = (unit_z < float(fall_z_threshold)) & units_active_mask
    fell_off_bridge = active_below_threshold.any(dim=1) & stable_mask
    fell_off_bridge_reward = torch.where(
        fell_off_bridge,
        torch.full_like(progress_reward, float(fell_off_bridge_reward_value)),
        torch.zeros_like(progress_reward),
    )

    active_units_count = units_active_mask.sum(dim=-1)
    active_units_success = torch.where(
        units_active_mask,
        unit_y > float(success_y),
        torch.ones_like(units_active_mask),
    )
    success = (
        stable_mask
        & ~fell_off_bridge
        & (active_units_count > 0)
        & active_units_success.all(dim=-1)
    )
    success_reward = success.to(dtype=progress_reward.dtype) * (
        float(success_reward_value) * float(progress_reward_weight)
    )

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(progress, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return (
        new_progress,
        progress_reward,
        guidance_reward,
        success,
        success_reward,
        fell_off_bridge,
        fell_off_bridge_reward,
        active_below_threshold,
    )


class _BridgeCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWBridgeScenario", bindings: MJWRuntimeBindings, bridge_mocap_id: int) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self.bridge_mocap_id = int(bridge_mocap_id)

    def settle_batch(self, *, specs: list[BridgeResetSpec]) -> list[BridgeSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: BridgeResetSpec) -> BridgeSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_bridge_position(bridge_x=spec.bridge_x)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        unit_y = self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1]
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        return BridgeSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            bridge_x=float(spec.bridge_x),
            progress=_compute_bridge_progress_baseline_np(unit_y=unit_y, active_mask=active_mask),
        )

    def _apply_bridge_position(self, *, bridge_x: float) -> None:
        self.data.mocap_pos[self.bridge_mocap_id] = [
            float(bridge_x),
            self.scenario.bridge_center_y,
            -(self.scenario.platform_height / 2.0),
        ]


class BridgeMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWBridgeScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWBridgeRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self._global_obs = torch.empty((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, 0),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.bridge_x = torch.zeros((bindings.num_envs, 1), device=bindings.device, dtype=torch.float32)
        self.progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._bridge_mocap_id = int(runtime_metadata.bridge_mocap_id)
        self._cpu_settler = _BridgeCPUResetSettler(
            scenario=scenario,
            bindings=bindings,
            bridge_mocap_id=self._bridge_mocap_id,
        )
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
        ] = _compute_bridge_reward_kernel
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
        return self.bridge_x

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> BridgeResetBatch:
        return BridgeResetBatch(
            common=self._sample_common_reset_batch(n_reset=n_reset, rng=rng),
            bridge_x=sample_float_or_bounded_dist(
                self.scenario.bridge_x,
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            ),
        )

    def select_reset_batch(self, *, reset_batch: BridgeResetBatch, mask: torch.Tensor) -> BridgeResetBatch:
        return BridgeResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            bridge_x=reset_batch.bridge_x[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: BridgeResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_bridge_position(world_idx=world_idx, bridge_x=reset_batch.bridge_x)
        mjw.forward(self.bindings.model, self.bindings.data)

        self.bridge_x[world_idx, 0] = reset_batch.bridge_x
        self.progress[world_idx] = _compute_bridge_progress_baseline_torch(
            unit_y=self._get_unit_y()[world_idx],
            active_mask=self.bindings.units_active_mask[world_idx],
        )

    def build_cpu_reset_specs(self, *, reset_batch: BridgeResetBatch) -> list[BridgeResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        bridge_x = reset_batch.bridge_x.detach().cpu().numpy()
        return [
            BridgeResetSpec(
                common=common_specs[i],
                bridge_x=float(bridge_x[i]),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(self, *, specs: list[BridgeResetSpec]) -> list[BridgeSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[BridgeSettledSnapshot]) -> None:
        self._apply_common_settled_snapshot_batch(
            world_idx=world_idx,
            snapshots=[snapshot.common for snapshot in snapshots],
        )
        self.bridge_x[world_idx, 0] = torch.as_tensor(
            [snapshot.bridge_x for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.bridge_x.dtype,
        )
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
            guidance_reward,
            success,
            success_reward,
            fell_off_bridge,
            fell_off_bridge_reward,
            _active_below_threshold,
        ) = self._reward_kernel(
            self._get_unit_y(),
            self._get_unit_z(),
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.progress,
            float(self.scenario.fall_z_threshold),
            float(self.scenario.fell_off_bridge_reward),
            float(self.scenario.success_y),
            float(self.scenario.success_reward),
            float(self.scenario.progress_reward_weight),
            float(self.scenario.potential_reward_discount_factor),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.progress[stable_mask] = new_progress[stable_mask]
        return MJWStepResult(
            reward=progress_reward + success_reward + guidance_reward + fell_off_bridge_reward,
            info={
                "progress_reward": progress_reward,
                "guidance_reward": guidance_reward,
                "success": success,
                "success_reward": success_reward,
                "fell_off_bridge": fell_off_bridge,
                "fell_off_bridge_reward": fell_off_bridge_reward,
                "reward_terms": {
                    "progress": progress_reward,
                    "success": success_reward,
                    "guidance": guidance_reward,
                    "fall": fell_off_bridge_reward,
                },
            },
            terminations=success | fell_off_bridge,
        )

    def _apply_bridge_position(self, *, world_idx: torch.Tensor, bridge_x: torch.Tensor) -> None:
        self.bindings.mocap_pos[world_idx, self._bridge_mocap_id, 0] = bridge_x
        self.bindings.mocap_pos[world_idx, self._bridge_mocap_id, 1] = float(self.scenario.bridge_center_y)
        self.bindings.mocap_pos[world_idx, self._bridge_mocap_id, 2] = -float(self.scenario.platform_height / 2.0)

    def _get_unit_y(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 1]

    def _get_unit_z(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 2]


def _compute_bridge_progress_baseline_np(*, unit_y: np.ndarray, active_mask: np.ndarray) -> float:
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    return float(np.asarray(unit_y, dtype=float)[active_units_mask].mean())


def _compute_bridge_progress_baseline_torch(*, unit_y: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    return masked_mean(unit_y, active_mask, dim=1)
