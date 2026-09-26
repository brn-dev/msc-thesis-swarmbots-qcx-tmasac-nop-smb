from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any, Protocol

import mujoco
import numpy as np
import torch
import warp as wp
from gymnasium import spaces

from swarmbots.mjw_env.mjw_kernels import apply_reset_unit_pose
from swarmbots.mjw_env.mjw_model_metadata import MJWModelMetadata
from swarmbots.mjw_env.mjw_torch_utils import sample_float_or_dist
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm, MJWSwarmPool


@dataclass(slots=True)
class MJWRuntimeBindings:
    device: torch.device
    wp_device: Any
    num_envs: int
    model: Any
    data: Any
    metadata: MJWModelMetadata
    pool: MJWSwarmPool
    pool_eq_active: torch.Tensor
    inactive_unit_positions: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor
    eq_active: torch.Tensor
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    xpos: torch.Tensor
    xquat: torch.Tensor
    xmat: torch.Tensor
    time: torch.Tensor
    qacc_warmstart: torch.Tensor
    act: torch.Tensor
    units_active_mask: torch.Tensor
    partner_unit: torch.Tensor
    partner_connector: torch.Tensor
    connection_twist_idx: torch.Tensor
    disconnect_potentials: torch.Tensor
    current_step: torch.Tensor
    is_first_episode: torch.Tensor
    base_qpos: torch.Tensor
    base_mocap_pos: torch.Tensor
    base_mocap_quat: torch.Tensor
    unit_qpos_adr: torch.Tensor
    qpos_wp: Any
    pool_active_mask_wp: Any
    pool_positions_wp: Any
    pool_quats_wp: Any
    inactive_unit_positions_wp: Any
    unit_qpos_adr_wp: Any


@dataclass(slots=True)
class MJWCommonResetBatch:
    pool_idx: torch.Tensor
    swarm_start: torch.Tensor
    initial_z_rotation: torch.Tensor


@dataclass(slots=True)
class MJWCommonResetSpec:
    pool_idx: int
    swarm_start: np.ndarray
    initial_z_rotation: float


@dataclass(slots=True)
class MJWCommonSettledSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    eq_active: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    time: float
    units_active_mask: np.ndarray
    partner_unit: np.ndarray
    partner_connector: np.ndarray
    connection_twist_idx: np.ndarray


@dataclass(slots=True)
class MJWStepResult:
    reward: torch.Tensor
    info: dict[str, Any]
    terminations: torch.Tensor | None = None


@dataclass(slots=True)
class MJWRecordingCameraConfig:
    lookat: tuple[float, float, float]
    distance: float
    azimuth: float
    elevation: float


class BaseMJWScenario(Protocol):
    swarm: MJWHomogeneousSwarm
    action_repeat: int
    actuator_strength: float
    inactive_area_location: tuple[float, float, float]
    include_connectors_xpos_in_obs: bool
    include_connectors_xquat_in_obs: bool
    quat_rot6d_representation: bool
    connection_dist_threshold: float
    connection_angle_threshold: float
    disconnect_potential_threshold: float
    continuous_connector_actions: bool
    potential_reward_discount_factor: float
    reset_settle_time: float
    reset_settle_timestep_scale: float
    swarm_start_x: Any
    swarm_start_y: Any
    randomize_initial_swarm_z_rotation: bool

    def get_settings(self) -> dict[str, Any]: ...
    def build_model(self) -> mujoco.MjModel: ...
    def get_single_observation_space(self) -> spaces.Dict: ...
    def get_single_action_space(self) -> spaces.Dict: ...
    def get_batched_observation_space(self, num_envs: int) -> spaces.Dict: ...
    def get_batched_action_space(self, num_envs: int) -> spaces.Dict: ...
    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None: ...
    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> Any: ...
    def create_runtime(
        self,
        *,
        bindings: MJWRuntimeBindings,
        runtime_metadata: Any,
    ) -> "BaseMJWScenarioRuntime": ...


class BaseMJWScenarioRuntime(abc.ABC):
    def __init__(
        self,
        *,
        scenario: BaseMJWScenario,
        bindings: MJWRuntimeBindings,
        runtime_metadata: Any = None,
    ) -> None:
        self.scenario = scenario
        self.bindings = bindings
        self.runtime_metadata = runtime_metadata

    @property
    @abc.abstractmethod
    def global_obs(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hidden_local_obs(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def hidden_global_obs(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def select_reset_batch(self, *, reset_batch: Any, mask: torch.Tensor) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: Any) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def build_cpu_reset_specs(self, *, reset_batch: Any) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def settle_cpu_reset_specs(self, *, specs: list[Any]) -> list[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[Any]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        raise NotImplementedError

    def _sample_common_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> MJWCommonResetBatch:
        active_pool_size = self.scenario.swarm.get_active_pool_size()
        return MJWCommonResetBatch(
            pool_idx=torch.randint(active_pool_size, (n_reset,), device=self.bindings.device, generator=rng),
            swarm_start=self._sample_swarm_start(n_reset=n_reset, rng=rng),
            initial_z_rotation=self._sample_initial_z_rotation(n_reset=n_reset, rng=rng),
        )

    def _select_common_reset_batch(
        self,
        *,
        common_reset_batch: MJWCommonResetBatch,
        mask: torch.Tensor,
    ) -> MJWCommonResetBatch:
        return MJWCommonResetBatch(
            pool_idx=common_reset_batch.pool_idx[mask],
            swarm_start=common_reset_batch.swarm_start[mask],
            initial_z_rotation=common_reset_batch.initial_z_rotation[mask],
        )

    def _apply_common_reset_batch(self, *, world_idx: torch.Tensor, common_reset_batch: MJWCommonResetBatch) -> None:
        bindings = self.bindings
        world_idx_i32 = world_idx.to(dtype=torch.int32)
        pool_idx_i32 = common_reset_batch.pool_idx.to(dtype=torch.int32)

        bindings.qpos[world_idx] = bindings.base_qpos
        bindings.qvel[world_idx] = 0.0
        if bindings.ctrl.numel() > 0:
            bindings.ctrl[world_idx] = 0.0
        if bindings.qacc_warmstart.numel() > 0:
            bindings.qacc_warmstart[world_idx] = 0.0
        if bindings.act.numel() > 0:
            bindings.act[world_idx] = 0.0
        if bindings.eq_active.numel() > 0:
            bindings.eq_active[world_idx] = bindings.pool_eq_active[common_reset_batch.pool_idx]
        if bindings.mocap_pos.numel() > 0:
            bindings.mocap_pos[world_idx] = bindings.base_mocap_pos
        if bindings.mocap_quat.numel() > 0:
            bindings.mocap_quat[world_idx] = bindings.base_mocap_quat
        bindings.time[world_idx] = 0.0

        wp.launch(
            kernel=apply_reset_unit_pose,
            dim=(int(world_idx.numel()), int(bindings.units_active_mask.shape[1])),
            inputs=[
                wp.from_torch(world_idx_i32),
                wp.from_torch(pool_idx_i32),
                bindings.pool_active_mask_wp,
                bindings.pool_positions_wp,
                bindings.pool_quats_wp,
                bindings.inactive_unit_positions_wp,
                wp.from_torch(common_reset_batch.swarm_start, dtype=wp.vec3),
                wp.from_torch(common_reset_batch.initial_z_rotation, dtype=wp.float32),
                bindings.unit_qpos_adr_wp,
            ],
            outputs=[bindings.qpos_wp],
            device=bindings.wp_device,
        )

        bindings.units_active_mask[world_idx] = bindings.pool.active_mask[common_reset_batch.pool_idx]
        bindings.partner_unit[world_idx] = bindings.pool.partner_unit[common_reset_batch.pool_idx].clamp(min=-1)
        bindings.partner_connector[world_idx] = bindings.pool.partner_connector[common_reset_batch.pool_idx].clamp(min=-1)
        bindings.connection_twist_idx[world_idx] = bindings.pool.twist_idx[common_reset_batch.pool_idx]
        bindings.disconnect_potentials[world_idx] = 0.0
        bindings.current_step[world_idx] = 0

    def _build_common_reset_specs(self, *, common_reset_batch: MJWCommonResetBatch) -> list[MJWCommonResetSpec]:
        pool_idx = common_reset_batch.pool_idx.detach().cpu().numpy()
        swarm_start = common_reset_batch.swarm_start.detach().cpu().numpy()
        initial_z_rotation = common_reset_batch.initial_z_rotation.detach().cpu().numpy()
        return [
            MJWCommonResetSpec(
                pool_idx=int(pool_idx[i]),
                swarm_start=swarm_start[i].copy(),
                initial_z_rotation=float(initial_z_rotation[i]),
            )
            for i in range(pool_idx.shape[0])
        ]

    def _apply_common_settled_snapshot_batch(
        self,
        *,
        world_idx: torch.Tensor,
        snapshots: list[MJWCommonSettledSnapshot],
    ) -> None:
        bindings = self.bindings
        bindings.qpos[world_idx] = torch.as_tensor(
            np.stack([snapshot.qpos for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.qpos.dtype,
        )
        bindings.qvel[world_idx] = torch.as_tensor(
            np.stack([snapshot.qvel for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.qvel.dtype,
        )
        if bindings.ctrl.numel() > 0:
            bindings.ctrl[world_idx] = 0.0
        if bindings.qacc_warmstart.numel() > 0:
            bindings.qacc_warmstart[world_idx] = 0.0
        if bindings.act.numel() > 0:
            bindings.act[world_idx] = 0.0
        if bindings.eq_active.numel() > 0:
            bindings.eq_active[world_idx] = torch.as_tensor(
                np.stack([snapshot.eq_active for snapshot in snapshots]),
                device=bindings.device,
                dtype=bindings.eq_active.dtype,
            )
        if bindings.mocap_pos.numel() > 0:
            bindings.mocap_pos[world_idx] = torch.as_tensor(
                np.stack([snapshot.mocap_pos for snapshot in snapshots]),
                device=bindings.device,
                dtype=bindings.mocap_pos.dtype,
            )
        if bindings.mocap_quat.numel() > 0:
            bindings.mocap_quat[world_idx] = torch.as_tensor(
                np.stack([snapshot.mocap_quat for snapshot in snapshots]),
                device=bindings.device,
                dtype=bindings.mocap_quat.dtype,
            )
        bindings.time[world_idx] = torch.as_tensor(
            [snapshot.time for snapshot in snapshots],
            device=bindings.device,
            dtype=bindings.time.dtype,
        )
        bindings.units_active_mask[world_idx] = torch.as_tensor(
            np.stack([snapshot.units_active_mask for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.units_active_mask.dtype,
        )
        bindings.partner_unit[world_idx] = torch.as_tensor(
            np.stack([snapshot.partner_unit for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.partner_unit.dtype,
        )
        bindings.partner_connector[world_idx] = torch.as_tensor(
            np.stack([snapshot.partner_connector for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.partner_connector.dtype,
        )
        bindings.connection_twist_idx[world_idx] = torch.as_tensor(
            np.stack([snapshot.connection_twist_idx for snapshot in snapshots]),
            device=bindings.device,
            dtype=bindings.connection_twist_idx.dtype,
        )
        bindings.disconnect_potentials[world_idx] = 0.0
        bindings.current_step[world_idx] = 0

    def _sample_swarm_start(self, *, n_reset: int, rng: torch.Generator) -> torch.Tensor:
        swarm_start_x = sample_float_or_dist(
            self.scenario.swarm_start_x,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        swarm_start_y = sample_float_or_dist(
            self.scenario.swarm_start_y,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        return torch.stack(
            (
                swarm_start_x,
                swarm_start_y,
                torch.full(
                    (n_reset,),
                    self.scenario.swarm.max_unit_extent * 1.1,
                    device=self.bindings.device,
                    dtype=torch.float32,
                ),
            ),
            dim=-1,
        )

    def _sample_initial_z_rotation(self, *, n_reset: int, rng: torch.Generator) -> torch.Tensor:
        if not self.scenario.randomize_initial_swarm_z_rotation:
            return torch.zeros((n_reset,), device=self.bindings.device, dtype=torch.float32)
        return torch.rand((n_reset,), device=self.bindings.device, dtype=torch.float32, generator=rng) * (2.0 * math.pi)


class BaseMJWCPUResetSettler(abc.ABC):
    def __init__(self, *, scenario: BaseMJWScenario, bindings: MJWRuntimeBindings) -> None:
        self.scenario = scenario
        self.bindings = bindings
        self.model = scenario.build_model()
        self.data = mujoco.MjData(self.model)

        self._base_qpos = self.data.qpos.copy()
        self._base_mocap_pos = self.data.mocap_pos.copy()
        self._base_mocap_quat = self.data.mocap_quat.copy()
        self._pool_positions = bindings.pool.positions.detach().cpu().numpy()
        self._pool_quats = bindings.pool.quats.detach().cpu().numpy()
        self._pool_active_mask = bindings.pool.active_mask.detach().cpu().numpy()
        self._pool_partner_unit = bindings.pool.partner_unit.detach().cpu().numpy().clip(min=-1)
        self._pool_partner_connector = bindings.pool.partner_connector.detach().cpu().numpy().clip(min=-1)
        self._pool_twist_idx = bindings.pool.twist_idx.detach().cpu().numpy()
        self._pool_eq_active = bindings.pool_eq_active.detach().cpu().numpy()
        self._inactive_unit_positions = bindings.inactive_unit_positions.detach().cpu().numpy()

    def _apply_common_reset(self, *, common_reset_spec: MJWCommonResetSpec) -> None:
        mujoco.mj_resetData(self.model, self.data)
        data = self.data
        data.qpos[:] = self._base_qpos
        data.qvel[:] = 0.0
        if data.ctrl.size > 0:
            data.ctrl[:] = 0.0
        if data.qacc_warmstart.size > 0:
            data.qacc_warmstart[:] = 0.0
        if data.act.size > 0:
            data.act[:] = 0.0
        if data.eq_active.size > 0:
            data.eq_active[:] = self._pool_eq_active[common_reset_spec.pool_idx]
        if data.mocap_pos.size > 0:
            data.mocap_pos[:] = self._base_mocap_pos
        if data.mocap_quat.size > 0:
            data.mocap_quat[:] = self._base_mocap_quat
        data.time = 0.0

        active_mask = self._pool_active_mask[common_reset_spec.pool_idx]
        angle = float(common_reset_spec.initial_z_rotation)
        rotate_swarm = angle != 0.0
        if rotate_swarm:
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)
            yaw_quat = np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)], dtype=np.float64)
        for unit_idx, qpos_adr in enumerate(self.bindings.metadata.unit_qpos_adr):
            if active_mask[unit_idx]:
                local_pos = self._pool_positions[common_reset_spec.pool_idx, unit_idx]
                if rotate_swarm:
                    data.qpos[qpos_adr] = common_reset_spec.swarm_start[0] + cos_angle * local_pos[0] - sin_angle * local_pos[1]
                    data.qpos[qpos_adr + 1] = common_reset_spec.swarm_start[1] + sin_angle * local_pos[0] + cos_angle * local_pos[1]
                    data.qpos[qpos_adr + 2] = common_reset_spec.swarm_start[2] + local_pos[2]
                    rotated_quat = np.empty(4, dtype=np.float64)
                    mujoco.mju_mulQuat(
                        rotated_quat,
                        yaw_quat,
                        self._pool_quats[common_reset_spec.pool_idx, unit_idx].astype(np.float64),
                    )
                    data.qpos[qpos_adr + 3 : qpos_adr + 7] = rotated_quat
                else:
                    data.qpos[qpos_adr : qpos_adr + 3] = local_pos + common_reset_spec.swarm_start
                    data.qpos[qpos_adr + 3 : qpos_adr + 7] = self._pool_quats[common_reset_spec.pool_idx, unit_idx]
            else:
                data.qpos[qpos_adr : qpos_adr + 3] = self._inactive_unit_positions[unit_idx]
                data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _build_common_snapshot(self, *, common_reset_spec: MJWCommonResetSpec) -> MJWCommonSettledSnapshot:
        return MJWCommonSettledSnapshot(
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
            eq_active=self.data.eq_active.copy(),
            mocap_pos=self.data.mocap_pos.copy(),
            mocap_quat=self.data.mocap_quat.copy(),
            time=float(self.data.time),
            units_active_mask=self._pool_active_mask[common_reset_spec.pool_idx].copy(),
            partner_unit=self._pool_partner_unit[common_reset_spec.pool_idx].copy(),
            partner_connector=self._pool_partner_connector[common_reset_spec.pool_idx].copy(),
            connection_twist_idx=self._pool_twist_idx[common_reset_spec.pool_idx].copy(),
        )

    def _settle_physics(self) -> None:
        if self.scenario.reset_settle_time <= 0:
            return
        remaining_time = self.scenario.reset_settle_time - self.data.time
        if remaining_time <= 0:
            return
        original_timestep = float(self.model.opt.timestep)
        effective_timestep = original_timestep * float(self.scenario.reset_settle_timestep_scale)
        nstep = math.ceil(remaining_time / effective_timestep)
        if self.scenario.reset_settle_timestep_scale != 1.0:
            self.model.opt.timestep = effective_timestep
        try:
            mujoco.mj_step(self.model, self.data, nstep=int(nstep))
        finally:
            if float(self.model.opt.timestep) != original_timestep:
                self.model.opt.timestep = original_timestep
