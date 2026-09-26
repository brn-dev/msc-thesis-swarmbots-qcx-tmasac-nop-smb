from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium.vector import AutoresetMode, VectorEnv
from loguru import logger

from swarmbots.mjw_env.mjw_env_tensor_ops import (
    MJWActionLayout,
    MJWObservationLayout,
    build_mjw_env_tensor_operations,
    should_compile_mjw_env_tensor_operations_by_default,
)
from swarmbots.mjw_env.mjw_kernels import (
    apply_connection_candidates,
    compute_best_connection_candidates,
    gather_connector_frames,
    update_binary_connector_disconnections,
    update_continuous_connector_disconnections,
)
from swarmbots.mjw_env.mjw_live_recording import MJWLiveEpisodeRecorder, MJWRecordingConfig, MJWWorldSnapshot
from swarmbots.mjw_env.mjw_model_metadata import MJWModelMetadata, build_model_metadata
from swarmbots.mjw_env.mjw_torch_utils import to_device_bool_tensor
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRuntimeBindings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWSwarmPool
from swarmbots.learn.discord_notifications import (
    notify_mjw_nefc_overflow_once,
    notify_mjw_simulation_instability_once,
)
from swarmbots.learn.tensor_conversion import to_numpy_array
from swarmbots.learn.torch_device import as_device as _resolve_torch_device
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH


def _build_inactive_unit_positions(
    *,
    num_units: int,
    max_unit_extent: float,
    inactive_area_location: tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    num_units_sqrt = int(math.ceil(math.sqrt(num_units)))
    unit_spacing = max_unit_extent * 3.0
    positions = torch.zeros((num_units, 3), device=device, dtype=torch.float32)
    positions[:, 2] = max_unit_extent
    for unit_idx in range(num_units):
        positions[unit_idx, 0] = (unit_idx // num_units_sqrt) * unit_spacing
        positions[unit_idx, 1] = (unit_idx % num_units_sqrt) * unit_spacing
    positions[:, :2] -= positions[:, :2].mean(dim=0, keepdim=True)
    positions += torch.tensor(inactive_area_location, device=device, dtype=torch.float32)
    return positions


def _mask_info_values(info_value: Any, unstable_mask: torch.Tensor) -> None:
    if isinstance(info_value, dict):
        for nested_value in info_value.values():
            _mask_info_values(nested_value, unstable_mask)
        return
    info_value[unstable_mask] = 0.0


@dataclass(slots=True)
class _PendingMJWStep:
    obs: dict[str, torch.Tensor]
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    infos: dict[str, Any]
    dones: torch.Tensor
    check_nefc_overflow: bool


def _capture_step_graph(model: Any, data: Any, nstep: int) -> Any | None:
    if nstep <= 0:
        return None
    with wp.ScopedCapture() as capture:
        for _ in range(nstep):
            mjw.step(model, data)
    return capture.graph


def _default_nconmax(*, num_units: int, num_total_connectors: int) -> int:
    return max(32, num_total_connectors + (2 * num_units))


def _default_njmax(*, num_units: int, nconmax: int) -> int:
    return max(160, (5 * nconmax) + (10 * num_units))


def _resolve_workspace_cap(
    *,
    explicit_value: int | None,
    scenario_value: int | None,
    fallback_value: int,
) -> int:
    if explicit_value is not None:
        return int(explicit_value)
    if scenario_value is not None:
        return int(scenario_value)
    return int(fallback_value)


class _SettledResetSnapshotBuffer:
    def __init__(
        self,
        *,
        scenario_runtime: Any,
        rng: torch.Generator,
        executor: ThreadPoolExecutor,
        capacity: int,
        batch_size: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"Expected capacity > 0, got {capacity}")
        if batch_size <= 0:
            raise ValueError(f"Expected batch_size > 0, got {batch_size}")

        self._scenario_runtime = scenario_runtime
        self._rng = rng
        self._executor = executor
        self._capacity = int(capacity)
        self._batch_size = int(batch_size)
        self._snapshots: deque[Any] = deque()
        self._future: Future[list[Any]] | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def ready_count(self) -> int:
        self.drain_ready()
        return len(self._snapshots)

    def drain_ready(self) -> None:
        future = self._future
        if future is None or not future.done():
            return
        self._snapshots.extend(future.result())
        self._future = None

    def fill_async(self) -> None:
        if self._future is not None or len(self._snapshots) >= self._capacity:
            return

        n_reset = min(self._batch_size, self._capacity - len(self._snapshots))
        reset_batch = self._scenario_runtime.sample_reset_batch(n_reset=n_reset, rng=self._rng)
        specs = self._scenario_runtime.build_cpu_reset_specs(reset_batch=reset_batch)
        self._future = self._executor.submit(self._scenario_runtime.settle_cpu_reset_specs, specs=specs)

    def take(self, n_snapshots: int) -> list[Any]:
        if n_snapshots <= 0:
            return []

        self.drain_ready()
        if len(self._snapshots) < n_snapshots and self._future is not None:
            self._snapshots.extend(self._future.result())
            self._future = None

        if len(self._snapshots) < n_snapshots:
            missing = n_snapshots - len(self._snapshots)
            reset_batch = self._scenario_runtime.sample_reset_batch(n_reset=missing, rng=self._rng)
            specs = self._scenario_runtime.build_cpu_reset_specs(reset_batch=reset_batch)
            self._snapshots.extend(self._scenario_runtime.settle_cpu_reset_specs(specs=specs))

        snapshots = [self._snapshots.popleft() for _ in range(n_snapshots)]
        self.fill_async()
        return snapshots

    def clear(self, *, wait: bool) -> None:
        self._snapshots.clear()
        future = self._future
        if future is None:
            return
        if wait or not future.cancel():
            future.result()
        self._future = None


class MJWSwarmBotsVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}
    supports_per_env_reset_seeds = False
    _pending_step: _PendingMJWStep | None = None

    def __init__(
        self,
        scenario: BaseMJWScenario,
        *,
        num_envs: int,
        episode_length: int = 500,
        first_episode_length: int | None = None,
        first_episode_lengths: list[int] | torch.Tensor | None = None,
        settle_initial_reset: bool = False,
        settled_reset_buffer_size: int | None = None,
        settled_reset_batch_size: int | None = None,
        simulation_unstable_reward: float = -1.0,
        device: str | torch.device = "cuda",
        nconmax: int | None = None,
        njmax: int | None = None,
        ccd_iterations: int | None = None,
        nefc_overflow_check_interval_steps: int = 128,
        compile_tensor_operations: bool | None = None,
        tensor_operations_compile_mode: str = "default",
    ) -> None:
        super().__init__()
        wp.init()

        if first_episode_length is not None and first_episode_lengths is not None:
            raise ValueError("Pass only one of first_episode_length or first_episode_lengths")
        if first_episode_length is not None and first_episode_length > episode_length:
            raise ValueError(
                f"first_episode_length can not be longer than episode_length ({episode_length}), got {first_episode_length}"
            )
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}")
        if nefc_overflow_check_interval_steps <= 0:
            raise ValueError(
                "nefc_overflow_check_interval_steps must be > 0, "
                f"got {nefc_overflow_check_interval_steps}"
            )
        if ccd_iterations is not None and ccd_iterations <= 0:
            raise ValueError(f"Expected ccd_iterations > 0, got {ccd_iterations}")
        if settled_reset_buffer_size is not None and settled_reset_buffer_size < 0:
            raise ValueError(f"Expected settled_reset_buffer_size >= 0, got {settled_reset_buffer_size}")
        if settled_reset_batch_size is not None and settled_reset_batch_size <= 0:
            raise ValueError(f"Expected settled_reset_batch_size > 0, got {settled_reset_batch_size}")

        self.device = _resolve_torch_device(device)
        self._wp_device = wp.device_from_torch(self.device)
        self.scenario = scenario
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.first_episode_length = first_episode_length
        self.first_episode_lengths = first_episode_lengths
        self.settle_initial_reset = bool(settle_initial_reset)
        self.settled_reset_buffer_size = settled_reset_buffer_size
        self.settled_reset_batch_size = settled_reset_batch_size
        self.action_repeat = int(scenario.action_repeat)
        self.simulation_unstable_reward = float(simulation_unstable_reward)
        self.render_mode = None
        self.action_backend = "torch"
        self._continuous_connector_actions = bool(getattr(scenario, "continuous_connector_actions", False))
        self._nefc_overflow_check_interval_steps = int(nefc_overflow_check_interval_steps)
        self.compile_tensor_operations = (
            should_compile_mjw_env_tensor_operations_by_default(self.device)
            if compile_tensor_operations is None
            else bool(compile_tensor_operations)
        )
        self.tensor_operations_compile_mode = str(tensor_operations_compile_mode)

        self.single_observation_space = scenario.get_single_observation_space()
        self.single_action_space = scenario.get_single_action_space()
        self.observation_space = scenario.get_batched_observation_space(self.num_envs)
        self.action_space = scenario.get_batched_action_space(self.num_envs)

        self._n_agents = scenario.swarm.num_units
        self._n_connectors = scenario.swarm.config.limbs_per_unit
        self._n_total_connectors = self._n_agents * self._n_connectors
        self._n_actuators = int(self.single_action_space["actuators"].shape[1])
        self._n_twists = len(self.scenario.swarm.config.connection_twist_values)

        self._host_model = scenario.build_model()
        if ccd_iterations is not None:
            self._host_model.opt.ccd_iterations = int(ccd_iterations)
        self._ccd_iterations = int(self._host_model.opt.ccd_iterations)
        self._metadata: MJWModelMetadata = build_model_metadata(self._host_model, scenario)
        self._scenario_runtime_metadata = self.scenario.build_runtime_metadata(host_model=self._host_model)

        scenario_nconmax = getattr(scenario, "physics_nconmax", None)
        scenario_njmax = getattr(scenario, "physics_njmax", None)
        default_nconmax = _default_nconmax(
            num_units=self._n_agents,
            num_total_connectors=self._n_total_connectors,
        )
        resolved_nconmax = _resolve_workspace_cap(
            explicit_value=nconmax,
            scenario_value=scenario_nconmax,
            fallback_value=default_nconmax,
        )
        default_njmax = _default_njmax(
            num_units=self._n_agents,
            nconmax=resolved_nconmax,
        )
        resolved_njmax = _resolve_workspace_cap(
            explicit_value=njmax,
            scenario_value=scenario_njmax,
            fallback_value=default_njmax,
        )
        if resolved_nconmax <= 0:
            raise ValueError(f"Expected nconmax > 0, got {resolved_nconmax}")
        if resolved_njmax <= 0:
            raise ValueError(f"Expected njmax > 0, got {resolved_njmax}")
        self._nconmax = resolved_nconmax
        self._njmax = resolved_njmax
        self._model = mjw.put_model(self._host_model)
        self._data = mjw.make_data(
            self._host_model,
            nworld=self.num_envs,
            nconmax=self._nconmax,
            njmax=self._njmax,
        )

        self._qpos = wp.to_torch(self._data.qpos)
        self._qvel = wp.to_torch(self._data.qvel)
        self._ctrl = wp.to_torch(self._data.ctrl)
        self._eq_active = wp.to_torch(self._data.eq_active)
        self._mocap_pos = wp.to_torch(self._data.mocap_pos)
        self._mocap_quat = wp.to_torch(self._data.mocap_quat)
        self._xpos = wp.to_torch(self._data.xpos)
        self._xquat = wp.to_torch(self._data.xquat)
        self._xmat = wp.to_torch(self._data.xmat)
        self._time = wp.to_torch(self._data.time)
        self._nefc = wp.to_torch(self._data.nefc)
        self._qacc_warmstart = wp.to_torch(self._data.qacc_warmstart)
        self._act = wp.to_torch(self._data.act)

        self._base_qpos = self._qpos[0].clone()
        self._base_mocap_pos = self._mocap_pos[0].clone()
        self._base_mocap_quat = self._mocap_quat[0].clone()

        self._qpos_flat_indices = torch.as_tensor(self._metadata.qpos_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._qvel_flat_indices = torch.as_tensor(self._metadata.qvel_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._ctrl_flat_indices = torch.as_tensor(self._metadata.ctrl_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._qpos_indices = torch.as_tensor(self._metadata.qpos_indices, device=self.device, dtype=torch.long)
        self._qvel_indices = torch.as_tensor(self._metadata.qvel_indices, device=self.device, dtype=torch.long)
        self._connector_body_indices = torch.as_tensor(
            self._metadata.connector_body_indices.reshape(-1),
            device=self.device,
            dtype=torch.long,
        )
        self._connector_body_indices_i32 = self._connector_body_indices.to(dtype=torch.int32)
        self._eq_indices = torch.as_tensor(self._metadata.eq_indices, device=self.device, dtype=torch.long)
        self._unit_qpos_adr = torch.as_tensor(self._metadata.unit_qpos_adr, device=self.device, dtype=torch.long)
        self._unit_qpos_adr_i32 = self._unit_qpos_adr.to(dtype=torch.int32)
        self._unit_dof_adr = torch.as_tensor(self._metadata.unit_dof_adr, device=self.device, dtype=torch.long)
        self._connector_unit_idx = torch.arange(self._n_agents, device=self.device, dtype=torch.long).repeat_interleave(
            self._n_connectors
        )
        self._connector_unit_idx_i32 = self._connector_unit_idx.to(dtype=torch.int32)
        self._unit_indices = torch.arange(self._n_agents, device=self.device, dtype=torch.long).view(1, -1, 1)
        self._connector_indices = torch.arange(self._n_connectors, device=self.device, dtype=torch.long).view(1, 1, -1)
        self._twist_values = torch.as_tensor(scenario.swarm.config.connection_twist_values, device=self.device, dtype=torch.float32)
        self._distance_threshold_sq = float(self.scenario.connection_dist_threshold) ** 2
        self._twist_step = 2.0 * math.pi / self._n_twists

        self._pool: MJWSwarmPool = scenario.swarm.build_pool(device=self.device)
        self._pool_eq_active = self._build_pool_eq_active()

        self._inactive_unit_positions = _build_inactive_unit_positions(
            num_units=self._n_agents,
            max_unit_extent=scenario.swarm.max_unit_extent,
            inactive_area_location=tuple(float(v) for v in scenario.inactive_area_location),
            device=self.device,
        )

        self.units_active_mask = torch.zeros((self.num_envs, self._n_agents), device=self.device, dtype=torch.bool)
        self.partner_unit = torch.full((self.num_envs, self._n_agents, self._n_connectors), -1, device=self.device, dtype=torch.long)
        self.partner_connector = torch.full_like(self.partner_unit, -1)
        self.connection_twist_idx = torch.full_like(self.partner_unit, -1)
        self.disconnect_potentials = torch.zeros((self.num_envs, self._n_agents, self._n_connectors), device=self.device, dtype=torch.float32)
        self.successful_connection_counts = torch.zeros(
            (self.num_envs, self._n_agents),
            device=self.device,
            dtype=torch.int64,
        )
        self._connection_active_before_update = torch.empty_like(self.partner_unit, dtype=torch.bool)
        self._connection_active_after_update = torch.empty_like(self.partner_unit, dtype=torch.bool)
        self._successful_connection_count_update = torch.empty_like(self.successful_connection_counts)

        self.current_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int64)
        self.is_first_episode = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self._best_partner_idx = torch.full((self.num_envs, self._n_total_connectors), -1, device=self.device, dtype=torch.int32)
        self._best_twist_idx = torch.full_like(self._best_partner_idx, -1)
        self._connector_positions = torch.empty(
            (self.num_envs, self._n_total_connectors, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._connector_x_axis = torch.empty_like(self._connector_positions)
        self._connector_y_axis = torch.empty_like(self._connector_positions)
        self._connector_z_axis = torch.empty_like(self._connector_positions)
        self._disconnect_update = torch.empty_like(self.disconnect_potentials)
        self._local_obs = torch.empty(
            (self.num_envs, *self.single_observation_space["local_obs"].shape),
            device=self.device,
            dtype=torch.float32,
        )
        self._episode_length_limit = torch.full((self.num_envs,), self.episode_length, device=self.device, dtype=torch.int64)
        self._first_episode_length_limit = None
        if self.first_episode_lengths is not None:
            first_episode_length_limit = torch.as_tensor(
                self.first_episode_lengths,
                device=self.device,
                dtype=torch.int64,
            ).reshape(-1)
            if tuple(first_episode_length_limit.shape) != (self.num_envs,):
                raise ValueError(
                    f"Expected first_episode_lengths shape ({self.num_envs},), got {tuple(first_episode_length_limit.shape)}"
                )
            if torch.any(first_episode_length_limit < 0):
                raise ValueError("first_episode_lengths must be >= 0")
            if torch.any(first_episode_length_limit > self.episode_length):
                raise ValueError(
                    f"first_episode_lengths can not be longer than episode_length ({self.episode_length})"
                )
            self._first_episode_length_limit = first_episode_length_limit
        elif self.first_episode_length is not None:
            self._first_episode_length_limit = torch.full(
                (self.num_envs,),
                int(self.first_episode_length),
                device=self.device,
                dtype=torch.int64,
            )

        self._physics_graph = _capture_step_graph(self._model, self._data, self.action_repeat) if self.device.type == "cuda" else None
        if self._physics_graph is None and self.device.type == "cuda":
            logger.warning("Physics CUDA graph could not be captured")

        free_joint_rot_dim = 6 if self.scenario.quat_rot6d_representation else 4
        num_hinges = self._qvel_indices.shape[1] - 6
        self._free_joint_pos_slice = slice(0, 3)
        self._free_joint_rot_slice = slice(3, 3 + free_joint_rot_dim)
        self._hinge_obs_slice = slice(self._free_joint_rot_slice.stop, self._free_joint_rot_slice.stop + (2 * num_hinges))
        self._qvel_obs_slice = slice(self._hinge_obs_slice.stop, self._hinge_obs_slice.stop + self._qvel_indices.shape[1])
        self._connector_obs_slice = slice(self._qvel_obs_slice.stop, self._qvel_obs_slice.stop + (self._n_connectors * 5))
        self._connector_xpos_obs_slice = slice(
            self._connector_obs_slice.stop,
            self._connector_obs_slice.stop + (self._n_connectors * 3),
        )
        self._tensor_operations = build_mjw_env_tensor_operations(
            observation_layout=MJWObservationLayout(
                num_envs=self.num_envs,
                num_agents=self._n_agents,
                num_connectors=self._n_connectors,
                free_joint_position=self._free_joint_pos_slice,
                free_joint_rotation=self._free_joint_rot_slice,
                hinge=self._hinge_obs_slice,
                qvel=self._qvel_obs_slice,
                connector=self._connector_obs_slice,
                connector_position=self._connector_xpos_obs_slice,
                use_rot6d=bool(self.scenario.quat_rot6d_representation),
                include_connector_positions=bool(self.scenario.include_connectors_xpos_in_obs),
            ),
            action_layout=MJWActionLayout(
                num_envs=self.num_envs,
                continuous_connectors=self._continuous_connector_actions,
            ),
            compile_operations=self.compile_tensor_operations,
            compile_mode=self.tensor_operations_compile_mode,
        )

        self._qpos_wp = wp.from_torch(self._qpos)
        self._xpos_wp = wp.from_torch(self._xpos, dtype=wp.vec3)
        self._xmat_wp = wp.from_torch(self._xmat)
        self._pool_active_mask_wp = wp.from_torch(self._pool.active_mask)
        self._pool_positions_wp = wp.from_torch(self._pool.positions, dtype=wp.vec3)
        self._pool_quats_wp = wp.from_torch(self._pool.quats)
        self._inactive_unit_positions_wp = wp.from_torch(self._inactive_unit_positions, dtype=wp.vec3)
        self._unit_qpos_adr_wp = wp.from_torch(self._unit_qpos_adr_i32)
        self._connector_body_indices_wp = wp.from_torch(self._connector_body_indices_i32)
        self._connector_unit_idx_wp = wp.from_torch(self._connector_unit_idx_i32)
        self._best_partner_idx_wp = wp.from_torch(self._best_partner_idx)
        self._best_twist_idx_wp = wp.from_torch(self._best_twist_idx)
        self._connector_positions_wp = wp.from_torch(self._connector_positions, dtype=wp.vec3)
        self._connector_x_axis_wp = wp.from_torch(self._connector_x_axis, dtype=wp.vec3)
        self._connector_y_axis_wp = wp.from_torch(self._connector_y_axis, dtype=wp.vec3)
        self._connector_z_axis_wp = wp.from_torch(self._connector_z_axis, dtype=wp.vec3)
        self._eq_indices_wp = wp.from_torch(self._eq_indices.reshape(-1))
        self._partner_unit_wp = wp.from_torch(self.partner_unit)
        self._partner_connector_wp = wp.from_torch(self.partner_connector)
        self._connection_twist_idx_wp = wp.from_torch(self.connection_twist_idx)
        self._disconnect_potentials_wp = wp.from_torch(self.disconnect_potentials)
        self._eq_active_wp = wp.from_torch(self._eq_active)

        self._runtime_bindings = MJWRuntimeBindings(
            device=self.device,
            wp_device=self._wp_device,
            num_envs=self.num_envs,
            model=self._model,
            data=self._data,
            metadata=self._metadata,
            pool=self._pool,
            pool_eq_active=self._pool_eq_active,
            inactive_unit_positions=self._inactive_unit_positions,
            qpos=self._qpos,
            qvel=self._qvel,
            ctrl=self._ctrl,
            eq_active=self._eq_active,
            mocap_pos=self._mocap_pos,
            mocap_quat=self._mocap_quat,
            xpos=self._xpos,
            xquat=self._xquat,
            xmat=self._xmat,
            time=self._time,
            qacc_warmstart=self._qacc_warmstart,
            act=self._act,
            units_active_mask=self.units_active_mask,
            partner_unit=self.partner_unit,
            partner_connector=self.partner_connector,
            connection_twist_idx=self.connection_twist_idx,
            disconnect_potentials=self.disconnect_potentials,
            current_step=self.current_step,
            is_first_episode=self.is_first_episode,
            base_qpos=self._base_qpos,
            base_mocap_pos=self._base_mocap_pos,
            base_mocap_quat=self._base_mocap_quat,
            unit_qpos_adr=self._unit_qpos_adr,
            qpos_wp=self._qpos_wp,
            pool_active_mask_wp=self._pool_active_mask_wp,
            pool_positions_wp=self._pool_positions_wp,
            pool_quats_wp=self._pool_quats_wp,
            inactive_unit_positions_wp=self._inactive_unit_positions_wp,
            unit_qpos_adr_wp=self._unit_qpos_adr_wp,
        )
        self._scenario_runtime = self.scenario.create_runtime(
            bindings=self._runtime_bindings,
            runtime_metadata=self._scenario_runtime_metadata,
        )

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(42 if scenario.seed is None else int(scenario.seed))
        self._use_settled_resets = float(self.scenario.reset_settle_time) > 0.0
        self._settle_executor: ThreadPoolExecutor | None = None
        self._settled_reset_buffer: _SettledResetSnapshotBuffer | None = None
        self._live_episode_recorder = MJWLiveEpisodeRecorder(scenario=self.scenario)
        self._initial_settled_reset_done = False
        self._nefc_overflow_notification_checked = False
        self._max_nefc_since_overflow_check = torch.zeros((), device=self.device, dtype=self._nefc.dtype)
        self._steps_since_nefc_overflow_check = 0
        self._pending_step: _PendingMJWStep | None = None
        self._step_status_host = torch.empty(
            self.num_envs + 1,
            device="cpu",
            dtype=torch.bool,
            pin_memory=self.device.type == "cuda",
        )
        self._step_status_ready = (
            torch.cuda.Event()
            if self.device.type == "cuda"
            else None
        )
        self._nefc_overflow_host = torch.empty(
            (),
            device="cpu",
            dtype=self._max_nefc_since_overflow_check.dtype,
            pin_memory=self.device.type == "cuda",
        )
        if self._use_settled_resets:
            resolved_buffer_size = (
                self.num_envs // 32
                if self.settled_reset_buffer_size is None
                else int(self.settled_reset_buffer_size)
            )
            if resolved_buffer_size > 0:
                self._settle_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mjw-reset-settle")
                resolved_batch_size = (
                    resolved_buffer_size
                    if self.settled_reset_batch_size is None
                    else min(int(self.settled_reset_batch_size), resolved_buffer_size)
                )
                self._settled_reset_buffer = _SettledResetSnapshotBuffer(
                    scenario_runtime=self._scenario_runtime,
                    rng=self._rng,
                    executor=self._settle_executor,
                    capacity=resolved_buffer_size,
                    batch_size=resolved_batch_size,
                )

    def get_settings(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.get_settings(),
            "episode_length": self.episode_length,
            "action_repeat": self.action_repeat,
            "simulation_unstable_reward": self.simulation_unstable_reward,
            "compile_tensor_operations": self.compile_tensor_operations,
            "tensor_operations_compile_mode": self.tensor_operations_compile_mode,
            "swarm_pool": {
                "pool_size": self.get_swarm_pool_size(),
                "active_pool_size": self.get_active_swarm_pool_size(),
            },
            "physics_workspace_caps": {
                "nconmax": self._nconmax,
                "njmax": self._njmax,
            },
            "physics_options": {
                "ccd_iterations": self._ccd_iterations,
            },
            "settled_reset_buffer": (
                {
                    "capacity": self._settled_reset_buffer.capacity,
                    "batch_size": self._settled_reset_buffer.batch_size,
                    "ready": self._settled_reset_buffer.ready_count(),
                }
                if self._settled_reset_buffer is not None
                else None
            ),
        }

    def get_swarm_pool_size(self) -> int:
        return self.scenario.swarm.get_pool_size()

    def get_active_swarm_pool_size(self) -> int:
        return self.scenario.swarm.get_active_pool_size()

    def set_active_swarm_pool_size(self, active_pool_size: int) -> int:
        return int(self.scenario.swarm.set_active_pool_size(active_pool_size))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if self._pending_step is not None:
            raise RuntimeError("reset() called while an MJW step is pending.")
        if seed is not None:
            if self._settled_reset_buffer is not None:
                self._settled_reset_buffer.clear(wait=True)
            self._rng.manual_seed(int(seed))
        reset_mask = (
            torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            if options is None or "reset_mask" not in options
            else to_device_bool_tensor(options["reset_mask"], device=self.device, expected_shape=(self.num_envs,))
        )
        force_settled = bool(options is not None and options.get("force_settled", False))
        if force_settled and not bool(torch.all(reset_mask)):
            raise ValueError("force_settled requires resetting every MJW environment.")
        if force_settled and self._use_settled_resets:
            self._reset_worlds_with_settled_snapshots(reset_mask)
            self._initial_settled_reset_done = True
        elif self._should_use_initial_settled_reset(reset_mask):
            self._reset_worlds_with_settled_snapshots(reset_mask)
            self._initial_settled_reset_done = True
        else:
            self._reset_worlds(reset_mask)
        self.successful_connection_counts[reset_mask] = 0
        if self._live_episode_recorder.is_active():
            reset_world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
            self._live_episode_recorder.on_episode_starts(
                world_idx=reset_world_idx.detach().cpu().numpy(),
                snapshots_by_world=self._capture_world_snapshots(reset_world_idx),
            )
        self._refill_settled_reset_buffer()
        return self._build_obs(), {}

    def close(self) -> None:
        if self._pending_step is not None:
            self.end_step()
        if hasattr(self, "_max_nefc_since_overflow_check"):
            self._maybe_notify_nefc_overflow()
        self._live_episode_recorder.close()
        if self._settled_reset_buffer is not None:
            self._settled_reset_buffer.clear(wait=False)
            self._settled_reset_buffer = None
        if self._settle_executor is not None:
            self._settle_executor.shutdown(wait=True, cancel_futures=True)
            self._settle_executor = None
        return None

    def step(
        self,
        actions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self.begin_step(actions)
        return self.end_step()

    @property
    def supports_deferred_step(self) -> bool:
        return self.device.type == "cuda"

    @property
    def has_pending_step(self) -> bool:
        return self._pending_step is not None

    def begin_step(self, actions: dict[str, torch.Tensor]) -> None:
        if self._pending_step is not None:
            raise RuntimeError("begin_step() called while another MJW step is pending.")

        actuators = torch.as_tensor(actions["actuators"], device=self.device, dtype=torch.float32)
        if self._continuous_connector_actions:
            connectors = torch.as_tensor(actions["connectors"], device=self.device, dtype=torch.float32)
        else:
            connectors = torch.as_tensor(actions["connectors"], device=self.device, dtype=torch.bool)
        if actuators.shape != (self.num_envs, self._n_agents, self._n_actuators):
            raise ValueError(f"Unexpected actuators shape {tuple(actuators.shape)}")
        if connectors.shape != (self.num_envs, self._n_agents, self._n_connectors):
            raise ValueError(f"Unexpected connectors shape {tuple(connectors.shape)}")

        trunc_limit = self._get_current_truncation_limit()
        self._refill_settled_reset_buffer()

        self._apply_actions(actuators=actuators, connectors=connectors)
        self._run_physics()
        self._update_max_nefc_since_overflow_check()
        self._steps_since_nefc_overflow_check += 1
        check_nefc_overflow = (
            self._steps_since_nefc_overflow_check
            >= self._nefc_overflow_check_interval_steps
        )

        unstable_mask, stable_mask = self._tensor_operations.begin_step(
            self._qpos,
            self._qvel,
            self.current_step,
        )

        step_result = self._scenario_runtime.compute_step_rewards(stable_mask=stable_mask)
        infos: dict[str, Any] = dict(step_result.info)
        scenario_terminations = (
            torch.zeros_like(unstable_mask)
            if step_result.terminations is None
            else step_result.terminations
        )
        rewards, terminations, truncations, dones = self._tensor_operations.finalize_step(
            step_result.reward,
            unstable_mask,
            stable_mask,
            scenario_terminations,
            self.current_step,
            trunc_limit,
            self.simulation_unstable_reward,
        )
        for value in infos.values():
            _mask_info_values(value, unstable_mask)

        obs = self._build_obs()
        obs = self._apply_error_obs(obs, unstable_mask)
        if self._live_episode_recorder.is_active():
            active_world_idx = self._live_episode_recorder.active_world_indices()
            if active_world_idx.size > 0:
                active_world_idx_t = torch.as_tensor(active_world_idx, device=self.device, dtype=torch.long)
                stable_active_world_idx = active_world_idx_t[stable_mask[active_world_idx_t]]
                self._live_episode_recorder.record_step(
                    rewards=rewards.detach().cpu().numpy(),
                    reward_terms=(
                        {
                            label: to_numpy_array(values)
                            for label, values in infos["reward_terms"].items()
                        }
                        if "reward_terms" in infos
                        else None
                    ),
                    dones=dones.detach().cpu().numpy(),
                    unstable_mask=unstable_mask.detach().cpu().numpy(),
                    snapshots_by_world=self._capture_world_snapshots(stable_active_world_idx),
                )

        step_status = torch.cat((dones.any().unsqueeze(0), unstable_mask))
        if self._step_status_ready is None:
            self._step_status_host.copy_(step_status)
            if check_nefc_overflow:
                self._nefc_overflow_host.copy_(self._max_nefc_since_overflow_check)
        else:
            self._step_status_host.copy_(step_status, non_blocking=True)
            if check_nefc_overflow:
                self._nefc_overflow_host.copy_(
                    self._max_nefc_since_overflow_check,
                    non_blocking=True,
                )
            self._step_status_ready.record(torch.cuda.current_stream(self.device))
        self._pending_step = _PendingMJWStep(
            obs=obs,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            dones=dones,
            check_nefc_overflow=check_nefc_overflow,
        )

    def end_step(
        self,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        pending_step = self._pending_step
        if pending_step is None:
            raise RuntimeError("end_step() called without a pending MJW step.")
        if self._step_status_ready is not None:
            self._step_status_ready.synchronize()
        self._pending_step = None

        obs = pending_step.obs
        rewards = pending_step.rewards
        terminations = pending_step.terminations
        truncations = pending_step.truncations
        infos = pending_step.infos
        dones = pending_step.dones
        step_status = self._step_status_host
        if pending_step.check_nefc_overflow:
            self._maybe_notify_nefc_overflow(
                required_njmax=int(self._nefc_overflow_host),
            )
        if bool(step_status[0]):
            self._inject_connection_episode_stats(infos=infos, dones=dones)
            unstable_world_indices = torch.nonzero(step_status[1:], as_tuple=True)[0].tolist()
            if unstable_world_indices:
                notify_mjw_simulation_instability_once(
                    scenario_name=type(self.scenario).__name__,
                    num_envs=self.num_envs,
                    unstable_world_indices=unstable_world_indices,
                )
            infos["final_obs"] = {key: value.clone() for key, value in obs.items()}
            infos["_final_obs"] = dones.clone()
            self.is_first_episode[dones] = False
            self._reset_done_worlds(dones)
            self.successful_connection_counts[dones] = 0
            if self._live_episode_recorder.is_active():
                done_world_idx = torch.nonzero(dones, as_tuple=False).flatten()
                self._live_episode_recorder.on_episode_starts(
                    world_idx=done_world_idx.detach().cpu().numpy(),
                    snapshots_by_world=self._capture_world_snapshots(done_world_idx),
                )
            obs = self._build_obs()

        return obs, rewards, terminations, truncations, infos

    def start_video_recording(
        self,
        *,
        video_folder: str,
        video_name_prefix: str,
        num_episodes: int = 5,
        max_parallel_episodes: int = 4,
        fps: int = 20,
        fps_mode: str = "compensate_stride",
        frame_stride: int = 1,
        width: int = DEFAULT_RECORDING_WIDTH,
        height: int = DEFAULT_RECORDING_HEIGHT,
        camera: int | str = -1,
    ) -> None:
        config = MJWRecordingConfig(
            video_folder=Path(video_folder),
            video_name_prefix=video_name_prefix,
            num_episodes=int(num_episodes),
            max_parallel_episodes=int(max_parallel_episodes),
            fps=int(fps),
            fps_mode=str(fps_mode),
            frame_stride=int(frame_stride),
            width=int(width),
            height=int(height),
            camera=camera,
        )
        episode_start_world_idx = torch.nonzero(self.current_step == 0, as_tuple=False).flatten()
        self._live_episode_recorder.start(
            config=config,
            episode_start_world_idx=episode_start_world_idx.detach().cpu().numpy(),
            snapshots_by_world=self._capture_world_snapshots(episode_start_world_idx),
        )

    def supports_live_recording(self) -> bool:
        return True

    def get_video_recording_status(self) -> dict[str, Any]:
        return self._live_episode_recorder.get_status()

    def _build_pool_eq_active(self) -> torch.Tensor:
        pool_eq_active = torch.zeros(
            (self._pool.size, int(self._host_model.neq)),
            device=self.device,
            dtype=self._eq_active.dtype if self._eq_active.numel() > 0 else torch.int32,
        )
        if pool_eq_active.numel() == 0:
            return pool_eq_active

        canonical_mask = self._pool.partner_unit >= 0
        canonical_mask &= self._connector_unit_idx.view(1, self._n_agents, self._n_connectors) < self._pool.partner_unit
        pool_idx, unit_idx, connector_idx = torch.nonzero(canonical_mask, as_tuple=True)
        if pool_idx.numel() == 0:
            return pool_eq_active

        other_unit = self._pool.partner_unit[pool_idx, unit_idx, connector_idx]
        other_connector = self._pool.partner_connector[pool_idx, unit_idx, connector_idx]
        twist_idx = self._pool.twist_idx[pool_idx, unit_idx, connector_idx]
        eq_idx = self._eq_indices[unit_idx, connector_idx, other_unit, other_connector, twist_idx]
        pool_eq_active[pool_idx, eq_idx] = 1
        return pool_eq_active

    def _run_physics(self) -> None:
        if self._physics_graph is not None:
            wp.capture_launch(self._physics_graph)
            return
        for _ in range(self.action_repeat):
            mjw.step(self._model, self._data)

    def _update_max_nefc_since_overflow_check(self) -> None:
        if self._nefc_overflow_notification_checked:
            return
        torch.maximum(
            self._max_nefc_since_overflow_check,
            self._nefc.max(),
            out=self._max_nefc_since_overflow_check,
        )

    def _maybe_notify_nefc_overflow(self, *, required_njmax: int | None = None) -> None:
        if self._nefc_overflow_notification_checked:
            return

        if required_njmax is None:
            required_njmax = int(self._max_nefc_since_overflow_check.item())
        self._steps_since_nefc_overflow_check = 0
        if required_njmax <= self._njmax:
            self._max_nefc_since_overflow_check.zero_()
            return

        self._nefc_overflow_notification_checked = True
        logger.warning(
            f"MJW nefc overflow detected: current njmax={self._njmax}, observed nefc={required_njmax}."
        )
        notify_mjw_nefc_overflow_once(
            scenario_name=type(self.scenario).__name__,
            num_envs=self.num_envs,
            nconmax=self._nconmax,
            njmax=self._njmax,
            required_njmax=required_njmax,
        )

    def _get_current_truncation_limit(self) -> torch.Tensor:
        if self._first_episode_length_limit is None:
            return self._episode_length_limit
        return torch.where(self.is_first_episode, self._first_episode_length_limit, self._episode_length_limit)

    def _refill_settled_reset_buffer(self) -> None:
        if self._settled_reset_buffer is None:
            return
        self._settled_reset_buffer.drain_ready()
        self._settled_reset_buffer.fill_async()

    def _reset_done_worlds(self, done_mask: torch.Tensor) -> None:
        if self._use_settled_resets:
            self._reset_world_indices_with_settled_snapshots(torch.nonzero(done_mask, as_tuple=False).flatten())
        else:
            self._reset_worlds(done_mask)

    def _reset_worlds(self, reset_mask: torch.Tensor) -> None:
        world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return
        self._scenario_runtime.apply_reset_batch(
            world_idx=world_idx,
            reset_batch=self._scenario_runtime.sample_reset_batch(n_reset=int(world_idx.numel()), rng=self._rng),
        )

    def _reset_worlds_with_settled_snapshots(self, reset_mask: torch.Tensor) -> None:
        world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return

        logger.info(f"Running initial settled reset for {int(world_idx.numel())} MJW envs.")
        self._reset_world_indices_with_settled_snapshots(world_idx)
        logger.info("Envs settled.")

    def _reset_world_indices_with_settled_snapshots(self, world_idx: torch.Tensor) -> None:
        if world_idx.numel() == 0:
            return

        if self._settled_reset_buffer is not None:
            snapshots = self._settled_reset_buffer.take(int(world_idx.numel()))
        else:
            reset_batch = self._scenario_runtime.sample_reset_batch(n_reset=int(world_idx.numel()), rng=self._rng)
            specs = self._scenario_runtime.build_cpu_reset_specs(reset_batch=reset_batch)
            snapshots = self._scenario_runtime.settle_cpu_reset_specs(specs=specs)
        self._scenario_runtime.apply_settled_reset_batch(world_idx=world_idx, snapshots=snapshots)

    def _should_use_initial_settled_reset(self, reset_mask: torch.Tensor) -> bool:
        if self._initial_settled_reset_done:
            return False
        if not self.settle_initial_reset:
            return False
        if not self._use_settled_resets:
            return False
        return bool(torch.all(reset_mask))

    def _apply_actions(self, *, actuators: torch.Tensor, connectors: torch.Tensor) -> None:
        connector_action = self._tensor_operations.prepare_actions(
            actuators,
            connectors,
            self.units_active_mask,
            self._ctrl,
            self._ctrl_flat_indices,
            float(self.scenario.actuator_strength),
        )
        if self._continuous_connector_actions:
            self._try_connect(connector_action > 0.0)
            self._disconnect_continuous(connector_action)
            return

        self._try_connect(connector_action)
        self._disconnect(connector_action)

    def _try_connect(self, connector_action: torch.Tensor) -> None:
        torch.ge(
            self.partner_unit,
            0,
            out=self._connection_active_before_update,
        )
        wp.launch(
            kernel=gather_connector_frames,
            dim=(self.num_envs, self._n_total_connectors),
            inputs=[
                self._xpos_wp,
                self._xmat_wp,
                self._connector_body_indices_wp,
            ],
            outputs=[
                self._connector_positions_wp,
                self._connector_x_axis_wp,
                self._connector_y_axis_wp,
                self._connector_z_axis_wp,
            ],
            device=self._wp_device,
        )

        wp.launch(
            kernel=compute_best_connection_candidates,
            dim=(self.num_envs, self._n_total_connectors),
            inputs=[
                wp.from_torch(connector_action),
                self._partner_unit_wp,
                self._connector_positions_wp,
                self._connector_x_axis_wp,
                self._connector_y_axis_wp,
                self._connector_z_axis_wp,
                self._connector_unit_idx_wp,
                self._n_total_connectors,
                self._n_connectors,
                self._distance_threshold_sq,
                float(self.scenario.connection_angle_threshold),
                self._twist_step,
                self._n_twists,
            ],
            outputs=[self._best_partner_idx_wp, self._best_twist_idx_wp],
            device=self._wp_device,
        )

        wp.launch(
            kernel=apply_connection_candidates,
            dim=(self.num_envs, self._n_total_connectors),
            inputs=[
                self._best_partner_idx_wp,
                self._best_twist_idx_wp,
                self._connector_unit_idx_wp,
                self._n_connectors,
                self._n_agents,
                self._n_twists,
                self._eq_indices_wp,
            ],
            outputs=[
                self._partner_unit_wp,
                self._partner_connector_wp,
                self._connection_twist_idx_wp,
                self._disconnect_potentials_wp,
                self._eq_active_wp,
            ],
            device=self._wp_device,
        )

        self._record_successful_connections()

    def _record_successful_connections(self) -> None:
        torch.logical_not(
            self._connection_active_before_update,
            out=self._connection_active_before_update,
        )
        torch.ge(
            self.partner_unit,
            0,
            out=self._connection_active_after_update,
        )
        torch.logical_and(
            self._connection_active_after_update,
            self._connection_active_before_update,
            out=self._connection_active_after_update,
        )
        torch.sum(
            self._connection_active_after_update,
            dim=-1,
            out=self._successful_connection_count_update,
        )
        self.successful_connection_counts.add_(self._successful_connection_count_update)

    def _inject_connection_episode_stats(
        self,
        *,
        infos: dict[str, Any],
        dones: torch.Tensor,
    ) -> None:
        counts = self.successful_connection_counts.to(dtype=torch.float32)
        active_mask = self.units_active_mask
        active_unit_count = active_mask.sum(dim=-1).clamp_min(1)
        mean = (counts * active_mask).sum(dim=-1) / active_unit_count
        squared_deviations = (counts - mean.unsqueeze(-1)).square() * active_mask
        std = torch.sqrt(squared_deviations.sum(dim=-1) / active_unit_count)
        minimum = counts.masked_fill(~active_mask, torch.inf).amin(dim=-1)
        maximum = counts.masked_fill(~active_mask, -torch.inf).amax(dim=-1)

        stats = infos.setdefault("episode", {})
        stats.update(
            {
                "successful_connections_per_unit_mean": torch.where(dones, mean, 0.0),
                "successful_connections_per_unit_std": torch.where(dones, std, 0.0),
                "successful_connections_per_unit_min": torch.where(dones, minimum, 0.0),
                "successful_connections_per_unit_max": torch.where(dones, maximum, 0.0),
            }
        )
        infos["_episode"] = dones.clone()

    def _disconnect(self, connector_action: torch.Tensor) -> None:
        if self.device.type == "cuda":
            wp.launch(
                kernel=update_binary_connector_disconnections,
                dim=(self.num_envs, self._n_agents, self._n_connectors),
                inputs=[
                    wp.from_torch(connector_action),
                    float(self.scenario.disconnect_potential_threshold),
                    self._n_connectors,
                    self._n_agents,
                    self._n_twists,
                    self._eq_indices_wp,
                ],
                outputs=[
                    self._partner_unit_wp,
                    self._partner_connector_wp,
                    self._connection_twist_idx_wp,
                    self._disconnect_potentials_wp,
                    self._eq_active_wp,
                ],
                device=self._wp_device,
            )
            return

        currently_active = self.partner_unit >= 0
        newly_deactivated = ~connector_action & currently_active
        disconnect_update = self._disconnect_update
        disconnect_update.zero_()
        disconnect_update[newly_deactivated] = 1.0

        deactivated_world, deactivated_unit, deactivated_connector = torch.nonzero(newly_deactivated, as_tuple=True)
        partner_units = self.partner_unit[deactivated_world, deactivated_unit, deactivated_connector]
        partner_connectors = self.partner_connector[deactivated_world, deactivated_unit, deactivated_connector]
        disconnect_update.index_put_(
            (deactivated_world, partner_units, partner_connectors),
            torch.ones_like(partner_units, dtype=disconnect_update.dtype),
            accumulate=True,
        )

        stayed_active = currently_active & (disconnect_update == 0)
        disconnect_update[stayed_active] = -2.0
        self.disconnect_potentials += disconnect_update
        self.disconnect_potentials.clamp_min_(0.0)

        canonical_disconnect = self.disconnect_potentials >= float(self.scenario.disconnect_potential_threshold)
        canonical_disconnect &= currently_active
        canonical_disconnect &= (
            (self._unit_indices < self.partner_unit)
            | ((self._unit_indices == self.partner_unit) & (self._connector_indices < self.partner_connector))
        )

        world_idx, unit1, connector1 = torch.nonzero(canonical_disconnect, as_tuple=True)
        unit2 = self.partner_unit[world_idx, unit1, connector1]
        connector2 = self.partner_connector[world_idx, unit1, connector1]
        eq_variants = self._eq_indices[unit1, connector1, unit2, connector2]
        self._eq_active[world_idx.unsqueeze(1), eq_variants] = 0

        self.partner_unit[world_idx, unit1, connector1] = -1
        self.partner_connector[world_idx, unit1, connector1] = -1
        self.connection_twist_idx[world_idx, unit1, connector1] = -1
        self.disconnect_potentials[world_idx, unit1, connector1] = 0.0

        self.partner_unit[world_idx, unit2, connector2] = -1
        self.partner_connector[world_idx, unit2, connector2] = -1
        self.connection_twist_idx[world_idx, unit2, connector2] = -1
        self.disconnect_potentials[world_idx, unit2, connector2] = 0.0

    def _disconnect_continuous(self, connector_action: torch.Tensor) -> None:
        if self.device.type == "cuda":
            wp.launch(
                kernel=update_continuous_connector_disconnections,
                dim=(self.num_envs, self._n_agents, self._n_connectors),
                inputs=[
                    wp.from_torch(connector_action),
                    float(self.scenario.disconnect_potential_threshold),
                    self._n_connectors,
                    self._n_agents,
                    self._n_twists,
                    self._eq_indices_wp,
                ],
                outputs=[
                    self._partner_unit_wp,
                    self._partner_connector_wp,
                    self._connection_twist_idx_wp,
                    self._disconnect_potentials_wp,
                    self._eq_active_wp,
                ],
                device=self._wp_device,
            )
            return

        currently_active = self.partner_unit >= 0

        partner_flat_idx = (
            self.partner_unit.clamp_min(0) * self._n_connectors
            + self.partner_connector.clamp_min(0)
        ).reshape(self.num_envs, self._n_total_connectors)
        partner_actions = torch.gather(
            connector_action.reshape(self.num_envs, self._n_total_connectors),
            dim=1,
            index=partner_flat_idx,
        ).reshape_as(connector_action)

        disconnect_update = self._disconnect_update
        torch.add(
            (-connector_action).clamp_min(0.0),
            (-partner_actions).clamp_min(0.0),
            out=disconnect_update,
        )
        hold_update = -2.0 * torch.minimum(connector_action.clamp_min(0.0), partner_actions.clamp_min(0.0))
        disconnect_update.copy_(torch.where(disconnect_update > 0.0, disconnect_update, hold_update))
        disconnect_update.masked_fill_(~currently_active, 0.0)
        self.disconnect_potentials += disconnect_update
        self.disconnect_potentials.clamp_min_(0.0)

        canonical_disconnect = self.disconnect_potentials >= float(self.scenario.disconnect_potential_threshold)
        canonical_disconnect &= currently_active
        canonical_disconnect &= (
            (self._unit_indices < self.partner_unit)
            | ((self._unit_indices == self.partner_unit) & (self._connector_indices < self.partner_connector))
        )

        world_idx, unit1, connector1 = torch.nonzero(canonical_disconnect, as_tuple=True)
        unit2 = self.partner_unit[world_idx, unit1, connector1]
        connector2 = self.partner_connector[world_idx, unit1, connector1]
        eq_variants = self._eq_indices[unit1, connector1, unit2, connector2]
        self._eq_active[world_idx.unsqueeze(1), eq_variants] = 0

        self.partner_unit[world_idx, unit1, connector1] = -1
        self.partner_connector[world_idx, unit1, connector1] = -1
        self.connection_twist_idx[world_idx, unit1, connector1] = -1
        self.disconnect_potentials[world_idx, unit1, connector1] = 0.0

        self.partner_unit[world_idx, unit2, connector2] = -1
        self.partner_connector[world_idx, unit2, connector2] = -1
        self.connection_twist_idx[world_idx, unit2, connector2] = -1
        self.disconnect_potentials[world_idx, unit2, connector2] = 0.0

    def _build_obs(self) -> dict[str, torch.Tensor]:
        local_obs = self._tensor_operations.build_local_obs(
            self._local_obs,
            self._qpos,
            self._qvel,
            self.partner_unit,
            self.connection_twist_idx,
            self.disconnect_potentials,
            self._twist_values,
            self._xpos,
            self._qpos_flat_indices,
            self._qvel_flat_indices,
            self._connector_body_indices,
        )
        return {
            "local_obs": local_obs,
            "global_obs": self._scenario_runtime.global_obs,
            "hidden_local_vars": self._scenario_runtime.hidden_local_obs,
            "hidden_global_vars": self._scenario_runtime.hidden_global_obs,
            "agent_mask": self.units_active_mask,
        }

    def _apply_error_obs(self, obs: dict[str, torch.Tensor], unstable_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        (
            obs["local_obs"],
            obs["global_obs"],
            obs["hidden_local_vars"],
            obs["hidden_global_vars"],
        ) = self._tensor_operations.mask_error_observations(
            obs["local_obs"],
            obs["global_obs"],
            obs["hidden_local_vars"],
            obs["hidden_global_vars"],
            unstable_mask,
        )
        return obs

    def _capture_world_snapshots(self, world_idx: torch.Tensor) -> dict[int, MJWWorldSnapshot]:
        if world_idx.numel() == 0:
            return {}

        world_idx_cpu = world_idx.detach().to(device="cpu", dtype=torch.long)
        qpos = self._qpos[world_idx].detach().cpu().numpy().copy()
        qvel = self._qvel[world_idx].detach().cpu().numpy().copy()
        eq_active = self._eq_active[world_idx].detach().cpu().numpy().copy()
        mocap_pos = self._mocap_pos[world_idx].detach().cpu().numpy().copy()
        mocap_quat = self._mocap_quat[world_idx].detach().cpu().numpy().copy()
        times = self._time[world_idx].detach().cpu().numpy().copy()

        return {
            int(world): MJWWorldSnapshot(
                qpos=qpos[i],
                qvel=qvel[i],
                eq_active=eq_active[i],
                mocap_pos=mocap_pos[i],
                mocap_quat=mocap_quat[i],
                time=float(times[i]),
            )
            for i, world in enumerate(world_idx_cpu.tolist())
        }
