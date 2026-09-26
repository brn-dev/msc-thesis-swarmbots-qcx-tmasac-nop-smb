from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.scenario_presets.multi_payload_goal import (
    PAYLOAD_GOAL_COLORS,
    PAYLOAD_GOAL_OBS_RECORD_DIM,
    normalize_count_probs,
)
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import PayloadShape, _hinges_per_limb
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm
from swarmbots.scenario_presets.scenario_obs_layouts import MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT
from swarmbots.utils.connector_actions import connector_action_space


@dataclass(frozen=True, slots=True)
class MJWMultiPayloadGoalRuntimeMetadata:
    payload_qpos_indices: np.ndarray
    goal_mocap_ids: np.ndarray


@dataclass
class MJWMultiPayloadGoalScenario:
    swarm: MJWHomogeneousSwarm
    timestep: float
    action_repeat: int
    actuator_strength: float
    progress_reward_weight: float
    forward_reward_weight: float
    guidance_reward_weight: float
    units_without_connections_reward_weight: float
    potential_reward_discount_factor: float
    include_connectors_xpos_in_obs: bool
    include_connectors_xquat_in_obs: bool
    quat_rot6d_representation: bool
    connection_dist_threshold: float
    connection_angle_threshold: float
    disconnect_potential_threshold: float
    friction: float | Iterable[float] | None
    reset_settle_time: float
    reset_settle_timestep_scale: float
    swarm_start_x: FloatOrDistParams
    swarm_start_y: FloatOrDistParams
    plane_size: float
    payload_shape: PayloadShape
    payload_radius: float
    payload_mass: float
    max_payloads: int
    active_payload_count_probs: Mapping[int, float] | None
    payload_spawn_y: float
    payload_spawn_margin: float
    goal_rect: tuple[float, float, float, float]
    goal_radius: float
    success_reward: float
    visualize_goal: bool = True
    continuous_connector_actions: bool = True
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False
    physics_nconmax: int | None = None
    physics_njmax: int | None = 250

    def __post_init__(self) -> None:
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWMultiPayloadGoalScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.plane_size = float(self.plane_size)
        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        self.max_payloads = int(self.max_payloads)
        if self.max_payloads < 1:
            raise ValueError(f"Expected max_payloads >= 1, got {self.max_payloads}")
        if self.payload_shape not in ("sphere", "box", "capsule"):
            raise ValueError(
                f"Expected payload_shape to be 'sphere', 'box', or 'capsule', got {self.payload_shape!r}"
            )
        self.payload_radius = float(self.payload_radius)
        if self.payload_radius <= 0.0:
            raise ValueError(f"Expected payload_radius > 0, got {self.payload_radius}")
        self.payload_mass = float(self.payload_mass)
        if self.payload_mass <= 0.0:
            raise ValueError(f"Expected payload_mass > 0, got {self.payload_mass}")
        self.payload_spawn_y = float(self.payload_spawn_y)
        self.payload_spawn_margin = float(self.payload_spawn_margin)
        if self.payload_spawn_margin < 2.0 * self.payload_radius:
            raise ValueError(
                "payload_spawn_margin must be at least two payload radii, "
                f"got {self.payload_spawn_margin} for radius {self.payload_radius}"
            )
        self.goal_rect = tuple(float(value) for value in self.goal_rect)
        if len(self.goal_rect) != 4:
            raise ValueError("goal_rect must be (x_min, x_max, y_min, y_max)")
        goal_x_min, goal_x_max, goal_y_min, goal_y_max = self.goal_rect
        if goal_x_min >= goal_x_max or goal_y_min >= goal_y_max:
            raise ValueError(f"Invalid goal_rect bounds: {self.goal_rect!r}")
        self.goal_radius = float(self.goal_radius)
        if self.goal_radius < 0.0:
            raise ValueError(f"Expected goal_radius >= 0, got {self.goal_radius}")
        self.success_reward = float(self.success_reward)
        self.forward_reward_weight = float(self.forward_reward_weight)
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)
        self.active_payload_count_probs = normalize_count_probs(
            probs=self.active_payload_count_probs,
            max_payloads=self.max_payloads,
        )
        if self.physics_nconmax is not None and self.physics_nconmax <= 0:
            raise ValueError(f"Expected physics_nconmax > 0, got {self.physics_nconmax}")
        if self.physics_njmax is not None and self.physics_njmax <= 0:
            raise ValueError(f"Expected physics_njmax > 0, got {self.physics_njmax}")
        self.inactive_area_location = (50.0, 0.0, 0.1)

    def get_settings(self) -> dict[str, Any]:
        return {
            "swarm": self.swarm.get_settings(),
            "timestep": self.timestep,
            "action_repeat": self.action_repeat,
            "actuator_strength": self.actuator_strength,
            "reward_weights": {
                "progress_reward_weight": self.progress_reward_weight,
                "guidance_reward_weight": self.guidance_reward_weight,
                "units_without_connections_reward_weight": self.units_without_connections_reward_weight,
            },
            "potential_reward_discount_factor": self.potential_reward_discount_factor,
            "include_connectors_xpos_in_obs": self.include_connectors_xpos_in_obs,
            "include_connectors_xquat_in_obs": self.include_connectors_xquat_in_obs,
            "quat_rot6d_representation": self.quat_rot6d_representation,
            "connection_dist_threshold": self.connection_dist_threshold,
            "connection_angle_threshold": self.connection_angle_threshold,
            "disconnect_potential_threshold": self.disconnect_potential_threshold,
            "continuous_connector_actions": self.continuous_connector_actions,
            "swarm_start_x": self.swarm_start_x,
            "swarm_start_y": self.swarm_start_y,
            "randomize_initial_swarm_z_rotation": self.randomize_initial_swarm_z_rotation,
            "friction": self.friction,
            "seed": self.seed,
            "reset_settle_time": self.reset_settle_time,
            "reset_settle_timestep_scale": self.reset_settle_timestep_scale,
            "scenario_type": "multi_payload_goal",
            "global_obs_layout": MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT,
            "plane_size": self.plane_size,
            "num_payloads": self.max_payloads,
            "max_payloads": self.max_payloads,
            "active_payload_count_probs": dict(self.active_payload_count_probs),
            "payload_shape": self.payload_shape,
            "payload_radius": self.payload_radius,
            "payload_mass": self.payload_mass,
            "payload_spawn_y": self.payload_spawn_y,
            "payload_spawn_margin": self.payload_spawn_margin,
            "goal_rect": self.goal_rect,
            "goal_radius": self.goal_radius,
            "visualize_goal": self.visualize_goal,
            "success_reward": self.success_reward,
            "forward_reward_weight": self.forward_reward_weight,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
            "physics_nconmax": self.physics_nconmax,
            "physics_njmax": self.physics_njmax,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        _, _, goal_y_min, goal_y_max = self.goal_rect
        lookat_y = (self.payload_spawn_y + ((goal_y_min + goal_y_max) / 2.0)) / 2.0
        distance = max(8.0, min(16.0, self.swarm.max_unit_extent * 10.0 + self.payload_radius * 8.0))
        return MJWRecordingCameraConfig(
            lookat=(0.0, float(lookat_y), max(0.5, self.payload_radius * 4.0)),
            distance=distance,
            azimuth=180.0,
            elevation=-35.0,
        )

    def build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.plane_size, self.plane_size, 0.1],
            rgba=[0.2, 0.3, 0.4, 1.0],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        for payload_idx in range(self.max_payloads):
            payload_name = self._payload_body_name(payload_idx)
            payload_body = worldbody.add_body(name=payload_name, pos=[0, 0, self.payload_radius])
            payload_body.add_freejoint(name=f"{payload_name}_freejoint")
            self._add_payload_geom(payload_body, rgba=self._payload_color(payload_idx))

            goal_body = worldbody.add_body(name=self._goal_body_name(payload_idx), mocap=True, pos=[0, 0, 0])
            if self.visualize_goal:
                goal_rgba = list(self._payload_color(payload_idx))
                goal_rgba[3] = 0.18
                goal_body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[self.goal_radius, 0.0, 0.0],
                    pos=[0.0, 0.0, self.payload_radius],
                    rgba=goal_rgba,
                    contype=0,
                    conaffinity=0,
                )

        model = spec.compile()
        model.opt.timestep = float(self.timestep)
        if self.friction is not None:
            if isinstance(self.friction, (int, float)):
                friction = np.asarray([float(self.friction), 0.005, 0.0001], dtype=float)
            else:
                friction = np.asarray(tuple(float(v) for v in self.friction), dtype=float)
            model.geom_friction[:] = friction
        return model

    def _add_payload_geom(
        self,
        payload_body: mujoco.MjsBody,
        *,
        rgba: tuple[float, float, float, float],
    ) -> None:
        geom_kwargs: dict[str, Any] = {
            "mass": self.payload_mass,
            "rgba": list(rgba),
        }
        if self.payload_shape == "sphere":
            payload_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.payload_radius, 0.0, 0.0],
                **geom_kwargs,
            )
        elif self.payload_shape == "box":
            payload_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.payload_radius, self.payload_radius, self.payload_radius],
                **geom_kwargs,
            )
        elif self.payload_shape == "capsule":
            payload_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[-self.payload_radius, 0.0, 0.0, self.payload_radius, 0.0, 0.0],
                size=[self.payload_radius, 0.0, 0.0],
                **geom_kwargs,
            )
        else:
            raise AssertionError(f"Unhandled payload_shape: {self.payload_shape}")

    def add_render_geoms(self, scene: mujoco.MjvScene) -> None:
        return

    def get_single_observation_space(self) -> spaces.Dict:
        limbs_per_unit = self.swarm.config.limbs_per_unit
        num_hinges = _hinges_per_limb(self.swarm.config.unit_config)
        free_joint_rot_dim = 6 if self.quat_rot6d_representation else 4
        qpos_obs_dim = 3 + free_joint_rot_dim + (2 * num_hinges)
        qvel_dim = 6 + num_hinges
        connector_obs_dim = limbs_per_unit * 5
        connectors_xpos_dim = limbs_per_unit * 3 if self.include_connectors_xpos_in_obs else 0
        local_obs_dim = qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim
        global_obs_dim = self.max_payloads * PAYLOAD_GOAL_OBS_RECORD_DIM
        return spaces.Dict(
            {
                "local_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.swarm.num_units, local_obs_dim),
                    dtype=np.float32,
                ),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(global_obs_dim,), dtype=np.float32),
                "hidden_local_vars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.swarm.num_units, 0),
                    dtype=np.float32,
                ),
                "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32),
                "agent_mask": spaces.MultiBinary((self.swarm.num_units,)),
            }
        )

    def get_single_action_space(self) -> spaces.Dict:
        actuators_per_unit = _hinges_per_limb(self.swarm.config.unit_config)
        return spaces.Dict(
            {
                "actuators": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.swarm.num_units, actuators_per_unit),
                    dtype=np.float32,
                ),
                "connectors": connector_action_space(
                    (self.swarm.num_units, self.swarm.config.limbs_per_unit),
                    continuous=self.continuous_connector_actions,
                ),
            }
        )

    def get_batched_observation_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_observation_space(), n=num_envs)

    def get_batched_action_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_action_space(), n=num_envs)

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWMultiPayloadGoalRuntimeMetadata:
        payload_qpos_indices = np.stack(
            [
                np.asarray(
                    mj_utils.qpos_indices_for_body(host_model, self._payload_body_name(payload_idx)),
                    dtype=np.int64,
                )
                for payload_idx in range(self.max_payloads)
            ],
            axis=0,
        )
        if payload_qpos_indices.shape != (self.max_payloads, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")

        goal_mocap_ids = np.zeros((self.max_payloads,), dtype=np.int64)
        for payload_idx in range(self.max_payloads):
            goal_body_id = mujoco.mj_name2id(
                host_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self._goal_body_name(payload_idx),
            )
            goal_mocap_ids[payload_idx] = int(host_model.body_mocapid[goal_body_id])
        return MJWMultiPayloadGoalRuntimeMetadata(
            payload_qpos_indices=payload_qpos_indices,
            goal_mocap_ids=goal_mocap_ids,
        )

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import (
            MultiPayloadGoalMJWScenarioRuntime,
        )

        return MultiPayloadGoalMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )

    @staticmethod
    def _payload_body_name(payload_idx: int) -> str:
        return f"Payload{payload_idx}"

    @staticmethod
    def _goal_body_name(payload_idx: int) -> str:
        return f"PayloadGoal{payload_idx}"

    @staticmethod
    def _payload_color(payload_idx: int) -> tuple[float, float, float, float]:
        return PAYLOAD_GOAL_COLORS[payload_idx % len(PAYLOAD_GOAL_COLORS)]
