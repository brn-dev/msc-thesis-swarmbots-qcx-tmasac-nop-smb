from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from swarmbots.utils.connector_actions import connector_action_space
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm
from swarmbots.scenario_presets.scenario_obs_layouts import CLIMB_GOAL_XYZ_GLOBAL_OBS_LAYOUT


def _hinges_per_limb(unit_config: tuple[object, ...]) -> int:
    total = 0
    for limb in unit_config:
        limb_type = str(limb.type).removeprefix("LimbType.")
        if limb_type in ("xy", "zx"):
            total += 2
        elif limb_type == "xyz":
            total += 3
        else:
            raise ValueError(f"Unsupported limb type: {limb_type}")
    return total


@dataclass
class MJWClimbScenario(BaseMJWScenario):
    swarm: MJWHomogeneousSwarm
    timestep: float
    action_repeat: int
    actuator_strength: float
    progress_reward_weight: float
    horizontal_reward_weight: float
    height_reward_weight: float
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
    cuboid_size_x: float
    cuboid_size_y: float
    cuboid_size_z: float
    cuboid_center_x: float
    cuboid_center_y: float
    horizontal_goal_radius: float
    height_goal_radius: float
    continuous_connector_actions: bool = True
    goal_radius: float | None = None
    goal_height_offset: float | None = None
    goal_success_reward: float = 5.0
    visualize_goal: bool = True
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False

    def __post_init__(self) -> None:
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWClimbScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")

        self.plane_size = float(self.plane_size)
        self.cuboid_size_x = float(self.cuboid_size_x)
        self.cuboid_size_y = float(self.cuboid_size_y)
        self.cuboid_size_z = float(self.cuboid_size_z)
        self.cuboid_center_x = float(self.cuboid_center_x)
        self.cuboid_center_y = float(self.cuboid_center_y)
        self.horizontal_goal_radius = float(self.horizontal_goal_radius)
        self.height_goal_radius = float(self.height_goal_radius)
        self.goal_radius = (
            min(self.cuboid_size_x, self.cuboid_size_y) * 0.375
            if self.goal_radius is None
            else float(self.goal_radius)
        )
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)
        self.goal_height_offset = (
            float(self.swarm.max_unit_extent) / 2.0
            if self.goal_height_offset is None
            else float(self.goal_height_offset)
        )
        self.goal_success_reward = float(self.goal_success_reward)
        self.horizontal_reward_weight = float(self.horizontal_reward_weight)
        self.height_reward_weight = float(self.height_reward_weight)
        self.inactive_area_location = (50.0, 0.0, 0.1)

        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        if self.cuboid_size_x <= 0.0:
            raise ValueError(f"Expected cuboid_size_x > 0, got {self.cuboid_size_x}")
        if self.cuboid_size_y <= 0.0:
            raise ValueError(f"Expected cuboid_size_y > 0, got {self.cuboid_size_y}")
        if self.cuboid_size_z <= 0.0:
            raise ValueError(f"Expected cuboid_size_z > 0, got {self.cuboid_size_z}")
        if self.horizontal_goal_radius < 0.0:
            raise ValueError(f"Expected horizontal_goal_radius >= 0, got {self.horizontal_goal_radius}")
        if self.height_goal_radius < 0.0:
            raise ValueError(f"Expected height_goal_radius >= 0, got {self.height_goal_radius}")
        if self.goal_radius < 0.0:
            raise ValueError(f"Expected goal_radius >= 0, got {self.goal_radius}")
        if self.goal_height_offset < 0.0:
            raise ValueError(f"Expected goal_height_offset >= 0, got {self.goal_height_offset}")

        self.goal_position = (
            self.cuboid_center_x,
            self.cuboid_center_y,
            self.cuboid_size_z + self.goal_height_offset,
        )

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
            "plane_size": self.plane_size,
            "cuboid_size_x": self.cuboid_size_x,
            "cuboid_size_y": self.cuboid_size_y,
            "cuboid_size_z": self.cuboid_size_z,
            "cuboid_center_x": self.cuboid_center_x,
            "cuboid_center_y": self.cuboid_center_y,
            "horizontal_goal_radius": self.horizontal_goal_radius,
            "height_goal_radius": self.height_goal_radius,
            "goal_radius": self.goal_radius,
            "goal_height_offset": self.goal_height_offset,
            "goal_success_reward": self.goal_success_reward,
            "global_obs_layout": CLIMB_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
            "visualize_goal": self.visualize_goal,
            "horizontal_reward_weight": self.horizontal_reward_weight,
            "height_reward_weight": self.height_reward_weight,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        return MJWRecordingCameraConfig(
            lookat=(self.cuboid_center_x, self.cuboid_center_y * 0.5, self.cuboid_size_z * 0.75),
            distance=max(6.0, min(14.0, self.cuboid_size_y + self.swarm.max_unit_extent * 6.0)),
            azimuth=180.0,
            elevation=-30.0,
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

        worldbody.add_geom(
            name="ClimbCuboid",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[
                self.cuboid_size_x / 2.0,
                self.cuboid_size_y / 2.0,
                self.cuboid_size_z / 2.0,
            ],
            pos=[
                self.cuboid_center_x,
                self.cuboid_center_y,
                self.cuboid_size_z / 2.0,
            ],
            rgba=[0.55, 0.55, 0.58, 1.0],
        )
        if self.visualize_goal:
            worldbody.add_geom(
                name="ClimbGoal",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.goal_radius, 0.0, 0.0],
                pos=list(self.goal_position),
                rgba=[0.1, 0.95, 0.35, 0.1],
                contype=0,
                conaffinity=0,
            )

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        model = spec.compile()
        model.opt.timestep = float(self.timestep)
        if self.friction is not None:
            if isinstance(self.friction, (int, float)):
                friction = np.asarray([float(self.friction), 0.005, 0.0001], dtype=float)
            else:
                friction = np.asarray(tuple(float(v) for v in self.friction), dtype=float)
            model.geom_friction[:] = friction
        return model

    def get_single_observation_space(self) -> spaces.Dict:
        limbs_per_unit = self.swarm.config.limbs_per_unit
        num_hinges = _hinges_per_limb(self.swarm.config.unit_config)
        free_joint_rot_dim = 6 if self.quat_rot6d_representation else 4
        qpos_obs_dim = 3 + free_joint_rot_dim + (2 * num_hinges)
        qvel_dim = 6 + num_hinges
        connector_obs_dim = limbs_per_unit * 5
        connectors_xpos_dim = limbs_per_unit * 3 if self.include_connectors_xpos_in_obs else 0
        local_obs_dim = qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim
        return spaces.Dict(
            {
                "local_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.swarm.num_units, local_obs_dim),
                    dtype=np.float32,
                ),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
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

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> None:
        return None

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_climb_runtime import ClimbMJWScenarioRuntime

        return ClimbMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
