from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from swarmbots.utils.connector_actions import connector_action_space
from swarmbots.mj_env.float_or_dist_params import BoundedDistParams, FloatOrDistParams
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm
from swarmbots.scenario_presets.move_to_goal_config import AbsoluteGoalConfig, MoveToGoalConfig, RelativePolarGoalConfig


@dataclass(frozen=True, slots=True)
class MJWMoveToRuntimeMetadata:
    goal_mocap_id: int


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
class MJWMoveToScenario(BaseMJWScenario):
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
    goal: MoveToGoalConfig
    goal_radius: float
    continuous_connector_actions: bool = True
    visualize_goal: bool = False
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False

    def __post_init__(self) -> None:
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWMoveToScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.plane_size = float(self.plane_size)
        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        self.goal_radius = float(self.goal_radius)
        if self.goal_radius < 0.0:
            raise ValueError(f"Expected goal_radius >= 0, got {self.goal_radius}")
        self.forward_reward_weight = float(self.forward_reward_weight)
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)
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
            "plane_size": self.plane_size,
            "goal": self.goal,
            "goal_radius": self.goal_radius,
            "visualize_goal": self.visualize_goal,
            "forward_reward_weight": self.forward_reward_weight,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        if isinstance(self.goal, AbsoluteGoalConfig):
            goal_y = self.goal.y if isinstance(self.goal.y, (int, float)) else 3.0
            lookat_y = float(goal_y) * 0.5
            distance = abs(float(goal_y)) + self.swarm.max_unit_extent * 6.0
        else:
            lookat_y = 0.0
            distance = self._recording_distance_for_relative_goal(self.goal)
        return MJWRecordingCameraConfig(
            lookat=(0.0, lookat_y, 0.7),
            distance=max(7.0, min(16.0, distance)),
            azimuth=180.0,
            elevation=-35.0,
        )

    def _recording_distance_for_relative_goal(self, goal: RelativePolarGoalConfig) -> float:
        if isinstance(goal.distance, (float, int)):
            max_goal_distance = abs(float(goal.distance))
        elif isinstance(goal.distance, BoundedDistParams):
            max_goal_distance = max(abs(float(goal.distance.low)), abs(float(goal.distance.high)))
        else:
            max_goal_distance = 3.0
        return max_goal_distance + self.swarm.max_unit_extent * 6.0

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

        goal_body = worldbody.add_body(name="Goal", mocap=True, pos=[0, 0, 0])
        if self.visualize_goal:
            goal_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[0, 0, 0.005, 0, 0, 0.035],
                size=[max(self.goal_radius, 0.01), 0, 0],
                rgba=[0.1, 0.95, 0.35, 0.8],
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
                "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.swarm.num_units, local_obs_dim), dtype=np.float32),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
                "hidden_local_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(self.swarm.num_units, 0), dtype=np.float32),
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

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWMoveToRuntimeMetadata:
        goal_body_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_BODY, "Goal")
        if goal_body_id < 0:
            raise ValueError("Goal body not found in MoveTo model.")
        return MJWMoveToRuntimeMetadata(goal_mocap_id=int(host_model.body_mocapid[goal_body_id]))

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_move_to_runtime import MoveToMJWScenarioRuntime

        return MoveToMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
