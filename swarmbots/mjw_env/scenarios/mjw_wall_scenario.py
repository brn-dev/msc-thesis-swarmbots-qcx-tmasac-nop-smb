from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from swarmbots.utils.connector_actions import connector_action_space
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRuntimeBindings, MJWRecordingCameraConfig
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm
from swarmbots.utils.mujoco_render_geoms import add_wall_y_reference_line_geoms


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
class MJWWallScenario(BaseMJWScenario):
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
    wall_height: float
    wall_distance: FloatOrDistParams
    street_width: float
    wall_pass_reward_weight: float
    wall_pass_thresholds: list[float]
    wall_success_threshold: float
    continuous_connector_actions: bool = True
    wall_pass_reward_skew: float = 0.0
    wall_success_reward: float = 0.0
    wall_climb_reward_weight: float = 0.0
    wall_climb_reward_distance: float = 0.45
    forward_reward_wall_boost_factor: float = 1.0
    forward_reward_wall_boost_distance: float | None = None
    forward_reward_wall_boost_height_margin: float | None = None
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False

    def __post_init__(self) -> None:
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWWallScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)
        self.forward_reward_wall_boost_factor = float(self.forward_reward_wall_boost_factor)
        if self.forward_reward_wall_boost_factor < 1.0:
            raise ValueError(
                f"Expected forward_reward_wall_boost_factor >= 1.0, got {self.forward_reward_wall_boost_factor}"
            )
        if self.forward_reward_wall_boost_distance is not None:
            self.forward_reward_wall_boost_distance = float(self.forward_reward_wall_boost_distance)
            if self.forward_reward_wall_boost_distance <= 0.0:
                raise ValueError(
                    f"Expected forward_reward_wall_boost_distance > 0, got {self.forward_reward_wall_boost_distance}"
                )
        if self.forward_reward_wall_boost_height_margin is not None:
            self.forward_reward_wall_boost_height_margin = float(self.forward_reward_wall_boost_height_margin)
            if self.forward_reward_wall_boost_height_margin < 0.0:
                raise ValueError(
                    "Expected forward_reward_wall_boost_height_margin >= 0, "
                    f"got {self.forward_reward_wall_boost_height_margin}"
                )
        self.wall_height = float(self.wall_height)
        self.wall_climb_reward_weight = float(self.wall_climb_reward_weight)
        self.wall_climb_reward_distance = float(self.wall_climb_reward_distance)
        self.wall_success_threshold = float(self.wall_success_threshold)
        self.wall_success_reward = float(self.wall_success_reward)
        if self.wall_climb_reward_distance <= 0.0:
            raise ValueError(f"Expected wall_climb_reward_distance > 0, got {self.wall_climb_reward_distance}")
        if self.wall_success_threshold <= 0.0:
            raise ValueError(f"Expected wall_success_threshold > 0, got {self.wall_success_threshold}")
        self.street_width = float(self.street_width)
        self.side_wall_x = self.street_width / 2.0
        self.total_thresholds = len(self.wall_pass_thresholds)
        self.inactive_area_location = (self.street_width * 2.0, 0.0, 0.1)

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
            "wall_height": self.wall_height,
            "wall_distance": self.wall_distance,
            "street_width": self.street_width,
            "forward_reward_weight": self.forward_reward_weight,
            "forward_reward_wall_boost_factor": self.forward_reward_wall_boost_factor,
            "forward_reward_wall_boost_distance": self.forward_reward_wall_boost_distance,
            "forward_reward_wall_boost_height_margin": self.forward_reward_wall_boost_height_margin,
            "wall_pass_reward_weight": self.wall_pass_reward_weight,
            "wall_pass_reward_skew": self.wall_pass_reward_skew,
            "wall_pass_thresholds": list(self.wall_pass_thresholds),
            "wall_success_threshold": self.wall_success_threshold,
            "wall_success_reward": self.wall_success_reward,
            "wall_climb_reward_weight": self.wall_climb_reward_weight,
            "wall_climb_reward_distance": self.wall_climb_reward_distance,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        wall_distance = self.wall_distance if isinstance(self.wall_distance, (int, float)) else 1.0
        lookat_y = max(0.75, min(float(wall_distance) * 0.9, float(wall_distance) + 0.5))
        lookat_z = max(0.35, self.wall_height * 1.25)
        distance = max(3.0, min(8.0, self.street_width * 0.45 + self.swarm.max_unit_extent * 1.5))
        return MJWRecordingCameraConfig(
            lookat=(0.0, lookat_y, lookat_z),
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
            size=[100, 100, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[self.side_wall_x, 0, 0],
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-self.side_wall_x, 0, 0],
        )

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        wall_body = worldbody.add_body(name="Wall", mocap=True, pos=[0, 0, 0])
        wall_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.side_wall_x, 0.1, self.wall_height],
            rgba=[0.5, 0.5, 0.6, 1],
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
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32),
                "hidden_local_vars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.swarm.num_units, self.total_thresholds),
                    dtype=np.float32,
                ),
                "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
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
        _ = host_model
        return None

    def add_render_geoms(self, scene: mujoco.MjvScene) -> None:
        add_wall_y_reference_line_geoms(scene, wall_distance=self.wall_distance)

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_wall_runtime import WallMJWScenarioRuntime

        return WallMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
