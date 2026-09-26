from __future__ import annotations

from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.scenario_presets.scenario_obs_layouts import CLIMB_GOAL_XYZ_GLOBAL_OBS_LAYOUT


class ClimbScenario(BaseScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        cuboid_size_x: float = 3.0,
        cuboid_size_y: float = 3.0,
        cuboid_size_z: float = 1.0,
        cuboid_center_x: float = 0.0,
        cuboid_center_y: float = 2.0,
        horizontal_goal_radius: float = 0.3,
        height_goal_radius: float = 0.1,
        goal_radius: float | None = None,
        goal_height_offset: float | None = None,
        goal_success_reward: float = 5.0,
        visualize_goal: bool = True,
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        horizontal_reward_weight: float = 1.0,
        height_reward_weight: float = 1.0,
        guidance_reward_weight: float = 1.0,
        units_without_connections_reward_weight: float = 0.0,
        potential_reward_discount_factor: float = 1.0,
        include_connectors_xpos_in_obs: bool = True,
        include_connectors_xquat_in_obs: bool = False,
        quat_rot6d_representation: bool = True,
        reset_settle_time: float = 0.0,
        reset_settle_timestep_scale: float = 1.0,
        swarm_start_x: FloatOrDistParams = 0.0,
        swarm_start_y: FloatOrDistParams = 0.0,
        randomize_initial_swarm_z_rotation: bool = False,
        continuous_connector_actions: bool = True,
        seed: int | None = None,
    ) -> None:
        self.plane_size = float(plane_size)
        self.cuboid_size_x = float(cuboid_size_x)
        self.cuboid_size_y = float(cuboid_size_y)
        self.cuboid_size_z = float(cuboid_size_z)
        self.cuboid_center_x = float(cuboid_center_x)
        self.cuboid_center_y = float(cuboid_center_y)
        self.horizontal_goal_radius = float(horizontal_goal_radius)
        self.height_goal_radius = float(height_goal_radius)
        self.goal_radius = (
            min(self.cuboid_size_x, self.cuboid_size_y) * 0.375
            if goal_radius is None
            else float(goal_radius)
        )
        self.goal_height_offset = (
            float(swarm.max_unit_extent) / 2.0 if goal_height_offset is None else float(goal_height_offset)
        )
        self.goal_success_reward = float(goal_success_reward)
        self.visualize_goal = bool(visualize_goal)
        self.horizontal_reward_weight = float(horizontal_reward_weight)
        self.height_reward_weight = float(height_reward_weight)

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

        self.goal_position = np.array(
            [self.cuboid_center_x, self.cuboid_center_y, self.cuboid_size_z + self.goal_height_offset],
            dtype=float,
        )

        super().__init__(
            swarm=swarm,
            timestep=timestep,
            action_repeat=action_repeat,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            potential_reward_discount_factor=potential_reward_discount_factor,
            seed=seed,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            friction=friction,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            force_elliptic_cone=force_elliptic_cone,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            swarm_start_x=swarm_start_x,
            swarm_start_y=swarm_start_y,
            randomize_initial_swarm_z_rotation=randomize_initial_swarm_z_rotation,
            continuous_connector_actions=continuous_connector_actions,
            inactive_area_location=None,
        )

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            {
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
            }
        )
        return settings

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: mujoco.MjsBody = spec.worldbody

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
                pos=self.goal_position.tolist(),
                rgba=[0.1, 0.95, 0.35, 0.1],
                contype=0,
                conaffinity=0,
            )
        return spec

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=settle)
        state["goal_position"] = self.goal_position.copy()
        state["success"] = False
        self._reset_progress_baselines(data, state)
        return state, connections

    def get_obs(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: dict,
        connections: SwarmConnections,
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)
        obs["global_obs"] = np.asarray(state["goal_position"], dtype=float).copy()
        obs["hidden_global_vars"] = np.zeros((0,), dtype=float)
        return obs

    def compute_progress_reward(
        self,
        data: mujoco.MjData,
        state: dict,
    ) -> float:
        old_horizontal_progress = state["horizontal_progress"]
        new_horizontal_progress = self._compute_horizontal_progress_baseline(
            data=data,
            units_active_mask=state.get("units_active_mask"),
        )
        state["horizontal_progress"] = new_horizontal_progress

        old_height_progress = state["height_progress"]
        new_height_progress = self._compute_axis_progress_baseline(
            data=data,
            units_active_mask=state.get("units_active_mask"),
            axis=2,
        )
        state["height_progress"] = new_height_progress

        horizontal_reward = (
            self.potential_reward_delta(new_horizontal_progress, old_horizontal_progress)
            * self.horizontal_reward_weight
        )
        height_reward = (
            self.potential_reward_delta(new_height_progress, old_height_progress)
            * self.height_reward_weight
        )
        progress_reward = horizontal_reward + height_reward

        state["progress"] = new_horizontal_progress + new_height_progress
        state["horizontal_reward"] = horizontal_reward
        state["height_reward"] = height_reward
        state["progress_reward"] = progress_reward
        return progress_reward

    def evaluate_step(
        self,
        action: SwarmActDict,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: dict,
        connections: SwarmConnections,
    ) -> tuple[float, bool]:
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is not None:
            self._enforce_inactive_units_state(model, data, units_active_mask)

        progress_reward = self.compute_progress_reward(data, state)
        horizontal_reward = state["horizontal_reward"]
        height_reward = state["height_reward"]
        terminated = self._compute_goal_success_termination(data, state)
        goal_success_reward = self.goal_success_reward if terminated else 0.0
        progress_reward += goal_success_reward
        state["progress_reward"] = progress_reward
        state["goal_success_reward"] = goal_success_reward

        guidance_reward = super().compute_guidance_reward(data, action, state, connections)
        state["guidance_reward"] = guidance_reward

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_horizontal_reward = horizontal_reward * progress_reward_weight
        weighted_height_reward = height_reward * progress_reward_weight
        weighted_goal_success_reward = goal_success_reward * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_horizontal_reward"] = weighted_horizontal_reward
        state["weighted_height_reward"] = weighted_height_reward
        state["weighted_goal_success_reward"] = weighted_goal_success_reward
        state["weighted_guidance_reward"] = weighted_guidance_reward
        state["reward_terms"] = {
            "horizontal": weighted_horizontal_reward,
            "height": weighted_height_reward,
            "success": weighted_goal_success_reward,
            "guidance": weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, terminated

    def _compute_goal_success_termination(
        self,
        data: mujoco.MjData,
        state: dict,
    ) -> bool:
        unit_positions = np.asarray(data.qpos[self._qpos_indices[:, :3]], dtype=float)
        distance_to_goal = np.linalg.norm(unit_positions - self.goal_position[np.newaxis, :], axis=1)
        units_inside_goal = distance_to_goal <= self.goal_radius
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            success = bool(units_inside_goal.all())
        else:
            active_units_mask = np.asarray(units_active_mask, dtype=bool)
            success = bool(active_units_mask.any() and units_inside_goal[active_units_mask].all())
        state["success"] = success
        return success

    def _reset_progress_baselines(self, data: mujoco.MjData, state: dict) -> None:
        units_active_mask = state.get("units_active_mask")
        horizontal_progress = self._compute_horizontal_progress_baseline(data=data, units_active_mask=units_active_mask)
        height_progress = self._compute_axis_progress_baseline(
            data=data,
            units_active_mask=units_active_mask,
            axis=2,
        )
        state["horizontal_progress"] = horizontal_progress
        state["height_progress"] = height_progress
        state["progress"] = horizontal_progress + height_progress

    def _compute_horizontal_progress_baseline(
        self,
        *,
        data: mujoco.MjData,
        units_active_mask: np.ndarray | None,
    ) -> float:
        unit_xy = np.asarray(data.qpos[self._qpos_indices[:, :2]], dtype=float)
        distance_to_goal = np.linalg.norm(unit_xy - self.goal_position[np.newaxis, :2], axis=1)
        remaining_distance = np.maximum(distance_to_goal - self.horizontal_goal_radius, 0.0)
        if units_active_mask is None:
            return -float(remaining_distance.mean())

        active_mask = np.asarray(units_active_mask, dtype=bool)
        active_units_count = int(active_mask.sum())
        if active_units_count == 0:
            return 0.0
        return -float(remaining_distance[active_mask].mean())

    def _compute_axis_progress_baseline(
        self,
        *,
        data: mujoco.MjData,
        units_active_mask: np.ndarray | None,
        axis: int,
    ) -> float:
        unit_axis_positions = np.asarray(data.qpos[self._qpos_indices[:, axis]], dtype=float)
        distance_to_goal = np.abs(unit_axis_positions - self.goal_position[axis])
        remaining_distance = np.maximum(distance_to_goal - self.height_goal_radius, 0.0)
        if units_active_mask is None:
            return -float(remaining_distance.mean())

        active_mask = np.asarray(units_active_mask, dtype=bool)
        active_units_count = int(active_mask.sum())
        if active_units_count == 0:
            return 0.0
        return -float(remaining_distance[active_mask].mean())
