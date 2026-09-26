from __future__ import annotations

from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.scenario_presets.scenario_obs_layouts import VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT


class VerticalReachScenario(BaseScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        wall_width: float = 6.0,
        wall_thickness: float = 0.35,
        wall_height: float = 6.0,
        wall_center_x: float = 0.0,
        wall_y: float = 2.5,
        goal_box_width: float = 1.0,
        goal_box_depth: float = 0.4,
        goal_box_height: float = 0.4,
        goal_center_z: float = 1.5,
        goal_success_reward: float = 5.0,
        reach_column_half_width: float = 1.0,
        reach_column_depth: float = 1.0,
        reach_column_reward_weight: float = 8.0,
        horizontal_goal_at_wall_contact: bool = True,
        visualize_goal: bool = True,
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        horizontal_reward_weight: float = 0.25,
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
        self.wall_width = float(wall_width)
        self.wall_thickness = float(wall_thickness)
        self.wall_height = float(wall_height)
        self.wall_center_x = float(wall_center_x)
        self.wall_y = float(wall_y)
        self.goal_box_width = float(goal_box_width)
        self.goal_box_depth = float(goal_box_depth)
        self.goal_box_height = float(goal_box_height)
        self.goal_success_reward = float(goal_success_reward)
        self.reach_column_half_width = float(reach_column_half_width)
        self.reach_column_depth = float(reach_column_depth)
        self.reach_column_reward_weight = float(reach_column_reward_weight)
        self.horizontal_goal_at_wall_contact = bool(horizontal_goal_at_wall_contact)
        self.visualize_goal = bool(visualize_goal)
        self.horizontal_reward_weight = float(horizontal_reward_weight)
        self.height_reward_weight = float(height_reward_weight)

        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        if self.wall_width <= 0.0:
            raise ValueError(f"Expected wall_width > 0, got {self.wall_width}")
        if self.wall_thickness <= 0.0:
            raise ValueError(f"Expected wall_thickness > 0, got {self.wall_thickness}")
        if self.wall_height <= 0.0:
            raise ValueError(f"Expected wall_height > 0, got {self.wall_height}")
        if self.goal_box_width <= 0.0:
            raise ValueError(f"Expected goal_box_width > 0, got {self.goal_box_width}")
        if self.goal_box_depth <= 0.0:
            raise ValueError(f"Expected goal_box_depth > 0, got {self.goal_box_depth}")
        if self.goal_box_height <= 0.0:
            raise ValueError(f"Expected goal_box_height > 0, got {self.goal_box_height}")
        if self.reach_column_half_width <= 0.0:
            raise ValueError(f"Expected reach_column_half_width > 0, got {self.reach_column_half_width}")
        if self.reach_column_depth <= 0.0:
            raise ValueError(f"Expected reach_column_depth > 0, got {self.reach_column_depth}")
        if goal_center_z <= 0.0:
            raise ValueError(f"Expected goal_center_z > 0, got {goal_center_z}")
        goal_box_bottom_z = goal_center_z - self.goal_box_height / 2.0
        goal_box_top_z = goal_center_z + self.goal_box_height / 2.0
        if goal_box_bottom_z < 0.0 or goal_box_top_z > self.wall_height:
            raise ValueError("Expected the goal box to fit within the wall height")

        self.goal_position = np.array(
            [
                self.wall_center_x,
                self.wall_y - self.wall_thickness / 2.0 - self.goal_box_depth / 2.0,
                float(goal_center_z),
            ],
            dtype=float,
        )
        self.goal_box_half_size = np.array(
            [self.goal_box_width / 2.0, self.goal_box_depth / 2.0, self.goal_box_height / 2.0],
            dtype=float,
        )
        self.reach_column_min_y = self.wall_y - self.wall_thickness / 2.0 - self.reach_column_depth
        self.reach_column_max_y = self.wall_y - self.wall_thickness / 2.0
        self.horizontal_goal_position = self.goal_position.copy()
        if self.horizontal_goal_at_wall_contact:
            self.horizontal_goal_position[1] = self.reach_column_max_y

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
                "wall_width": self.wall_width,
                "wall_thickness": self.wall_thickness,
                "wall_height": self.wall_height,
                "wall_center_x": self.wall_center_x,
                "wall_y": self.wall_y,
                "goal_box_width": self.goal_box_width,
                "goal_box_depth": self.goal_box_depth,
                "goal_box_height": self.goal_box_height,
                "goal_center_z": float(self.goal_position[2]),
                "goal_success_reward": self.goal_success_reward,
                "reach_column_half_width": self.reach_column_half_width,
                "reach_column_depth": self.reach_column_depth,
                "reach_column_reward_weight": self.reach_column_reward_weight,
                "horizontal_goal_at_wall_contact": self.horizontal_goal_at_wall_contact,
                "global_obs_layout": VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
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
            name="VerticalReachWall",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.wall_width / 2.0, self.wall_thickness / 2.0, self.wall_height / 2.0],
            pos=[self.wall_center_x, self.wall_y, self.wall_height / 2.0],
            rgba=[0.55, 0.55, 0.58, 1.0],
        )
        if self.visualize_goal:
            worldbody.add_geom(
                name="VerticalReachGoal",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=self.goal_box_half_size.tolist(),
                pos=self.goal_position.tolist(),
                rgba=[0.1, 0.95, 0.35, 0.25],
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
        old_height_progress = state["height_progress"]
        old_reach_column_progress = state["reach_column_progress"]
        new_horizontal_progress, new_height_progress, new_reach_column_progress = self._compute_progress_baselines(
            data,
            state,
        )
        state["horizontal_progress"] = new_horizontal_progress
        state["height_progress"] = new_height_progress
        state["reach_column_progress"] = new_reach_column_progress

        horizontal_reward = (
            self.potential_reward_delta(new_horizontal_progress, old_horizontal_progress)
            * self.horizontal_reward_weight
        )
        height_reward = (
            self.potential_reward_delta(new_height_progress, old_height_progress)
            * self.height_reward_weight
        )
        reach_column_reward = (
            self.potential_reward_delta(new_reach_column_progress, old_reach_column_progress)
            * self.reach_column_reward_weight
        )
        progress_reward = horizontal_reward + height_reward + reach_column_reward

        state["progress"] = new_horizontal_progress + new_height_progress + new_reach_column_progress
        state["horizontal_reward"] = horizontal_reward
        state["height_reward"] = height_reward
        state["reach_column_reward"] = reach_column_reward
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
        reach_column_reward = state["reach_column_reward"]
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
        weighted_reach_column_reward = reach_column_reward * progress_reward_weight
        weighted_goal_success_reward = goal_success_reward * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_horizontal_reward"] = weighted_horizontal_reward
        state["weighted_height_reward"] = weighted_height_reward
        state["weighted_reach_column_reward"] = weighted_reach_column_reward
        state["weighted_goal_success_reward"] = weighted_goal_success_reward
        state["weighted_guidance_reward"] = weighted_guidance_reward
        state["reward_terms"] = {
            "horizontal": weighted_horizontal_reward,
            "height": weighted_height_reward,
            "reach_column": weighted_reach_column_reward,
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
        units_inside_goal = self._compute_units_inside_goal(unit_positions)
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            success = bool(units_inside_goal.any())
        else:
            active_units_mask = np.asarray(units_active_mask, dtype=bool)
            success = bool((units_inside_goal & active_units_mask).any())
        state["success"] = success
        return success

    def _reset_progress_baselines(self, data: mujoco.MjData, state: dict) -> None:
        horizontal_progress, height_progress, reach_column_progress = self._compute_progress_baselines(data, state)
        state["horizontal_progress"] = horizontal_progress
        state["height_progress"] = height_progress
        state["reach_column_progress"] = reach_column_progress
        state["progress"] = horizontal_progress + height_progress + reach_column_progress

    def _compute_progress_baselines(self, data: mujoco.MjData, state: dict) -> tuple[float, float, float]:
        unit_positions = np.asarray(data.qpos[self._qpos_indices[:, :3]], dtype=float)
        active_mask = state.get("units_active_mask")
        if active_mask is None:
            active_units_mask = np.ones((self.num_units,), dtype=bool)
        else:
            active_units_mask = np.asarray(active_mask, dtype=bool)
        horizontal_progress, height_progress = _compute_vertical_reach_progress_baselines_np(
            unit_positions=unit_positions,
            goal_position=self.goal_position,
            horizontal_goal_position=self.horizontal_goal_position,
            goal_box_half_size=self.goal_box_half_size,
            active_units_mask=active_units_mask,
            horizontal_reward_weight=self.horizontal_reward_weight,
            height_reward_weight=self.height_reward_weight,
        )
        reach_column_progress = _compute_reach_column_progress_baseline_np(
            unit_positions=unit_positions,
            goal_position=self.goal_position,
            active_units_mask=active_units_mask,
            reach_column_half_width=self.reach_column_half_width,
            reach_column_min_y=self.reach_column_min_y,
            reach_column_max_y=self.reach_column_max_y,
        )
        return horizontal_progress, height_progress, reach_column_progress

    def _compute_units_inside_goal(self, unit_positions: np.ndarray) -> np.ndarray:
        return np.all(np.abs(unit_positions - self.goal_position[np.newaxis, :]) <= self.goal_box_half_size, axis=1)


def _compute_vertical_reach_progress_baselines_np(
    *,
    unit_positions: np.ndarray,
    goal_position: np.ndarray,
    horizontal_goal_position: np.ndarray,
    goal_box_half_size: np.ndarray,
    active_units_mask: np.ndarray,
    horizontal_reward_weight: float,
    height_reward_weight: float,
) -> tuple[float, float]:
    if not np.asarray(active_units_mask, dtype=bool).any():
        return 0.0, 0.0

    horizontal_abs_delta = np.abs(unit_positions[:, :2] - horizontal_goal_position[np.newaxis, :2])
    horizontal_outside_distance = np.maximum(horizontal_abs_delta - goal_box_half_size[np.newaxis, :2], 0.0)
    horizontal_distance = np.linalg.norm(horizontal_outside_distance, axis=1)
    height_distance = np.maximum(np.abs(unit_positions[:, 2] - goal_position[2]) - goal_box_half_size[2], 0.0)
    weighted_distance = (
        horizontal_distance * float(horizontal_reward_weight)
        + height_distance * float(height_reward_weight)
    )
    weighted_distance = np.where(active_units_mask, weighted_distance, np.inf)
    best_unit_idx = int(np.argmin(weighted_distance))
    return -float(horizontal_distance[best_unit_idx]), -float(height_distance[best_unit_idx])


def _compute_reach_column_progress_baseline_np(
    *,
    unit_positions: np.ndarray,
    goal_position: np.ndarray,
    active_units_mask: np.ndarray,
    reach_column_half_width: float,
    reach_column_min_y: float,
    reach_column_max_y: float,
) -> float:
    active_units_mask = np.asarray(active_units_mask, dtype=bool)
    in_column = (
        active_units_mask
        & (np.abs(unit_positions[:, 0] - goal_position[0]) <= float(reach_column_half_width))
        & (unit_positions[:, 1] >= float(reach_column_min_y))
        & (unit_positions[:, 1] <= float(reach_column_max_y))
    )
    max_z = float(unit_positions[in_column, 2].max()) if in_column.any() else 0.0
    return -max(float(goal_position[2]) - max_z, 0.0)
