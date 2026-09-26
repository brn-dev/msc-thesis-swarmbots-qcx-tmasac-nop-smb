from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import (
    BoundedDistParams,
    FloatOrBoundedDistParams,
    FloatOrDistParams,
    eval_fodp,
)
from swarmbots.mj_env.scenarios.base_scenario import (
    BaseScenario,
    SwarmActDict,
    SwarmObsDict,
)
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class FindOpeningScenario(BaseScenario):
    def __init__(
            self,
            swarm: BaseSwarm,
            *,
            timestep: float = 0.002,
            action_repeat: int = 15,
            street_width: float = 10.0,
            wall_y: float = 2.0,
            wall_height: float = 2.0,
            wall_thickness: float = 0.2,
            wall_segment_width: float = 100.0,
            opening_width: float = 1.5,
            opening_x: FloatOrBoundedDistParams = 0.0,
            opening_y_margin: float = 1.0,
            success_reward: float = 5.0,
            opening_distance_reward_weight: float = 1.0,
            opening_distance_reward_falloff_distance: float | None = None,
            wall_exploration_cell_count: int = 0,
            wall_exploration_cell_reward: float = 0.0,
            wall_exploration_cell_depth: float = 1.0,
            actuator_strength: float = 8.0,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5,
            disconnect_potential_threshold: float = 5.0,
            friction: float | Iterable[float] | None = None,
            force_elliptic_cone: bool = False,
            progress_reward_weight: float = 1.0,
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
        self.street_width = float(street_width)
        self.wall_y = float(wall_y)
        self.wall_height = float(wall_height)
        self.wall_thickness = float(wall_thickness)
        self.wall_segment_width = float(wall_segment_width)
        self.opening_width = float(opening_width)
        self.opening_x_param = opening_x
        self.opening_y_margin = float(opening_y_margin)
        self.success_reward = float(success_reward)
        self.opening_distance_reward_weight = float(opening_distance_reward_weight)
        self.opening_distance_reward_falloff_distance = (
            None
            if opening_distance_reward_falloff_distance is None
            else float(opening_distance_reward_falloff_distance)
        )
        self.wall_exploration_cell_count = int(wall_exploration_cell_count)
        self.wall_exploration_cell_reward = float(wall_exploration_cell_reward)
        self.wall_exploration_cell_depth = float(wall_exploration_cell_depth)

        if self.street_width <= 0.0:
            raise ValueError(f"Expected street_width > 0, got {self.street_width}")
        if self.wall_height <= 0.0:
            raise ValueError(f"Expected wall_height > 0, got {self.wall_height}")
        if self.wall_thickness <= 0.0:
            raise ValueError(f"Expected wall_thickness > 0, got {self.wall_thickness}")
        if self.wall_segment_width < self.street_width:
            raise ValueError(
                f"Expected wall_segment_width >= street_width, got "
                f"{self.wall_segment_width} and {self.street_width}"
            )
        if not 0.0 < self.opening_width < self.street_width:
            raise ValueError(
                f"Expected 0 < opening_width < street_width, got {self.opening_width} and {self.street_width}"
            )
        if self.opening_y_margin <= 0.0:
            raise ValueError(f"Expected opening_y_margin > 0, got {self.opening_y_margin}")
        if (
            self.opening_distance_reward_falloff_distance is not None
            and self.opening_distance_reward_falloff_distance <= 0.0
        ):
            raise ValueError(
                "Expected opening_distance_reward_falloff_distance > 0 or None, "
                f"got {self.opening_distance_reward_falloff_distance}"
            )
        if self.wall_exploration_cell_count < 0:
            raise ValueError(f"Expected wall_exploration_cell_count >= 0, got {self.wall_exploration_cell_count}")
        if self.wall_exploration_cell_depth <= 0.0:
            raise ValueError(f"Expected wall_exploration_cell_depth > 0, got {self.wall_exploration_cell_depth}")

        self.side_wall_x = self.street_width / 2.0
        self._validate_opening_x()

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
            inactive_area_location=[self.street_width * 2.0, 0.0, 0.1],
            _reset_in_init=False,
        )

        self._dummy_state, self._dummy_connections = self.reset_scenario(
            self.dummy_model,
            self.dummy_data,
        )

    def _validate_opening_x(self) -> None:
        min_x = -self.side_wall_x + (self.opening_width / 2.0)
        max_x = self.side_wall_x - (self.opening_width / 2.0)
        if isinstance(self.opening_x_param, BoundedDistParams):
            if self.opening_x_param.low < min_x or self.opening_x_param.high > max_x:
                raise ValueError(
                    f"Expected opening_x bounds within [{min_x}, {max_x}], got {self.opening_x_param}"
                )
            return
        if not isinstance(self.opening_x_param, (int, float)):
            raise TypeError(
                f"Expected opening_x to be float or BoundedDistParams, got {self.opening_x_param!r}"
            )
        opening_x = float(self.opening_x_param)
        if opening_x < min_x or opening_x > max_x:
            raise ValueError(f"Expected opening_x within [{min_x}, {max_x}], got {opening_x}")

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
            "scenario_type": "find_opening",
            "street_width": self.street_width,
            "wall_y": self.wall_y,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "wall_segment_width": self.wall_segment_width,
            "opening_width": self.opening_width,
            "opening_x": self.opening_x_param,
            "opening_y_margin": self.opening_y_margin,
            "success_reward": self.success_reward,
            "opening_distance_reward_weight": self.opening_distance_reward_weight,
            "opening_distance_reward_falloff_distance": self.opening_distance_reward_falloff_distance,
            "wall_exploration_cell_count": self.wall_exploration_cell_count,
            "wall_exploration_cell_reward": self.wall_exploration_cell_reward,
            "wall_exploration_cell_depth": self.wall_exploration_cell_depth,
        })
        return settings

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: mujoco.MjsBody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[100, 100, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        side_wall_height = max(5.0, self.wall_height)
        for x in (-self.side_wall_x, self.side_wall_x):
            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.1, 100, side_wall_height / 2.0],
                rgba=[0.3, 0.4, 0.5, 0.1],
                pos=[x, 0, side_wall_height / 2.0],
            )

        barrier_body = worldbody.add_body(name="FindOpeningBarrier", mocap=True, pos=[0, 0, 0])
        segment_half_width = self.wall_segment_width / 2.0
        segment_center_offset = (self.opening_width / 2.0) + segment_half_width
        for name, x in (
            ("FindOpeningBarrier-left", -segment_center_offset),
            ("FindOpeningBarrier-right", segment_center_offset),
        ):
            barrier_body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[segment_half_width, self.wall_thickness / 2.0, self.wall_height / 2.0],
                rgba=[0.5, 0.5, 0.6, 1],
                pos=[x, 0, self.wall_height / 2.0],
            )
        return spec

    def reset_scenario(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)
        opening_x = float(eval_fodp(self.opening_x_param, self.rng))
        self._apply_opening_position(model=model, data=data, opening_x=opening_x)
        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state["opening_x"] = opening_x
        state["opening_potential"] = self._compute_opening_potential(
            data,
            state.get("units_active_mask"),
            opening_x,
        )
        state["wall_exploration_visited_cells"] = self._compute_wall_exploration_visited_cells(
            data,
            state.get("units_active_mask"),
        )
        state["hidden_global_vars"] = self._get_hidden_global_vars(state)
        state["success"] = False
        return state, connections

    def _apply_opening_position(
            self,
            *,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            opening_x: float,
    ) -> None:
        barrier_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "FindOpeningBarrier",
        )
        if barrier_body_id < 0:
            raise RuntimeError("Find-opening barrier body not found in model")
        mocap_id = int(model.body_mocapid[barrier_body_id])
        data.mocap_pos[mocap_id] = [opening_x, self.wall_y, 0.0]

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections,
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)
        hidden_global_vars = self._get_hidden_global_vars(state)
        state["hidden_global_vars"] = hidden_global_vars
        obs["hidden_global_vars"] = hidden_global_vars.copy()
        return obs

    def _get_hidden_global_vars(self, state: dict) -> np.ndarray:
        visited_cells = np.asarray(
            state.get(
                "wall_exploration_visited_cells",
                np.zeros((self.wall_exploration_cell_count,), dtype=bool),
            ),
            dtype=float,
        )
        return np.concatenate(
            (
                np.array([float(state["opening_x"])], dtype=float),
                visited_cells,
            )
        )

    def compute_progress_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        old_potential = float(state["opening_potential"])
        new_potential = self._compute_opening_potential(
            data,
            state.get("units_active_mask"),
            float(state["opening_x"]),
        )
        state["opening_potential"] = new_potential
        opening_reward = (
            self.potential_reward_delta(new_potential, old_potential)
            * self.opening_distance_reward_weight
        )
        exploration_reward = self._compute_wall_exploration_reward(data, state)
        state["opening_reward"] = opening_reward
        state["progress_reward"] = opening_reward + exploration_reward
        return state["progress_reward"]

    def evaluate_step(
            self,
            action: SwarmActDict,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections,
    ) -> tuple[float, bool]:
        reward, _ = super().evaluate_step(action, model, data, state, connections)
        success = self._compute_success(data, state.get("units_active_mask"))
        success_reward = self.success_reward if success else 0.0
        weighted_success_reward = success_reward * self.reward_weights["progress_reward_weight"]

        state["success"] = success
        state["success_reward"] = success_reward
        state["weighted_success_reward"] = weighted_success_reward
        state["progress_reward"] += success_reward
        state["weighted_progress_reward"] += weighted_success_reward
        weighted_opening_reward = (
            state["opening_reward"] * self.reward_weights["progress_reward_weight"]
        )
        weighted_exploration_reward = (
            state["wall_exploration_reward"] * self.reward_weights["progress_reward_weight"]
        )
        state["weighted_opening_reward"] = weighted_opening_reward
        state["weighted_wall_exploration_reward"] = weighted_exploration_reward
        state["weighted_units_without_connections_reward"] = state["weighted_guidance_reward"]
        state["reward_terms"] = {
            "opening": weighted_opening_reward,
            "exploration": weighted_exploration_reward,
            "success": weighted_success_reward,
            "units_without_connections": state["weighted_guidance_reward"],
        }
        return reward + weighted_success_reward, success

    def _compute_wall_exploration_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        visited_cells = np.asarray(
            state.get("wall_exploration_visited_cells"),
            dtype=bool,
        )
        current_cells = self._compute_wall_exploration_visited_cells(
            data,
            state.get("units_active_mask"),
        )
        if visited_cells.shape != current_cells.shape:
            visited_cells = np.zeros_like(current_cells)
        newly_visited_cells = current_cells & ~visited_cells
        state["wall_exploration_visited_cells"] = visited_cells | current_cells
        state["hidden_global_vars"] = self._get_hidden_global_vars(state)
        wall_exploration_reward = float(newly_visited_cells.sum()) * self.wall_exploration_cell_reward
        state["wall_exploration_reward"] = wall_exploration_reward
        state["wall_exploration_new_cells"] = int(newly_visited_cells.sum())
        return wall_exploration_reward

    def _compute_wall_exploration_visited_cells(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
    ) -> np.ndarray:
        if self.wall_exploration_cell_count == 0:
            return np.zeros((0,), dtype=bool)

        unit_x = np.asarray(data.qpos[self._qpos_indices[:, 0]], dtype=float)
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        if units_active_mask is None:
            active_mask = np.ones((self.num_units,), dtype=bool)
        else:
            active_mask = np.asarray(units_active_mask, dtype=bool)
        front_y = self.wall_y - (self.wall_thickness / 2.0)
        in_exploration_strip = (
            active_mask
            & (unit_x >= -self.side_wall_x)
            & (unit_x <= self.side_wall_x)
            & (unit_y >= front_y - self.wall_exploration_cell_depth)
            & (unit_y <= front_y)
        )
        visited_cells = np.zeros((self.wall_exploration_cell_count,), dtype=bool)
        if not in_exploration_strip.any():
            return visited_cells

        cell_width = self.street_width / float(self.wall_exploration_cell_count)
        cell_indices = np.floor((unit_x[in_exploration_strip] + self.side_wall_x) / cell_width).astype(int)
        cell_indices = np.clip(cell_indices, 0, self.wall_exploration_cell_count - 1)
        visited_cells[cell_indices] = True
        return visited_cells

    def _compute_opening_potential(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
            opening_x: float,
    ) -> float:
        unit_x = np.asarray(data.qpos[self._qpos_indices[:, 0]], dtype=float)
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        distance_to_waypoint = np.hypot(
            unit_x - float(opening_x),
            unit_y - self._opening_waypoint_y(),
        )
        shaped_distance = self._shape_opening_reward_distance(distance_to_waypoint)
        if units_active_mask is None:
            return -float(shaped_distance.mean())
        active_mask = np.asarray(units_active_mask, dtype=bool)
        if not active_mask.any():
            return 0.0
        return -float(shaped_distance[active_mask].mean())

    def _shape_opening_reward_distance(self, distance_to_waypoint: np.ndarray) -> np.ndarray:
        if self.opening_distance_reward_falloff_distance is None:
            return distance_to_waypoint
        falloff_distance = self.opening_distance_reward_falloff_distance
        return falloff_distance * np.log1p(distance_to_waypoint / falloff_distance)

    def _compute_success(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
    ) -> bool:
        unit_past_barrier = (
            np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
            > self._opening_waypoint_y()
        )
        if units_active_mask is None:
            return bool(unit_past_barrier.all())
        active_mask = np.asarray(units_active_mask, dtype=bool)
        return bool(active_mask.any() and unit_past_barrier[active_mask].all())

    def _opening_waypoint_y(self) -> float:
        return self.wall_y + self.opening_y_margin
