from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp
from swarmbots.mj_env.scenarios.base_scenario import (
    BaseScenario,
    SwarmActDict,
    SwarmObsDict,
)
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.utils.mujoco_render_geoms import add_wall_y_reference_line_geoms


def _wall_pass_rank_weights(active_units_count: int, skew: float) -> np.ndarray:
    if active_units_count <= 0:
        return np.zeros((0,), dtype=float)
    ranks = np.arange(1, active_units_count + 1, dtype=float)
    if skew == 0.0:
        return np.ones_like(ranks)
    unnormalized = np.power(ranks / float(active_units_count), float(skew))
    return unnormalized * (float(active_units_count) / float(unnormalized.sum()))


class WallScenario(BaseScenario):
    def __init__(
            self,
            swarm: BaseSwarm,
            *,
            timestep: float = 0.002,
            action_repeat: int = 15,
            wall_height: float = 0.5,
            wall_distance: FloatOrDistParams = 2.0,
            street_width: float = 10.0,
            actuator_strength: float = 8.0,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5,
            disconnect_potential_threshold: float = 5.0,
            friction: float | Iterable[float] | None = None,
            force_elliptic_cone: bool = False,
            progress_reward_weight: float = 1.0,
            forward_reward_weight: float = 1.0,
            forward_reward_wall_boost_factor: float = 1.0,
            forward_reward_wall_boost_distance: float | None = None,
            forward_reward_wall_boost_height_margin: float | None = None,
            guidance_reward_weight: float = 1.0,
            wall_pass_reward_weight: float = 0.0,
            wall_pass_reward_skew: float = 0.0,
            wall_pass_thresholds: list[float] | None = None,
            wall_success_threshold: float,
            wall_success_reward: float = 0.0,
            wall_climb_reward_weight: float = 0.0,
            wall_climb_reward_distance: float = 0.45,
            potential_reward_discount_factor: float = 1.0,
            units_without_connections_reward_weight: float = 0.0,
            include_connectors_xpos_in_obs: bool = True,
            include_connectors_xquat_in_obs: bool = False,
            quat_rot6d_representation: bool = True,
            reset_settle_time: float = 0,
            reset_settle_timestep_scale: float = 1.0,
            swarm_start_x: FloatOrDistParams = 0.0,
            swarm_start_y: FloatOrDistParams = 0.0,
            randomize_initial_swarm_z_rotation: bool = False,
            continuous_connector_actions: bool = True,
            seed: int | None = None,
    ) -> None:
        self.street_width = float(street_width)
        self.side_wall_x = self.street_width / 2.0
        self.wall_height = float(wall_height)
        self.wall_distance = wall_distance
        self.forward_reward_weight = float(forward_reward_weight)
        self.forward_reward_wall_boost_factor = float(forward_reward_wall_boost_factor)
        self.forward_reward_wall_boost_distance = (
            None
            if forward_reward_wall_boost_distance is None
            else float(forward_reward_wall_boost_distance)
        )
        self.forward_reward_wall_boost_height_margin = (
            None
            if forward_reward_wall_boost_height_margin is None
            else float(forward_reward_wall_boost_height_margin)
        )
        if self.forward_reward_wall_boost_factor < 1.0:
            raise ValueError(
                f"Expected forward_reward_wall_boost_factor >= 1.0, got {self.forward_reward_wall_boost_factor}"
            )
        if self.forward_reward_wall_boost_distance is not None and self.forward_reward_wall_boost_distance <= 0.0:
            raise ValueError(
                f"Expected forward_reward_wall_boost_distance > 0, got {self.forward_reward_wall_boost_distance}"
            )
        if self.forward_reward_wall_boost_height_margin is not None and self.forward_reward_wall_boost_height_margin < 0.0:
            raise ValueError(
                "Expected forward_reward_wall_boost_height_margin >= 0, "
                f"got {self.forward_reward_wall_boost_height_margin}"
            )
        self.wall_pass_reward_weight = float(wall_pass_reward_weight)
        self.wall_pass_reward_skew = float(wall_pass_reward_skew)
        self.wall_climb_reward_weight = float(wall_climb_reward_weight)
        self.wall_climb_reward_distance = float(wall_climb_reward_distance)
        if self.wall_climb_reward_distance <= 0.0:
            raise ValueError(f"Expected wall_climb_reward_distance > 0, got {self.wall_climb_reward_distance}")
        if wall_pass_thresholds is None:
            wall_pass_thresholds = [0.0]
        self.wall_pass_thresholds = np.sort(wall_pass_thresholds)
        self.wall_success_threshold = float(wall_success_threshold)
        self.wall_success_reward = float(wall_success_reward)
        if self.wall_success_threshold <= 0.0:
            raise ValueError(f"Expected wall_success_threshold > 0, got {self.wall_success_threshold}")

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

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
            "wall_height": self.wall_height,
            "wall_distance": self.wall_distance,
            "street_width": self.street_width,
            "forward_reward_weight": self.forward_reward_weight,
            "forward_reward_wall_boost_factor": self.forward_reward_wall_boost_factor,
            "forward_reward_wall_boost_distance": self.forward_reward_wall_boost_distance,
            "forward_reward_wall_boost_height_margin": self.forward_reward_wall_boost_height_margin,
            "wall_pass_reward_weight": self.wall_pass_reward_weight,
            "wall_pass_reward_skew": self.wall_pass_reward_skew,
            "wall_pass_thresholds": self.wall_pass_thresholds.tolist(),
            "wall_success_threshold": self.wall_success_threshold,
            "wall_success_reward": self.wall_success_reward,
            "wall_climb_reward_weight": self.wall_climb_reward_weight,
            "wall_climb_reward_distance": self.wall_climb_reward_distance,
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

        wall_body = worldbody.add_body(name="Wall", mocap=True, pos=[0, 0, 0])
        wall_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.side_wall_x, 0.1, self.wall_height],
            rgba=[0.5, 0.5, 0.6, 1],
        )
        return spec

    def reset_scenario(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)

        hidden_global_vars: list[float] = []
        wall_y = self.reset_wall(data, model, hidden_global_vars)

        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state["hidden_global_vars"] = np.array(hidden_global_vars, dtype=float)
        state["wall_y"] = wall_y
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        state["wall_climb_done_mask"] = unit_y > wall_y
        state["wall_climb_potential"] = (
            np.zeros((self.num_units,), dtype=np.float32)
            if self.wall_climb_reward_weight == 0.0
            else self._compute_wall_climb_potential(data, state)
        )
        wall_pass_absolute_thresholds = self._compute_wall_pass_thresholds(wall_y)
        state["wall_pass_absolute_thresholds"] = wall_pass_absolute_thresholds
        passed_thresholds_mask = unit_y[:, np.newaxis] > wall_pass_absolute_thresholds[np.newaxis, :]
        state["passed_thresholds_mask"] = passed_thresholds_mask
        state["next_threshold_for_unit"] = passed_thresholds_mask.sum(axis=1).astype(int)
        state["num_wall_thresholds_passed"] = 0
        state["wall_thresholds_passed_reward"] = 0.0
        forward_progress_unit_y = self._compute_forward_reward_unit_potential(data, state)
        state["progress"] = self._mean_active_forward_progress(forward_progress_unit_y, state.get("units_active_mask"))
        state["forward_progress_unit_y"] = forward_progress_unit_y

        return state, connections

    def add_render_geoms(self, scene: mujoco.MjvScene) -> None:
        add_wall_y_reference_line_geoms(scene, wall_distance=self.wall_distance)

    def reset_wall(
            self,
            data: mujoco.MjData,
            model: mujoco.MjModel,
            hidden_global_vars: list[float],
    ) -> float:
        wall_y = eval_fodp(self.wall_distance, self.rng)
        hidden_global_vars.append(wall_y)
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Wall")
        mocap_id = model.body_mocapid[wall_id]
        data.mocap_pos[mocap_id] = [0.0, wall_y, 0.0]
        return float(wall_y)

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections,
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)
        obs["hidden_global_vars"] = state["hidden_global_vars"].copy()
        obs["hidden_local_vars"] = np.asarray(state["passed_thresholds_mask"], dtype=float).copy()
        return obs

    def _compute_wall_pass_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        wall_pass_absolute_thresholds = np.asarray(state.get("wall_pass_absolute_thresholds"), dtype=float)
        next_threshold_for_unit = np.asarray(state.get("next_threshold_for_unit"), dtype=int)
        if wall_pass_absolute_thresholds.size == 0 or next_threshold_for_unit.shape != (self.num_units,):
            state["num_wall_thresholds_passed"] = 0
            state["wall_thresholds_passed_reward"] = 0.0
            return 0.0

        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        units_active_mask = state.get("units_active_mask")
        active_units_mask = None if units_active_mask is None else np.asarray(units_active_mask, dtype=bool)
        active_units_count = self.num_units if active_units_mask is None else int(active_units_mask.sum())
        num_wall_thresholds = int(self.wall_pass_thresholds.size)

        num_wall_thresholds_passed = 0
        wall_pass_reward_units = 0.0
        if self.wall_pass_reward_skew != 0.0:
            thresholds_passed_by_active_units = np.zeros(wall_pass_absolute_thresholds.size, dtype=int)
            if active_units_count > 0:
                active_next_thresholds = (
                    next_threshold_for_unit if active_units_mask is None else next_threshold_for_unit[active_units_mask]
                )
                for next_threshold_idx in active_next_thresholds:
                    thresholds_passed_by_active_units[:int(next_threshold_idx)] += 1
            rank_weights = _wall_pass_rank_weights(active_units_count, self.wall_pass_reward_skew)

        for unit_idx in range(self.num_units):
            if active_units_mask is not None and not active_units_mask[unit_idx]:
                continue
            next_threshold_idx = int(next_threshold_for_unit[unit_idx])
            while (next_threshold_idx < wall_pass_absolute_thresholds.size
                   and unit_y[unit_idx] > wall_pass_absolute_thresholds[next_threshold_idx]):
                if self.wall_pass_reward_skew == 0.0:
                    wall_pass_reward_units += 1.0
                else:
                    next_rank = thresholds_passed_by_active_units[next_threshold_idx]
                    wall_pass_reward_units += float(rank_weights[next_rank])
                    thresholds_passed_by_active_units[next_threshold_idx] = next_rank + 1
                next_threshold_idx += 1
                num_wall_thresholds_passed += 1
            next_threshold_for_unit[unit_idx] = next_threshold_idx

        if active_units_count > 0 and num_wall_thresholds > 0:
            wall_thresholds_passed_reward = (
                wall_pass_reward_units / (active_units_count * num_wall_thresholds)
            ) * self.wall_pass_reward_weight
        else:
            wall_thresholds_passed_reward = 0.0
        state["next_threshold_for_unit"] = next_threshold_for_unit
        if wall_pass_absolute_thresholds.size > 0:
            unit_threshold_indices = np.arange(wall_pass_absolute_thresholds.size, dtype=int)
            state["passed_thresholds_mask"] = (
                unit_threshold_indices[np.newaxis, :] < next_threshold_for_unit[:, np.newaxis]
            )
        else:
            state["passed_thresholds_mask"] = np.zeros((self.num_units, 0), dtype=bool)
        state["num_wall_thresholds_passed"] = num_wall_thresholds_passed
        state["wall_thresholds_passed_reward"] = wall_thresholds_passed_reward

        return wall_thresholds_passed_reward

    def _compute_wall_success_termination(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> bool:
        wall_y = state.get("wall_y")
        if wall_y is None:
            state["success"] = False
            return False

        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        unit_past_wall = unit_y > (float(wall_y) + self.wall_success_threshold)
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            success = bool(unit_past_wall.all())
        else:
            active_units_mask = np.asarray(units_active_mask, dtype=bool)
            success = bool(active_units_mask.any() and unit_past_wall[active_units_mask].all())
        state["success"] = success
        return success

    def _compute_wall_climb_potential(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> np.ndarray:
        wall_y = state.get("wall_y")
        if wall_y is None:
            return np.zeros((self.num_units,), dtype=np.float32)

        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        unit_z = np.asarray(data.qpos[self._qpos_indices[:, 2]], dtype=float)
        distance_to_wall = float(wall_y) - unit_y
        approach = np.clip(1.0 - (distance_to_wall / self.wall_climb_reward_distance), 0.0, 1.0)
        approach = np.where(
            (distance_to_wall >= 0.0) & (distance_to_wall <= self.wall_climb_reward_distance),
            approach,
            0.0,
        )
        approach = np.sqrt(approach)
        unit_ground_z = float(getattr(self.swarm, "body_radius", 0.1))
        target_lift = max(self.wall_height - unit_ground_z, 1e-6)
        height = np.clip((unit_z - unit_ground_z) / target_lift, 0.0, 1.0)
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            active_mask = np.ones((self.num_units,), dtype=np.float32)
        else:
            active_mask = np.asarray(units_active_mask, dtype=bool).astype(np.float32)
        return (approach * height * active_mask).astype(np.float32, copy=False)

    def _compute_wall_climb_done_mask(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> np.ndarray:
        wall_y = state.get("wall_y")
        if wall_y is None:
            return np.zeros((self.num_units,), dtype=bool)

        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        done_mask = unit_y > float(wall_y)
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is not None:
            done_mask &= np.asarray(units_active_mask, dtype=bool)
        return done_mask

    def _compute_wall_climb_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        if getattr(self, "wall_climb_reward_weight", 0.0) == 0.0:
            state["wall_climb_reward"] = 0.0
            return 0.0

        previous_potential = np.asarray(state.get("wall_climb_potential"), dtype=np.float32)
        current_potential = self._compute_wall_climb_potential(data, state)
        if previous_potential.shape != current_potential.shape:
            previous_potential = np.zeros_like(current_potential)

        previous_done_mask = np.asarray(state.get("wall_climb_done_mask"), dtype=bool)
        if previous_done_mask.shape != current_potential.shape:
            previous_done_mask = np.zeros(current_potential.shape, dtype=bool)
        crossed_wall_y_mask = self._compute_wall_climb_done_mask(data, state)
        if crossed_wall_y_mask.shape != current_potential.shape:
            crossed_wall_y_mask = np.zeros(current_potential.shape, dtype=bool)
        newly_done_mask = crossed_wall_y_mask & ~previous_done_mask
        done_mask = previous_done_mask | newly_done_mask
        current_potential = np.where(done_mask, 0.0, current_potential).astype(np.float32, copy=False)
        wall_climb_delta = self.potential_reward_delta(current_potential, previous_potential)
        wall_climb_delta = np.where(
            newly_done_mask & (wall_climb_delta < 0.0),
            0.0,
            wall_climb_delta,
        )
        state["wall_climb_potential"] = current_potential
        state["wall_climb_done_mask"] = done_mask

        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            climb_delta = float(wall_climb_delta.mean())
        else:
            active_mask = np.asarray(units_active_mask, dtype=bool)
            active_units_count = int(active_mask.sum())
            climb_delta = (
                float(wall_climb_delta[active_mask].mean())
                if active_units_count > 0
                else 0.0
            )

        wall_climb_reward = climb_delta * self.wall_climb_reward_weight
        state["wall_climb_reward"] = wall_climb_reward
        return wall_climb_reward

    def compute_progress_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        old_progress = state["progress"]
        current_unit_y = self._compute_forward_reward_unit_potential(data, state)
        new_progress = self._mean_active_forward_progress(current_unit_y, state.get("units_active_mask"))
        state["progress"] = new_progress

        previous_unit_y = np.asarray(state.get("forward_progress_unit_y"), dtype=float)
        if previous_unit_y.shape == current_unit_y.shape:
            unit_forward_delta = self.potential_reward_delta(current_unit_y, previous_unit_y)
            forward_reward = self._mean_active_forward_delta(unit_forward_delta, state.get("units_active_mask"))
        else:
            forward_reward = self.potential_reward_delta(new_progress, old_progress)
        state["forward_progress_unit_y"] = current_unit_y
        wall_pass_reward = self._compute_wall_pass_reward(data, state)
        wall_climb_reward = self._compute_wall_climb_reward(data, state)
        progress_reward = forward_reward * self.forward_reward_weight + wall_pass_reward + wall_climb_reward

        state["forward_reward"] = forward_reward
        state["progress_reward"] = progress_reward
        state["wall_pass_reward"] = wall_pass_reward
        state["wall_climb_reward"] = wall_climb_reward
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

        self.compute_progress_reward(data, state)
        forward_reward = state["forward_reward"]
        wall_pass_reward = state["wall_pass_reward"]
        wall_climb_reward = state["wall_climb_reward"]
        terminated = self._compute_wall_success_termination(data, state)
        wall_success_reward = self.wall_success_reward if terminated else 0.0
        progress_reward = state["progress_reward"] + wall_success_reward
        state["progress_reward"] = progress_reward
        state["wall_success_reward"] = wall_success_reward

        units_without_connections_reward = super().compute_guidance_reward(data, action, state, connections)
        state["units_without_connections_reward"] = units_without_connections_reward
        state["guidance_reward"] = units_without_connections_reward

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_forward_reward = forward_reward * self.forward_reward_weight * progress_reward_weight
        weighted_wall_pass_reward = wall_pass_reward * progress_reward_weight
        weighted_wall_climb_reward = wall_climb_reward * progress_reward_weight
        weighted_wall_success_reward = wall_success_reward * progress_reward_weight
        weighted_units_without_connections_reward = (
            units_without_connections_reward * self.reward_weights["guidance_reward_weight"]
        )
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_forward_reward"] = weighted_forward_reward
        state["weighted_forward_progress_reward"] = weighted_forward_reward
        state["weighted_wall_pass_reward"] = weighted_wall_pass_reward
        state["weighted_wall_climb_reward"] = weighted_wall_climb_reward
        state["weighted_wall_success_reward"] = weighted_wall_success_reward
        state["weighted_units_without_connections_reward"] = weighted_units_without_connections_reward
        state["weighted_guidance_reward"] = weighted_units_without_connections_reward
        state["reward_terms"] = {
            "forward": weighted_forward_reward,
            "wall": weighted_wall_pass_reward,
            "climb": weighted_wall_climb_reward,
            "success": weighted_wall_success_reward,
            "units_without_connections": weighted_units_without_connections_reward,
        }
        return weighted_progress_reward + weighted_units_without_connections_reward, terminated

    def _compute_wall_pass_thresholds(self, wall_y: float) -> np.ndarray:
        return float(wall_y) + self.wall_pass_thresholds

    def _compute_forward_progress_baseline(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
            state: dict,
    ) -> float:
        unit_y = self._compute_forward_progress_unit_y(data, state)
        return self._mean_active_forward_progress(unit_y, units_active_mask)

    def _compute_forward_progress_unit_y(self, data: mujoco.MjData, state: dict) -> np.ndarray:
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        return np.minimum(unit_y, self._forward_reward_cap_y(state))

    def _compute_forward_reward_unit_potential(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> np.ndarray:
        unit_y = self._compute_forward_progress_unit_y(data, state)
        return self._apply_forward_reward_wall_boost_to_potential(unit_y, data, state)

    def _forward_reward_cap_y(self, state: dict) -> float:
        return float(state["wall_y"]) + self.wall_success_threshold

    def _mean_active_forward_progress(
            self,
            unit_y: np.ndarray,
            units_active_mask: np.ndarray | None,
    ) -> float:
        if units_active_mask is None:
            return float(unit_y.mean())
        active_units_mask = np.asarray(units_active_mask, dtype=bool)
        if not active_units_mask.any():
            return 0.0
        return float(unit_y[active_units_mask].mean())

    def _mean_active_forward_delta(
            self,
            unit_forward_delta: np.ndarray,
            units_active_mask: np.ndarray | None,
    ) -> float:
        if units_active_mask is None:
            return float(unit_forward_delta.mean())
        active_units_mask = np.asarray(units_active_mask, dtype=bool)
        if not active_units_mask.any():
            return 0.0
        return float(unit_forward_delta[active_units_mask].mean())

    def _apply_forward_reward_wall_boost_to_potential(
            self,
            unit_forward_potential: np.ndarray,
            data: mujoco.MjData,
            state: dict,
    ) -> np.ndarray:
        boost_factor = float(getattr(self, "forward_reward_wall_boost_factor", 1.0))
        if boost_factor == 1.0:
            return unit_forward_potential

        wall_y = state.get("wall_y")
        if wall_y is None:
            return unit_forward_potential

        qpos_indices = np.asarray(self._qpos_indices)
        unit_y = np.asarray(data.qpos[qpos_indices[:, 1]], dtype=float)
        unit_z = np.asarray(data.qpos[qpos_indices[:, 2]], dtype=float)
        boost_mask = _compute_forward_reward_wall_boost_mask_np(
            unit_y=unit_y,
            unit_z=unit_z,
            wall_y=float(wall_y),
            wall_height=self.wall_height,
            boost_distance=self._forward_reward_wall_boost_distance(),
            height_margin=self._forward_reward_wall_boost_height_margin(),
            wall_half_thickness=0.1,
        )
        return np.where(boost_mask, unit_forward_potential * boost_factor, unit_forward_potential)

    def _forward_reward_wall_boost_distance(self) -> float:
        configured = getattr(self, "forward_reward_wall_boost_distance", None)
        if configured is not None:
            return float(configured)
        return float(getattr(self, "wall_climb_reward_distance", 0.45))

    def _forward_reward_wall_boost_height_margin(self) -> float:
        configured = getattr(self, "forward_reward_wall_boost_height_margin", None)
        if configured is not None:
            return float(configured)
        return float(getattr(self.swarm, "body_radius", 0.1))


def _compute_forward_reward_wall_boost_mask_np(
        *,
        unit_y: np.ndarray,
        unit_z: np.ndarray,
        wall_y: float,
        wall_height: float,
        boost_distance: float,
        height_margin: float,
        wall_half_thickness: float,
) -> np.ndarray:
    distance_to_wall = float(wall_y) - unit_y
    near_wall = (distance_to_wall >= -float(wall_half_thickness)) & (distance_to_wall <= float(boost_distance))
    high_enough = unit_z >= (float(wall_height) + float(height_margin))
    return near_wall & high_enough
