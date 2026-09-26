from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.scenario_presets.move_to_goal_config import AbsoluteGoalConfig, MoveToGoalConfig, sample_move_to_goal_position


class MoveToScenario(BaseScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        goal: MoveToGoalConfig = AbsoluteGoalConfig(),
        goal_radius: float = 0.25,
        visualize_goal: bool = False,
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        forward_reward_weight: float = 1.0,
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
        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")

        self.goal = goal
        self.goal_radius = float(goal_radius)
        if self.goal_radius < 0.0:
            raise ValueError(f"Expected goal_radius >= 0, got {self.goal_radius}")
        self.visualize_goal = bool(visualize_goal)

        self.forward_reward_weight = float(forward_reward_weight)

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
                "goal": self.goal,
                "goal_radius": self.goal_radius,
                "visualize_goal": self.visualize_goal,
                "forward_reward_weight": self.forward_reward_weight,
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
        return spec

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)
        goal_position = self._sample_goal_position(swarm_start_location=state["swarm_start_location"])
        self._set_goal_marker(model, data, goal_position)
        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state["goal_position"] = goal_position
        state["progress"] = self._compute_goal_progress_baseline(
            data=data,
            units_active_mask=state.get("units_active_mask"),
            goal_position=goal_position,
        )
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
        old_progress = state["progress"]
        new_progress = self._compute_goal_progress_baseline(
            data=data,
            units_active_mask=state.get("units_active_mask"),
            goal_position=np.asarray(state["goal_position"], dtype=float),
        )
        state["progress"] = new_progress

        forward_reward = self.potential_reward_delta(new_progress, old_progress) * self.forward_reward_weight
        state["forward_reward"] = forward_reward
        state["progress_reward"] = forward_reward
        return forward_reward

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
        forward_reward = state["forward_reward"]

        guidance_reward = super().compute_guidance_reward(data, action, state, connections)
        state["guidance_reward"] = guidance_reward

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_forward_reward = forward_reward * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_forward_reward"] = weighted_forward_reward
        state["weighted_forward_progress_reward"] = weighted_forward_reward
        state["weighted_guidance_reward"] = weighted_guidance_reward
        state["reward_terms"] = {
            "forward": weighted_forward_reward,
            "guidance": weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, False

    def _sample_goal_position(self, *, swarm_start_location: np.ndarray) -> np.ndarray:
        return sample_move_to_goal_position(
            goal=self.goal,
            rng=self.rng,
            swarm_start_location=swarm_start_location,
        )

    def _set_goal_marker(self, model: mujoco.MjModel, data: mujoco.MjData, goal_position: np.ndarray) -> None:
        goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Goal")
        goal_mocap_id = model.body_mocapid[goal_body_id]
        data.mocap_pos[goal_mocap_id] = [float(goal_position[0]), float(goal_position[1]), 0.0]

    def _compute_goal_progress_baseline(
        self,
        *,
        data: mujoco.MjData,
        units_active_mask: np.ndarray | None,
        goal_position: np.ndarray,
    ) -> float:
        unit_xy = np.asarray(data.qpos[self._qpos_indices[:, :2]], dtype=float)
        distance_to_goal = np.linalg.norm(unit_xy - goal_position[np.newaxis, :], axis=1)
        remaining_distance = np.maximum(distance_to_goal - self.goal_radius, 0.0)
        if units_active_mask is None:
            return -float(remaining_distance.mean())

        active_mask = np.asarray(units_active_mask, dtype=bool)
        active_units_count = int(active_mask.sum())
        if active_units_count == 0:
            return 0.0
        return -float(remaining_distance[active_mask].mean())
