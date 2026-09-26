from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import mujoco
import numpy as np

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadShape
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.scenario_presets.multi_payload_goal import (
    PAYLOAD_GOAL_COLORS,
    PAYLOAD_GOAL_IDENTITY_ROT6D,
    PAYLOAD_GOAL_OBS_RECORD_DIM,
    normalize_count_probs,
)
from swarmbots.scenario_presets.scenario_obs_layouts import MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT


class MultiPayloadGoalScenario(BaseScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        max_payloads: int = 4,
        active_payload_count_probs: Mapping[int, float] | None = None,
        payload_shape: PayloadShape = "box",
        payload_radius: float = 0.2,
        payload_mass: float = 1.0,
        payload_spawn_y: float = 1.0,
        payload_spawn_margin: float = 0.7,
        goal_rect: tuple[float, float, float, float] = (-1.5, 1.5, 2.2, 4.0),
        goal_radius: float = 0.35,
        visualize_goal: bool = True,
        success_reward: float = 5.0,
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
        self.max_payloads = int(max_payloads)
        self.payload_shape: PayloadShape = payload_shape
        self.payload_radius = float(payload_radius)
        self.payload_mass = float(payload_mass)
        self.payload_spawn_y = float(payload_spawn_y)
        self.payload_spawn_margin = float(payload_spawn_margin)
        self.goal_rect = tuple(float(value) for value in goal_rect)
        self.goal_radius = float(goal_radius)
        self.visualize_goal = bool(visualize_goal)
        self.success_reward = float(success_reward)
        self.forward_reward_weight = float(forward_reward_weight)

        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        if self.max_payloads < 1:
            raise ValueError(f"Expected max_payloads >= 1, got {self.max_payloads}")
        if self.payload_shape not in ("sphere", "box", "capsule"):
            raise ValueError(
                f"Expected payload_shape to be 'sphere', 'box', or 'capsule', got {self.payload_shape!r}"
            )
        if self.payload_radius <= 0.0:
            raise ValueError(f"Expected payload_radius > 0, got {self.payload_radius}")
        if self.payload_mass <= 0.0:
            raise ValueError(f"Expected payload_mass > 0, got {self.payload_mass}")
        if self.payload_spawn_margin < 2.0 * self.payload_radius:
            raise ValueError(
                "payload_spawn_margin must be at least two payload radii, "
                f"got {self.payload_spawn_margin} for radius {self.payload_radius}"
            )
        if len(self.goal_rect) != 4:
            raise ValueError("goal_rect must be (x_min, x_max, y_min, y_max)")
        goal_x_min, goal_x_max, goal_y_min, goal_y_max = self.goal_rect
        if goal_x_min >= goal_x_max or goal_y_min >= goal_y_max:
            raise ValueError(f"Invalid goal_rect bounds: {self.goal_rect!r}")
        if self.goal_radius < 0.0:
            raise ValueError(f"Expected goal_radius >= 0, got {self.goal_radius}")

        self.active_payload_count_probs = normalize_count_probs(
            probs=active_payload_count_probs,
            max_payloads=self.max_payloads,
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
            _reset_in_init=False,
        )

        self._payload_names = tuple(f"Payload{payload_idx}" for payload_idx in range(self.max_payloads))
        self._goal_names = tuple(f"PayloadGoal{payload_idx}" for payload_idx in range(self.max_payloads))
        self._payload_qpos_indices = np.stack(
            [
                np.asarray(mj_utils.qpos_indices_for_body(self.dummy_model, name), dtype=int)
                for name in self._payload_names
            ],
            axis=0,
        )
        self._payload_qvel_indices = [
            np.asarray(mj_utils.dof_indices_for_body(self.dummy_model, name), dtype=int)
            for name in self._payload_names
        ]
        if self._payload_qpos_indices.shape != (self.max_payloads, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            {
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

        for payload_idx in range(self.max_payloads):
            payload_body = worldbody.add_body(
                name=self._payload_body_name(payload_idx),
                pos=[0, 0, self.payload_radius],
            )
            payload_body.add_freejoint(name=f"{self._payload_body_name(payload_idx)}_freejoint")
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

        return spec

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)
        active_payload_mask = self._sample_active_payload_mask()
        goal_positions = self._sample_goal_positions()
        self._reset_payloads(data=data, state=state, active_payload_mask=active_payload_mask)
        self._set_goal_markers(
            model=model,
            data=data,
            goal_positions=goal_positions,
            active_payload_mask=active_payload_mask,
        )
        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state["active_payload_mask"] = active_payload_mask
        state["goal_positions"] = goal_positions
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
        obs["global_obs"] = self._get_payload_goal_obs(data, state)
        obs["hidden_global_vars"] = np.zeros((0,), dtype=float)
        return obs

    def compute_progress_reward(self, data: mujoco.MjData, state: dict) -> float:
        old_payload_progress = np.asarray(state["payload_progress"], dtype=float)
        new_payload_progress = self._compute_payload_goal_progress(data=data, state=state)
        state["payload_progress"] = new_payload_progress
        state["progress"] = float(new_payload_progress.sum())

        forward_reward = float(
            self.potential_reward_delta(new_payload_progress, old_payload_progress).sum()
            * self.forward_reward_weight
        )
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
        terminated = self._compute_success_termination(data, state)
        goal_success_reward = self.success_reward if terminated else 0.0
        progress_reward += goal_success_reward
        state["goal_success_reward"] = goal_success_reward
        state["progress_reward"] = progress_reward

        guidance_reward = super().compute_guidance_reward(data, action, state, connections)
        state["guidance_reward"] = guidance_reward

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_forward_reward = forward_reward * progress_reward_weight
        weighted_goal_success_reward = goal_success_reward * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_forward_reward"] = weighted_forward_reward
        state["weighted_forward_progress_reward"] = weighted_forward_reward
        state["weighted_goal_success_reward"] = weighted_goal_success_reward
        state["weighted_guidance_reward"] = weighted_guidance_reward
        state["reward_terms"] = {
            "forward": weighted_forward_reward,
            "success": weighted_goal_success_reward,
            "guidance": weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, terminated

    def _reset_progress_baselines(self, data: mujoco.MjData, state: dict) -> None:
        payload_progress = self._compute_payload_goal_progress(data=data, state=state)
        state["payload_progress"] = payload_progress
        state["progress"] = float(payload_progress.sum())

    def _sample_active_payload_mask(self) -> np.ndarray:
        counts = list(self.active_payload_count_probs.keys())
        probabilities = list(self.active_payload_count_probs.values())
        num_active_payloads = int(self.rng.choice(counts, p=probabilities))
        active_payload_mask = np.zeros(self.max_payloads, dtype=bool)
        active_payload_indices = self.rng.choice(
            self.max_payloads,
            size=num_active_payloads,
            replace=False,
        )
        active_payload_mask[active_payload_indices] = True
        return active_payload_mask

    def _sample_goal_positions(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.goal_rect
        goal_positions = np.zeros((self.max_payloads, 2), dtype=float)
        goal_positions[:, 0] = self.rng.uniform(x_min, x_max, size=self.max_payloads)
        goal_positions[:, 1] = self.rng.uniform(y_min, y_max, size=self.max_payloads)
        return goal_positions

    def _reset_payloads(
        self,
        *,
        data: mujoco.MjData,
        state: dict,
        active_payload_mask: np.ndarray,
    ) -> None:
        swarm_start_location = np.asarray(state["swarm_start_location"], dtype=float)
        payload_positions = np.zeros((self.max_payloads, 3), dtype=float)
        active_payload_indices = np.flatnonzero(active_payload_mask)
        slot_x_positions = self._payload_spawn_x_positions(active_payload_indices.size)
        self.rng.shuffle(slot_x_positions)

        inactive_base = np.asarray(self.inactive_area_location, dtype=float)
        for slot_idx, payload_idx in enumerate(active_payload_indices):
            payload_position = np.array(
                [
                    swarm_start_location[0] + slot_x_positions[slot_idx],
                    swarm_start_location[1] + self.payload_spawn_y,
                    self.payload_radius,
                ],
                dtype=float,
            )
            payload_positions[payload_idx] = payload_position
            self._set_payload_pose(data=data, payload_idx=int(payload_idx), position=payload_position)

        for payload_idx in np.flatnonzero(~active_payload_mask):
            payload_position = inactive_base + np.array(
                [float(payload_idx) * self.payload_spawn_margin, 0.0, self.payload_radius],
                dtype=float,
            )
            payload_positions[payload_idx] = payload_position
            self._set_payload_pose(data=data, payload_idx=int(payload_idx), position=payload_position)

        state["payload_spawn_position"] = payload_positions

    def _set_payload_pose(self, *, data: mujoco.MjData, payload_idx: int, position: np.ndarray) -> None:
        data.qpos[self._payload_qpos_indices[payload_idx, :3]] = position
        data.qpos[self._payload_qpos_indices[payload_idx, 3:7]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        qvel_indices = self._payload_qvel_indices[payload_idx]
        if qvel_indices.size > 0:
            data.qvel[qvel_indices] = 0.0

    def _set_goal_markers(
        self,
        *,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        goal_positions: np.ndarray,
        active_payload_mask: np.ndarray,
    ) -> None:
        inactive_base = np.asarray(self.inactive_area_location, dtype=float)
        for payload_idx in range(self.max_payloads):
            goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self._goal_body_name(payload_idx))
            goal_mocap_id = int(model.body_mocapid[goal_body_id])
            if active_payload_mask[payload_idx]:
                goal_xy = goal_positions[payload_idx]
                data.mocap_pos[goal_mocap_id] = [float(goal_xy[0]), float(goal_xy[1]), 0.0]
            else:
                data.mocap_pos[goal_mocap_id] = inactive_base

    def _compute_payload_goal_progress(self, *, data: mujoco.MjData, state: dict) -> np.ndarray:
        active_payload_mask = np.asarray(state["active_payload_mask"], dtype=bool)
        goal_positions = self._goal_positions_xyz(state)
        payload_positions = self._get_payload_positions(data)
        distance_to_goal = np.linalg.norm(payload_positions - goal_positions, axis=1)
        remaining_distance = np.maximum(distance_to_goal - self.goal_radius, 0.0)
        progress = np.zeros(self.max_payloads, dtype=float)
        progress[active_payload_mask] = -remaining_distance[active_payload_mask]
        return progress

    def _compute_success_termination(self, data: mujoco.MjData, state: dict) -> bool:
        active_payload_mask = np.asarray(state["active_payload_mask"], dtype=bool)
        if not active_payload_mask.any():
            state["success"] = False
            return False
        goal_positions = self._goal_positions_xyz(state)
        payload_positions = self._get_payload_positions(data)
        distance_to_goal = np.linalg.norm(payload_positions - goal_positions, axis=1)
        success = bool((distance_to_goal[active_payload_mask] <= self.goal_radius).all())
        state["success"] = success
        return success

    def _get_payload_goal_obs(self, data: mujoco.MjData, state: dict) -> np.ndarray:
        active_payload_mask = np.asarray(state["active_payload_mask"], dtype=bool)
        goal_positions = np.asarray(state["goal_positions"], dtype=float)
        obs = np.zeros((self.max_payloads, PAYLOAD_GOAL_OBS_RECORD_DIM), dtype=float)
        obs[:, 4:10] = np.asarray(PAYLOAD_GOAL_IDENTITY_ROT6D, dtype=float)
        payload_positions = self._get_payload_positions(data)
        for payload_idx in range(self.max_payloads):
            if not active_payload_mask[payload_idx]:
                continue
            obs[payload_idx, 0] = 1.0
            obs[payload_idx, 1:4] = payload_positions[payload_idx]
            obs[payload_idx, 4:10] = self._get_payload_orientation_rot6d_for_payload(data, payload_idx)
            obs[payload_idx, 10:12] = goal_positions[payload_idx]
        return obs.reshape(-1)

    def _get_payload_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self._payload_qpos_indices[:, :3]], dtype=float)

    def _goal_positions_xyz(self, state: dict) -> np.ndarray:
        goal_xy = np.asarray(state["goal_positions"], dtype=float)
        goal_positions = np.zeros((self.max_payloads, 3), dtype=float)
        goal_positions[:, :2] = goal_xy
        goal_positions[:, 2] = self.payload_radius
        return goal_positions

    def _get_payload_orientation_rot6d_for_payload(self, data: mujoco.MjData, payload_idx: int) -> np.ndarray:
        payload_quat = np.asarray(data.qpos[self._payload_qpos_indices[payload_idx, 3:7]], dtype=float)
        return quat_to_rot6d(payload_quat, axis=-1)

    def _payload_spawn_x_positions(self, num_active_payloads: int) -> np.ndarray:
        centered = np.arange(num_active_payloads, dtype=float) - ((num_active_payloads - 1) / 2.0)
        return centered * self.payload_spawn_margin

    def _add_payload_geom(self, payload_body: mujoco.MjsBody, *, rgba: tuple[float, float, float, float]) -> None:
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

    @staticmethod
    def _payload_body_name(payload_idx: int) -> str:
        return f"Payload{payload_idx}"

    @staticmethod
    def _goal_body_name(payload_idx: int) -> str:
        return f"PayloadGoal{payload_idx}"

    @staticmethod
    def _payload_color(payload_idx: int) -> tuple[float, float, float, float]:
        return PAYLOAD_GOAL_COLORS[payload_idx % len(PAYLOAD_GOAL_COLORS)]
