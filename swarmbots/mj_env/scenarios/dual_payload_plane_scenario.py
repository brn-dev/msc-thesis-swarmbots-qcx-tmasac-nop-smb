import math
from typing import Any, Iterable

import mujoco
import numpy as np

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict
from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario, PayloadShape
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


def _as_payload_pair(value: FloatOrDistParams | Iterable[FloatOrDistParams]) -> tuple[FloatOrDistParams, FloatOrDistParams]:
    if isinstance(value, (int, float)):
        return value, value
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(f"Expected exactly two payload offsets, got {len(values)}")
    return values[0], values[1]


class DualPayloadPlaneScenario(PayloadPlaneScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        payload_shape: PayloadShape = "sphere",
        payload_radius: float = 0.2,
        payload_mass: float = 1.0,
        payload_offset_x: FloatOrDistParams | Iterable[FloatOrDistParams] = (-0.45, 0.45),
        payload_offset_y: FloatOrDistParams | Iterable[FloatOrDistParams] = (0.75, 0.75),
        payload_pos_observable: bool = True,
        actuator_strength: float = 8.0,
        connection_dist_threshold: float = 0.1,
        connection_angle_threshold: float = -0.5,
        disconnect_potential_threshold: float = 5.0,
        friction: float | Iterable[float] | None = None,
        force_elliptic_cone: bool = False,
        progress_reward_weight: float = 1.0,
        forward_reward_weight: float = 1.0,
        forward_reward_max_y: float | None = None,
        lagging_payload_weight: float = 0.75,
        towards_payload_reward_weight: float = 1.0,
        towards_payload_goal_radius: float | None = None,
        payload_centering_penalty_weight: float = 1.0,
        payload_centering_penalty_power: float = 1.0,
        payload_centering_tolerance: float = 0.0,
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

        self.payload_shape: PayloadShape = payload_shape
        if self.payload_shape not in ("sphere", "box", "capsule"):
            raise ValueError(
                f"Expected payload_shape to be 'sphere', 'box', or 'capsule', got {self.payload_shape!r}"
            )
        self.payload_radius = float(payload_radius)
        if self.payload_radius <= 0.0:
            raise ValueError(f"Expected payload_radius > 0, got {self.payload_radius}")
        self.payload_mass = float(payload_mass)
        if self.payload_mass <= 0.0:
            raise ValueError(f"Expected payload_mass > 0, got {self.payload_mass}")
        self.payload_offset_x = _as_payload_pair(payload_offset_x)
        self.payload_offset_y = _as_payload_pair(payload_offset_y)
        self.payload_pos_observable = bool(payload_pos_observable)

        self.forward_reward_weight = float(forward_reward_weight)
        self.forward_reward_max_y = None if forward_reward_max_y is None else float(forward_reward_max_y)
        self.lagging_payload_weight = float(lagging_payload_weight)
        if not 0.0 <= self.lagging_payload_weight <= 1.0:
            raise ValueError(f"Expected lagging_payload_weight in [0, 1], got {self.lagging_payload_weight}")
        self.towards_payload_reward_weight = float(towards_payload_reward_weight)
        self.towards_payload_goal_radius = (
            self.payload_radius if towards_payload_goal_radius is None else float(towards_payload_goal_radius)
        )
        if self.towards_payload_goal_radius < 0.0:
            raise ValueError(f"Expected towards_payload_goal_radius >= 0, got {self.towards_payload_goal_radius}")
        self.payload_centering_penalty_weight = float(payload_centering_penalty_weight)
        if self.payload_centering_penalty_weight < 0.0:
            raise ValueError(
                "Expected payload_centering_penalty_weight >= 0, "
                f"got {self.payload_centering_penalty_weight}"
            )
        self.payload_centering_penalty_power = float(payload_centering_penalty_power)
        if self.payload_centering_penalty_power <= 0.0:
            raise ValueError(
                f"Expected payload_centering_penalty_power > 0, got {self.payload_centering_penalty_power}"
            )
        self.payload_centering_tolerance = float(payload_centering_tolerance)
        if self.payload_centering_tolerance < 0.0:
            raise ValueError(
                f"Expected payload_centering_tolerance >= 0, got {self.payload_centering_tolerance}"
            )

        BaseScenario.__init__(
            self,
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

        self._payload_names = ("Payload0", "Payload1")
        self.towards_payload_units_per_payload = max(1, math.ceil(self.num_units / 3))
        self._payload_qpos_indices = np.stack(
            [np.asarray(mj_utils.qpos_indices_for_body(self.dummy_model, name), dtype=int) for name in self._payload_names],
            axis=0,
        )
        self._payload_qvel_indices = [
            np.asarray(mj_utils.dof_indices_for_body(self.dummy_model, name), dtype=int) for name in self._payload_names
        ]
        if self._payload_qpos_indices.shape != (2, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            {
                "num_payloads": 2,
                "scenario_type": "dual_payload_plane",
                "towards_payload_units_per_payload": self.towards_payload_units_per_payload,
                "lagging_payload_weight": self.lagging_payload_weight,
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

        for payload_name in ("Payload0", "Payload1"):
            payload_body = worldbody.add_body(name=payload_name, pos=[0, 0, self.payload_radius])
            payload_body.add_freejoint(name=f"{payload_name}_freejoint")
            self._add_payload_geom(payload_body)

        return spec

    def compute_progress_reward(self, data: mujoco.MjData, state: dict) -> float:
        old_payload_progress = np.asarray(state["payload_progress"], dtype=float)
        old_back_payload_progress = float(state["back_payload_progress"])
        new_payload_progress = self._compute_payload_progress_baseline(data)
        new_back_payload_progress = float(np.min(new_payload_progress))
        state["payload_progress"] = new_payload_progress
        state["back_payload_progress"] = new_back_payload_progress
        state["progress"] = float(new_payload_progress.sum() + new_back_payload_progress)

        payload_progress_delta = self.potential_reward_delta(new_payload_progress, old_payload_progress)
        back_progress_delta = self.potential_reward_delta(new_back_payload_progress, old_back_payload_progress)
        num_payloads = float(payload_progress_delta.shape[0])
        forward_reward = float(
            ((1.0 - self.lagging_payload_weight) * payload_progress_delta.sum())
            + (self.lagging_payload_weight * num_payloads * back_progress_delta)
        )
        towards_payload_reward = self._compute_towards_payload_reward(data, state)
        payload_x_penalty = self._compute_payload_x_penalty(data)
        progress_reward = (forward_reward * self.forward_reward_weight) + towards_payload_reward + payload_x_penalty

        state["forward_reward"] = forward_reward
        state["towards_payload_reward"] = towards_payload_reward
        state["payload_x_penalty"] = payload_x_penalty
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
        return PayloadPlaneScenario.evaluate_step(self, action, model, data, state, connections)

    def _reset_payload(self, data: mujoco.MjData, state: dict) -> None:
        swarm_start_location = np.asarray(state["swarm_start_location"], dtype=float)
        payload_positions = []
        for payload_idx in range(2):
            payload_position = np.array(
                [
                    swarm_start_location[0] + eval_fodp(self.payload_offset_x[payload_idx], self.rng),
                    swarm_start_location[1] + eval_fodp(self.payload_offset_y[payload_idx], self.rng),
                    self.payload_radius,
                ],
                dtype=float,
            )
            data.qpos[self._payload_qpos_indices[payload_idx, :3]] = payload_position
            data.qpos[self._payload_qpos_indices[payload_idx, 3:7]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            qvel_indices = self._payload_qvel_indices[payload_idx]
            if qvel_indices.size > 0:
                data.qvel[qvel_indices] = 0.0
            payload_positions.append(payload_position)
        state["payload_spawn_position"] = np.stack(payload_positions, axis=0)

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = BaseScenario.reset_scenario(self, model, data, settle=False)
        self._reset_payload(data, state)
        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        payload_progress = self._compute_payload_progress_baseline(data)
        back_payload_progress = float(np.min(payload_progress))
        state["payload_progress"] = payload_progress
        state["back_payload_progress"] = back_payload_progress
        state["progress"] = float(payload_progress.sum() + back_payload_progress)
        state["towards_payload_progress"] = self._compute_towards_payload_progress_baseline(data, state)
        return state, connections

    def _compute_payload_progress_baseline(self, data: mujoco.MjData) -> np.ndarray:
        payload_y = self._get_payload_positions(data)[:, 1]
        if self.forward_reward_max_y is not None:
            payload_y = np.minimum(payload_y, self.forward_reward_max_y)
        return np.asarray(payload_y, dtype=float)

    def _compute_towards_payload_reward(self, data: mujoco.MjData, state: dict) -> float:
        if self.towards_payload_reward_weight == 0.0:
            return 0.0

        old_progress = np.asarray(state["towards_payload_progress"], dtype=float)
        new_progress = self._compute_towards_payload_progress_baseline(data, state)
        state["towards_payload_progress"] = new_progress
        return float(self.potential_reward_delta(new_progress, old_progress).sum() * self.towards_payload_reward_weight)

    def _compute_towards_payload_progress_baseline(self, data: mujoco.MjData, state: dict) -> np.ndarray:
        payload_positions = self._get_payload_positions(data)
        virtual_goal_positions = np.asarray(payload_positions[:, :2], dtype=float).copy()
        virtual_goal_positions[:, 1] -= self.payload_radius
        unit_xy = np.asarray(data.qpos[self._qpos_indices[:, :2]], dtype=float)
        delta = unit_xy[:, np.newaxis, :] - virtual_goal_positions[np.newaxis, :, :]
        distance_to_goal = np.linalg.norm(delta, axis=-1)
        remaining_distance = np.maximum(distance_to_goal - self.towards_payload_goal_radius, 0.0)

        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            active_distances = remaining_distance
        else:
            active_distances = remaining_distance[np.asarray(units_active_mask, dtype=bool)]
        active_units_count = int(active_distances.shape[0])
        if active_units_count == 0:
            return np.zeros((2,), dtype=float)
        nearest_count = min(self.towards_payload_units_per_payload, active_units_count)
        nearest_distances = np.sort(active_distances, axis=0)[:nearest_count]
        return -nearest_distances.mean(axis=0)

    def _compute_payload_x_penalty(self, data: mujoco.MjData) -> float:
        if self.payload_centering_penalty_weight == 0.0:
            return 0.0
        payload_x = np.maximum(
            np.abs(self._get_payload_positions(data)[:, 0]) - self.payload_centering_tolerance,
            0.0,
        )
        penalty_magnitude = payload_x ** self.payload_centering_penalty_power
        return -float(penalty_magnitude.sum() * self.payload_centering_penalty_weight)

    def _get_payload_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self._payload_qpos_indices[:, :3]], dtype=float)

    def _get_payload_obs(self, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            [
                np.concatenate(
                    (
                        np.asarray(data.qpos[self._payload_qpos_indices[payload_idx, :3]], dtype=float),
                        self._get_payload_orientation_rot6d_for_payload(data, payload_idx),
                    ),
                    axis=0,
                )
                for payload_idx in range(2)
            ],
            axis=0,
        )

    def _get_payload_orientation_rot6d_for_payload(self, data: mujoco.MjData, payload_idx: int) -> np.ndarray:
        payload_quat = np.asarray(data.qpos[self._payload_qpos_indices[payload_idx, 3:7]], dtype=float)
        return self._quat_to_rot6d(payload_quat)

    @staticmethod
    def _quat_to_rot6d(payload_quat: np.ndarray) -> np.ndarray:
        from swarmbots.mj_env.quat_rot6d import quat_to_rot6d

        return quat_to_rot6d(payload_quat, axis=-1)
