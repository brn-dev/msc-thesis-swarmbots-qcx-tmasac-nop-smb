from typing import Any, Iterable, Literal

import mujoco
import numpy as np

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp
from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.utils.mujoco_render_geoms import add_payload_centering_boundary_geoms

PayloadShape = Literal["sphere", "box", "capsule"]


class PayloadPlaneScenario(BaseScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        timestep: float = 0.002,
        action_repeat: int = 15,
        plane_size: float = 100.0,
        payload_shape: PayloadShape = "sphere",
        payload_radius: float = 0.2,
        payload_mass: float = 1.0,
        payload_offset_x: FloatOrDistParams = 0.0,
        payload_offset_y: FloatOrDistParams = 0.75,
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
        self.payload_offset_x = payload_offset_x
        self.payload_offset_y = payload_offset_y
        self.payload_pos_observable = bool(payload_pos_observable)

        self.forward_reward_weight = float(forward_reward_weight)
        self.forward_reward_max_y = None if forward_reward_max_y is None else float(forward_reward_max_y)
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

        self._payload_qpos_indices = np.asarray(mj_utils.qpos_indices_for_body(self.dummy_model, "Payload"), dtype=int)
        self._payload_qvel_indices = np.asarray(mj_utils.dof_indices_for_body(self.dummy_model, "Payload"), dtype=int)
        if self._payload_qpos_indices.shape[0] < 7:
            raise ValueError("Payload body must expose a free joint with 7 qpos values.")

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            {
                "plane_size": self.plane_size,
                "payload_shape": self.payload_shape,
                "payload_radius": self.payload_radius,
                "payload_mass": self.payload_mass,
                "payload_offset_x": self.payload_offset_x,
                "payload_offset_y": self.payload_offset_y,
                "payload_pos_observable": self.payload_pos_observable,
                "forward_reward_weight": self.forward_reward_weight,
                "forward_reward_max_y": self.forward_reward_max_y,
                "towards_payload_reward_weight": self.towards_payload_reward_weight,
                "towards_payload_goal_radius": self.towards_payload_goal_radius,
                "payload_centering_penalty_weight": self.payload_centering_penalty_weight,
                "payload_centering_penalty_power": self.payload_centering_penalty_power,
                "payload_centering_tolerance": self.payload_centering_tolerance,
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
        self._add_static_geoms(worldbody)

        payload_body = worldbody.add_body(name="Payload", pos=[0, 0, self.payload_radius])
        payload_body.add_freejoint(name="Payload_freejoint")
        self._add_payload_geom(payload_body)

        return spec

    def _add_static_geoms(self, worldbody: mujoco.MjsBody) -> None:
        pass

    def _add_payload_geom(self, payload_body: mujoco.MjsBody) -> None:
        geom_kwargs: dict[str, Any] = {
            "mass": self.payload_mass,
            "rgba": [0.92, 0.72, 0.18, 1.0],
        }
        if self.payload_shape == "sphere":
            payload_body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[self.payload_radius, 0, 0],
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
                fromto=[-self.payload_radius, 0, 0, self.payload_radius, 0, 0],
                size=[self.payload_radius, 0, 0],
                **geom_kwargs,
            )
        else:
            raise AssertionError(f"Unhandled payload_shape: {self.payload_shape}")

    def add_render_geoms(self, scene: mujoco.MjvScene) -> None:
        if self.payload_centering_penalty_weight == 0.0:
            return
        add_payload_centering_boundary_geoms(
            scene,
            tolerance=self.payload_centering_tolerance,
            half_length=self.plane_size,
        )

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)
        self._reset_payload(data, state)
        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state["progress"] = self._compute_payload_progress_baseline(data)
        state["towards_payload_progress"] = self._compute_towards_payload_progress_baseline(data, state)
        return state, connections

    def get_obs(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: dict,
        connections: SwarmConnections,
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)
        payload_obs = self._get_payload_obs(data)
        if self.payload_pos_observable:
            obs["global_obs"] = payload_obs
            obs["hidden_global_vars"] = np.zeros((0,), dtype=float)
        else:
            obs["global_obs"] = np.zeros((0,), dtype=float)
            obs["hidden_global_vars"] = payload_obs
        return obs

    def compute_progress_reward(
        self,
        data: mujoco.MjData,
        state: dict,
    ) -> float:
        old_progress = state["progress"]
        new_progress = self._compute_payload_progress_baseline(data)
        state["progress"] = new_progress

        forward_reward = self.potential_reward_delta(new_progress, old_progress)
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
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is not None:
            self._enforce_inactive_units_state(model, data, units_active_mask)

        self.compute_progress_reward(data, state)
        forward_reward = state["forward_reward"]
        towards_payload_reward = state["towards_payload_reward"]
        payload_x_penalty = state["payload_x_penalty"]
        progress_reward = state["progress_reward"]

        guidance_reward = super().compute_guidance_reward(data, action, state, connections)
        state["guidance_reward"] = guidance_reward

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_forward_reward = forward_reward * self.forward_reward_weight * progress_reward_weight
        weighted_towards_payload_reward = towards_payload_reward * progress_reward_weight
        weighted_payload_x_penalty = payload_x_penalty * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state["weighted_progress_reward"] = weighted_progress_reward
        state["weighted_forward_reward"] = weighted_forward_reward
        state["weighted_forward_progress_reward"] = weighted_forward_reward
        state["weighted_towards_payload_reward"] = weighted_towards_payload_reward
        state["weighted_payload_x_penalty"] = weighted_payload_x_penalty
        state["weighted_guidance_reward"] = weighted_guidance_reward
        state["reward_terms"] = {
            "forward": weighted_forward_reward,
            "towards_payload": weighted_towards_payload_reward,
            "payload_x": weighted_payload_x_penalty,
            "guidance": weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, False

    def _reset_payload(self, data: mujoco.MjData, state: dict) -> None:
        swarm_start_location = np.asarray(state["swarm_start_location"], dtype=float)
        payload_position = np.array(
            [
                swarm_start_location[0] + eval_fodp(self.payload_offset_x, self.rng),
                swarm_start_location[1] + eval_fodp(self.payload_offset_y, self.rng),
                self.payload_radius,
            ],
            dtype=float,
        )
        data.qpos[self._payload_qpos_indices[:3]] = payload_position
        data.qpos[self._payload_qpos_indices[3:7]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self._payload_qvel_indices.size > 0:
            data.qvel[self._payload_qvel_indices] = 0.0
        state["payload_spawn_position"] = payload_position.copy()

    def _compute_payload_progress_baseline(self, data: mujoco.MjData) -> float:
        payload_y = float(self._get_payload_position(data)[1])
        if self.forward_reward_max_y is not None:
            payload_y = min(payload_y, self.forward_reward_max_y)
        return payload_y

    def _compute_towards_payload_reward(self, data: mujoco.MjData, state: dict) -> float:
        if self.towards_payload_reward_weight == 0.0:
            return 0.0

        old_progress = state["towards_payload_progress"]
        new_progress = self._compute_towards_payload_progress_baseline(data, state)
        state["towards_payload_progress"] = new_progress
        return self.potential_reward_delta(new_progress, old_progress) * self.towards_payload_reward_weight

    def _compute_towards_payload_progress_baseline(self, data: mujoco.MjData, state: dict) -> float:
        payload_position = self._get_payload_position(data)
        virtual_goal_position = np.asarray(payload_position[:2], dtype=float).copy()
        virtual_goal_position[1] -= self.payload_radius
        unit_xy = np.asarray(data.qpos[self._qpos_indices[:, :2]], dtype=float)
        distance_to_goal = np.linalg.norm(unit_xy - virtual_goal_position[np.newaxis, :], axis=1)
        remaining_distance = np.maximum(distance_to_goal - self.towards_payload_goal_radius, 0.0)

        units_active_mask = state.get("units_active_mask")
        if units_active_mask is None:
            return -float(remaining_distance.mean())

        active_mask = np.asarray(units_active_mask, dtype=bool)
        active_units_count = int(active_mask.sum())
        if active_units_count == 0:
            return 0.0
        return -float(remaining_distance[active_mask].mean())

    def _compute_payload_x_penalty(self, data: mujoco.MjData) -> float:
        if self.payload_centering_penalty_weight == 0.0:
            return 0.0
        payload_x = max(abs(float(self._get_payload_position(data)[0])) - self.payload_centering_tolerance, 0.0)
        penalty_magnitude = payload_x ** self.payload_centering_penalty_power
        return -penalty_magnitude * self.payload_centering_penalty_weight

    def _get_payload_position(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(data.qpos[self._payload_qpos_indices[:3]], dtype=float)

    def _get_payload_orientation_rot6d(self, data: mujoco.MjData) -> np.ndarray:
        payload_quat = np.asarray(data.qpos[self._payload_qpos_indices[3:7]], dtype=float)
        return quat_to_rot6d(payload_quat, axis=-1)

    def _get_payload_obs(self, data: mujoco.MjData) -> np.ndarray:
        return np.concatenate(
            (
                self._get_payload_position(data),
                self._get_payload_orientation_rot6d(data),
            ),
            axis=0,
        )
