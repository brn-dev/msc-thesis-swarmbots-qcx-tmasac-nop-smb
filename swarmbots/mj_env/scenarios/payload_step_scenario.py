from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from swarmbots.mj_env.scenarios.base_scenario import SwarmActDict
from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.utils.mujoco_step_geoms import add_payload_step_geom, payload_step_settings


class PayloadStepScenario(PayloadPlaneScenario):
    def __init__(
        self,
        swarm: BaseSwarm,
        step_start_y: float = 1.5,
        step_length: float = 3.0,
        step_width: float = 4.0,
        step_height: float = 0.08,
        payload_step_height_reward_weight: float = 4.0,
        payload_step_height_reward_distance: float = 0.7,
        payload_success_y: float | None = 3.2,
        payload_success_reward: float = 5.0,
        payload_success_height_tolerance: float = 0.03,
        **kwargs: Any,
    ) -> None:
        self.step_start_y = float(step_start_y)
        self.step_length = float(step_length)
        self.step_width = float(step_width)
        self.step_height = float(step_height)
        if self.step_length <= 0.0:
            raise ValueError(f"Expected step_length > 0, got {self.step_length}")
        if self.step_width <= 0.0:
            raise ValueError(f"Expected step_width > 0, got {self.step_width}")
        if self.step_height <= 0.0:
            raise ValueError(f"Expected step_height > 0, got {self.step_height}")

        self.payload_step_height_reward_weight = float(payload_step_height_reward_weight)
        self.payload_step_height_reward_distance = float(payload_step_height_reward_distance)
        if self.payload_step_height_reward_distance <= 0.0:
            raise ValueError(
                "Expected payload_step_height_reward_distance > 0, "
                f"got {self.payload_step_height_reward_distance}"
            )
        self.payload_success_y = None if payload_success_y is None else float(payload_success_y)
        self.payload_success_reward = float(payload_success_reward)
        self.payload_success_height_tolerance = float(payload_success_height_tolerance)
        if self.payload_success_height_tolerance < 0.0:
            raise ValueError(
                "Expected payload_success_height_tolerance >= 0, "
                f"got {self.payload_success_height_tolerance}"
            )

        super().__init__(swarm=swarm, **kwargs)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update(
            payload_step_settings(
                step_start_y=self.step_start_y,
                step_length=self.step_length,
                step_width=self.step_width,
                step_height=self.step_height,
                payload_step_height_reward_weight=self.payload_step_height_reward_weight,
                payload_step_height_reward_distance=self.payload_step_height_reward_distance,
                payload_success_y=self.payload_success_y,
                payload_success_reward=self.payload_success_reward,
                payload_success_height_tolerance=self.payload_success_height_tolerance,
            )
        )
        return settings

    def _add_static_geoms(self, worldbody: mujoco.MjsBody) -> None:
        add_payload_step_geom(
            worldbody,
            step_start_y=self.step_start_y,
            step_length=self.step_length,
            step_width=self.step_width,
            step_height=self.step_height,
        )

    def reset_scenario(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=settle)
        state["payload_step_height_potential"] = self._compute_payload_step_height_potential(data)
        state["payload_step_height_done"] = self._payload_has_reached_step(data)
        state["success"] = False
        return state, connections

    def compute_progress_reward(
        self,
        data: mujoco.MjData,
        state: dict,
    ) -> float:
        progress_reward = super().compute_progress_reward(data, state)
        payload_step_height_reward = self._compute_payload_step_height_reward(data, state)
        progress_reward += payload_step_height_reward
        state["payload_step_height_reward"] = payload_step_height_reward
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
        reward, _terminated = super().evaluate_step(action, model, data, state, connections)

        progress_reward_weight = self.reward_weights["progress_reward_weight"]
        weighted_height_reward = state["payload_step_height_reward"] * progress_reward_weight
        terminated = self._compute_payload_success_termination(data, state)
        payload_success_reward = self.payload_success_reward if terminated else 0.0
        weighted_success_reward = payload_success_reward * progress_reward_weight

        state["payload_success_reward"] = payload_success_reward
        state["weighted_payload_step_height_reward"] = weighted_height_reward
        state["weighted_payload_success_reward"] = weighted_success_reward
        state["weighted_progress_reward"] += weighted_success_reward
        state["progress_reward"] += payload_success_reward
        state["reward_terms"]["payload_step_height"] = weighted_height_reward
        state["reward_terms"]["success"] = weighted_success_reward

        return reward + weighted_success_reward, terminated

    def _compute_payload_step_height_reward(self, data: mujoco.MjData, state: dict) -> float:
        if self.payload_step_height_reward_weight == 0.0:
            return 0.0

        previous_potential = float(state.get("payload_step_height_potential", 0.0))
        previous_done = bool(state.get("payload_step_height_done", False))
        crossed_step = self._payload_has_reached_step(data)
        done = previous_done or crossed_step
        current_potential = 0.0 if done else self._compute_payload_step_height_potential(data)
        height_delta = self.potential_reward_delta(current_potential, previous_potential)
        if crossed_step and not previous_done and height_delta < 0.0:
            height_delta = 0.0

        state["payload_step_height_potential"] = current_potential
        state["payload_step_height_done"] = done
        return height_delta * self.payload_step_height_reward_weight

    def _compute_payload_step_height_potential(self, data: mujoco.MjData) -> float:
        payload_position = self._get_payload_position(data)
        distance_to_step = self.step_start_y - float(payload_position[1])
        if distance_to_step < 0.0 or distance_to_step > self.payload_step_height_reward_distance:
            return 0.0

        approach = 1.0 - (distance_to_step / self.payload_step_height_reward_distance)
        height = np.clip(
            (float(payload_position[2]) - self.payload_radius) / self.step_height,
            0.0,
            1.0,
        )
        return float(np.sqrt(approach) * height)

    def _payload_has_reached_step(self, data: mujoco.MjData) -> bool:
        return bool(float(self._get_payload_position(data)[1]) >= self.step_start_y)

    def _compute_payload_success_termination(self, data: mujoco.MjData, state: dict) -> bool:
        if self.payload_success_y is None:
            state["success"] = False
            return False

        payload_position = self._get_payload_position(data)
        success = bool(
            float(payload_position[1]) >= self.payload_success_y
            and float(payload_position[2]) >= self._payload_success_min_z()
        )
        state["success"] = success
        return success

    def _payload_success_min_z(self) -> float:
        return max(
            self.payload_radius,
            self.payload_radius + self.step_height - self.payload_success_height_tolerance,
        )
