from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco

from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRuntimeBindings
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneScenario
from swarmbots.utils.mujoco_step_geoms import add_payload_step_geom, payload_step_settings


@dataclass
class MJWPayloadStepScenario(MJWPayloadPlaneScenario):
    step_start_y: float = 1.5
    step_length: float = 3.0
    step_width: float = 4.0
    step_height: float = 0.08
    payload_step_height_reward_weight: float = 4.0
    payload_step_height_reward_distance: float = 0.7
    payload_success_y: float | None = 3.2
    payload_success_reward: float = 5.0
    payload_success_height_tolerance: float = 0.03

    def __post_init__(self) -> None:
        self.step_start_y = float(self.step_start_y)
        self.step_length = float(self.step_length)
        self.step_width = float(self.step_width)
        self.step_height = float(self.step_height)
        if self.step_length <= 0.0:
            raise ValueError(f"Expected step_length > 0, got {self.step_length}")
        if self.step_width <= 0.0:
            raise ValueError(f"Expected step_width > 0, got {self.step_width}")
        if self.step_height <= 0.0:
            raise ValueError(f"Expected step_height > 0, got {self.step_height}")

        self.payload_step_height_reward_weight = float(self.payload_step_height_reward_weight)
        self.payload_step_height_reward_distance = float(self.payload_step_height_reward_distance)
        if self.payload_step_height_reward_distance <= 0.0:
            raise ValueError(
                "Expected payload_step_height_reward_distance > 0, "
                f"got {self.payload_step_height_reward_distance}"
            )
        self.payload_success_y = None if self.payload_success_y is None else float(self.payload_success_y)
        self.payload_success_reward = float(self.payload_success_reward)
        self.payload_success_height_tolerance = float(self.payload_success_height_tolerance)
        if self.payload_success_height_tolerance < 0.0:
            raise ValueError(
                "Expected payload_success_height_tolerance >= 0, "
                f"got {self.payload_success_height_tolerance}"
            )
        super().__post_init__()

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

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_payload_step_runtime import PayloadStepMJWScenarioRuntime

        return PayloadStepMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
