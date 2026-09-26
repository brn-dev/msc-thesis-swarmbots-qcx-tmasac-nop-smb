from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.scenarios.dual_payload_plane_scenario import _as_payload_pair
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import (
    MJWPayloadPlaneRuntimeMetadata,
    MJWPayloadPlaneScenario,
)


@dataclass
class MJWDualPayloadPlaneScenario(MJWPayloadPlaneScenario):
    lagging_payload_weight: float = 0.75

    def __post_init__(self) -> None:
        self.payload_offset_x = _as_payload_pair(self.payload_offset_x)
        self.payload_offset_y = _as_payload_pair(self.payload_offset_y)
        super().__post_init__()
        self.lagging_payload_weight = float(self.lagging_payload_weight)
        if not 0.0 <= self.lagging_payload_weight <= 1.0:
            raise ValueError(f"Expected lagging_payload_weight in [0, 1], got {self.lagging_payload_weight}")
        self.towards_payload_units_per_payload = max(1, math.ceil(self.swarm.num_units / 3))

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

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        numeric_offsets = [float(v) for v in self.payload_offset_y if isinstance(v, (int, float))]
        payload_offset_y = float(np.mean(numeric_offsets)) if numeric_offsets else 0.75
        distance = max(8.0, min(14.0, self.swarm.max_unit_extent * 10.0 + self.payload_radius * 6.0))
        return MJWRecordingCameraConfig(
            lookat=(0.0, payload_offset_y, max(0.5, self.payload_radius * 4.0)),
            distance=distance,
            azimuth=180.0,
            elevation=-35.0,
        )

    def build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.plane_size, self.plane_size, 0.1],
            rgba=[0.2, 0.3, 0.4, 1.0],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        for payload_name in ("Payload0", "Payload1"):
            payload_body = worldbody.add_body(name=payload_name, pos=[0, 0, self.payload_radius])
            payload_body.add_freejoint(name=f"{payload_name}_freejoint")
            self._add_payload_geom(payload_body)

        model = spec.compile()
        model.opt.timestep = float(self.timestep)
        if self.friction is not None:
            if isinstance(self.friction, (int, float)):
                friction = np.asarray([float(self.friction), 0.005, 0.0001], dtype=float)
            else:
                friction = np.asarray(tuple(float(v) for v in self.friction), dtype=float)
            model.geom_friction[:] = friction
        return model

    def get_single_observation_space(self) -> spaces.Dict:
        obs_space = super().get_single_observation_space()
        spaces_dict = dict(obs_space.spaces)
        global_obs_dim = 18 if self.payload_pos_observable else 0
        hidden_global_vars_dim = 0 if self.payload_pos_observable else 18
        spaces_dict["global_obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=(global_obs_dim,), dtype=np.float32)
        spaces_dict["hidden_global_vars"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(hidden_global_vars_dim,),
            dtype=np.float32,
        )
        return spaces.Dict(spaces_dict)

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWPayloadPlaneRuntimeMetadata:
        payload_qpos_indices = np.stack(
            [
                np.asarray(mj_utils.qpos_indices_for_body(host_model, payload_name), dtype=np.int64)
                for payload_name in ("Payload0", "Payload1")
            ],
            axis=0,
        )
        if payload_qpos_indices.shape != (2, 7):
            raise ValueError("Each payload body must expose a free joint with 7 qpos values.")
        return MJWPayloadPlaneRuntimeMetadata(payload_qpos_indices=payload_qpos_indices)

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_dual_payload_plane_runtime import DualPayloadPlaneMJWScenarioRuntime

        return DualPayloadPlaneMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
