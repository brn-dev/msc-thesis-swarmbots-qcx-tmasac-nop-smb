from __future__ import annotations

from typing import Any

import mujoco


def add_payload_step_geom(
    worldbody: mujoco.MjsBody,
    *,
    step_start_y: float,
    step_length: float,
    step_width: float,
    step_height: float,
) -> None:
    worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[float(step_width) / 2.0, float(step_length) / 2.0, float(step_height) / 2.0],
        pos=[0.0, float(step_start_y) + (float(step_length) / 2.0), float(step_height) / 2.0],
        rgba=[0.48, 0.52, 0.56, 1.0],
    )


def payload_step_settings(
    *,
    step_start_y: float,
    step_length: float,
    step_width: float,
    step_height: float,
    payload_step_height_reward_weight: float,
    payload_step_height_reward_distance: float,
    payload_success_y: float | None,
    payload_success_reward: float,
    payload_success_height_tolerance: float,
) -> dict[str, Any]:
    return {
        "scenario_type": "payload_step",
        "step_start_y": step_start_y,
        "step_length": step_length,
        "step_width": step_width,
        "step_height": step_height,
        "payload_step_height_reward_weight": payload_step_height_reward_weight,
        "payload_step_height_reward_distance": payload_step_height_reward_distance,
        "payload_success_y": payload_success_y,
        "payload_success_reward": payload_success_reward,
        "payload_success_height_tolerance": payload_success_height_tolerance,
    }
