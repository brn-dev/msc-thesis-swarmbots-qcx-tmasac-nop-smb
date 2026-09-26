from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Literal

from swarmbots.mj_env.float_or_dist_params import SplitUniformDistParams, UniformDistParams
from swarmbots.scenario_presets.move_to_goal_config import RelativePolarGoalConfig


def make_scenario_kwargs(*parts: Mapping[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for part in parts:
        merged.update(deepcopy(dict(part)))
    return merged

WallDifficulty = Literal["easy", "medium", "hard"]
PoWallDifficulty = Literal["easy", "medium"]

COMMON_SCENARIO_KWARGS: dict[str, object] = {
    "timestep": 0.003,
    "action_repeat": 10,
    "friction": [1.25, 7e-3, 1.25e-4],
    "actuator_strength": 15.0,
    "connection_dist_threshold": 0.1,
    "connection_angle_threshold": -0.5,
    "disconnect_potential_threshold": 5.0,
    "continuous_connector_actions": True,
    "progress_reward_weight": 1.0,
    "guidance_reward_weight": 1.0,
    "units_without_connections_reward_weight": -1e-5,
    "potential_reward_discount_factor": 1.0,
    "include_connectors_xpos_in_obs": True,
    "include_connectors_xquat_in_obs": False,
    "quat_rot6d_representation": True,
    "reset_settle_time": 1.0,
    "reset_settle_timestep_scale": 3,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "randomize_initial_swarm_z_rotation": False,
}

WALL_SCENARIO_KWARGS: dict[str, object] = {
    "swarm_start_x": 0.0,
    "swarm_start_y": UniformDistParams(0.25, 0.75),
    "wall_distance": 1.0,
    "street_width": 10.0,
    "wall_height": 0.20,
    "forward_reward_weight": 1.0,
    "forward_reward_wall_boost_factor": 1.0,
    "wall_pass_reward_weight": 10.0,
    "wall_pass_reward_skew": 0.0,
    "wall_pass_thresholds": [-0.1, 0.1, 0.3, 0.5],
    "wall_success_threshold": 1.0,
    "wall_success_reward": 5.0,
    "wall_climb_reward_weight": 0.0,
    "wall_climb_reward_distance": 0.45,
}

WALL_DIFFICULTY_UPDATES: dict[WallDifficulty, dict[str, object]] = {
    "easy": {
        "wall_height": 0.2,
    },
    "medium": {
        "wall_height": 0.3,
        "swarm_start_y": UniformDistParams(0.25, 0.75),
        "forward_reward_weight": 2.0,
        "forward_reward_wall_boost_factor": 2.0,
        "wall_climb_reward_weight": 4.0,
        "wall_pass_reward_weight": 5.0,
        "wall_pass_reward_skew": 4.0,
        "wall_pass_thresholds": [0.2, 0.5],
        "units_without_connections_reward_weight": -3e-3,
    },
    "hard": {
        "wall_height": 0.4,
        "swarm_start_y": UniformDistParams(0.25, 0.75),
        "forward_reward_weight": 2.0,
        "forward_reward_wall_boost_factor": 2.0,
        "wall_climb_reward_weight": 4.0,
        "wall_pass_reward_weight": 5.0,
        "wall_pass_reward_skew": 4.0,
        "wall_pass_thresholds": [0.2, 0.5],
        "units_without_connections_reward_weight": -3e-3,
    },
}

def wall_scenario_kwargs_with_difficulty(difficulty: WallDifficulty | None) -> dict[str, object]:
    kwargs = WALL_SCENARIO_KWARGS.copy()
    if difficulty is not None:
        kwargs.update(WALL_DIFFICULTY_UPDATES[difficulty])
    return kwargs


EASY_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("easy")
MEDIUM_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("medium")
HARD_WALL_SCENARIO_KWARGS = wall_scenario_kwargs_with_difficulty("hard")


BRIDGE_SCENARIO_KWARGS: dict[str, object] = {
    "street_width": 6.0,
    "bridge_width": 1.5,
    "bridge_length": 4.0,
    "bridge_x": 0.0,
    "platform_length": 4.0,
    "platform_height": 0.2,
    "fall_z_threshold": -1.5,
    "success_margin": 0.5,
    "success_reward": 10.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.5,
    "fell_off_bridge_reward": -0.5,
}

FIND_OPENING_SCENARIO_KWARGS: dict[str, object] = {
    "street_width": 10.0,
    "wall_y": 2.0,
    "wall_height": 2.0,
    "wall_thickness": 0.2,
    "wall_segment_width": 100.0,
    "opening_width": 1,
    "opening_x": SplitUniformDistParams(-4.0, 4.0, margin=1.0),
    "opening_y_margin": 1.0,
    "success_reward": 10.0,
    "opening_distance_reward_weight": 3.0,
    "opening_distance_reward_falloff_distance": 2.0,
    "wall_exploration_cell_count": 10,
    "wall_exploration_cell_reward": 0.15,
    "wall_exploration_cell_depth": 0.5,
    "swarm_start_x": UniformDistParams(-0.25, 0.25),
    "swarm_start_y": UniformDistParams(-0.25, 0.25),
}

CLIMB_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": UniformDistParams(0.0, 0.5),
    "cuboid_size_x": 4.0,
    "cuboid_size_y": 4.0,
    "cuboid_size_z": 0.2,
    "cuboid_center_x": 0.0,
    "cuboid_center_y": 3,
    "horizontal_goal_radius": 0.3,
    "height_goal_radius": 0.1,
    "goal_radius": None,
    "goal_height_offset": None,
    "goal_success_reward": 5.0,
    "horizontal_reward_weight": 2.0,
    "height_reward_weight": 8.0,
    "visualize_goal": True,
    "units_without_connections_reward_weight": -1e-4,
}

VERTICAL_REACH_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": UniformDistParams(-0.25, 0.25),
    "swarm_start_y": UniformDistParams(-0.25, 0.25),
    "wall_width": 6.0,
    "wall_thickness": 0.35,
    "wall_height": 6.0,
    "wall_center_x": 0.0,
    "wall_y": 1.5,
    "goal_box_width": 1.0,
    "goal_box_depth": 0.4,
    "goal_box_height": 0.4,
    "goal_center_z": 0.8,
    "goal_success_reward": 10.0,
    "reach_column_half_width": 0.6,
    "reach_column_depth": 0.4,
    "reach_column_reward_weight": 8.0,
    "horizontal_goal_at_wall_contact": True,
    "horizontal_reward_weight": 1.0,
    "height_reward_weight": 0.25,
    "visualize_goal": True,
    "units_without_connections_reward_weight": -3e-3,
}

PAYLOAD_PLANE_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "forward_reward_weight": 10.0,
    "towards_payload_reward_weight": 1.0,
    "towards_payload_goal_radius": 0.5,
    "payload_centering_penalty_weight": 0.05,
    "payload_centering_penalty_power": 1.0,
    "payload_centering_tolerance": 0.5,
    "payload_shape": "box",
    "payload_radius": 0.3,
    "payload_mass": 3.0,
    "payload_offset_x": 0.0,
    "payload_offset_y": 1.3,
    "forward_reward_max_y": None,
}

PAYLOAD_STEP_SCENARIO_KWARGS: dict[str, object] = {
    **PAYLOAD_PLANE_SCENARIO_KWARGS,
    "payload_shape": "box",
    "payload_radius": 0.25,
    "payload_mass": 2.0,
    "payload_offset_y": 1.2,
    "forward_reward_max_y": 2.0,
    "step_start_y": 1.5,
    "step_length": 3.0,
    "step_width": 4.0,
    "step_height": 0.1,
    "payload_step_height_reward_weight": 4.0,
    "payload_step_height_reward_distance": 0.7,
    "payload_success_y": 2.5,
    "payload_success_reward": 5.0,
    "payload_success_height_tolerance": 0.03,
}

DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS: dict[str, object] = {
    **PAYLOAD_PLANE_SCENARIO_KWARGS,
    "payload_centering_tolerance": 0.75,
    "payload_offset_x": (-0.35, 0.35),
    "payload_offset_y": (1.3, 1.3),
    "lagging_payload_weight": 0.75,
}

MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": UniformDistParams(-0.25, 0.25),
    "swarm_start_y": UniformDistParams(-0.25, 0.25),
    "forward_reward_weight": 10.0,
    "payload_shape": "box",
    "payload_radius": 0.3,
    "payload_mass": 3.0,
    "max_payloads": 4,
    "active_payload_count_probs": {2: 1.0, 3: 1.0, 4: 1.0},
    "payload_spawn_y": 1.0,
    "payload_spawn_margin": 0.8,
    "goal_rect": (-2.5, 2.5, 2.5, 4.5),
    "goal_radius": 0.35,
    "visualize_goal": True,
    "success_reward": 5.0,
}

MOVE_TO_SCENARIO_KWARGS: dict[str, object] = {
    "plane_size": 100.0,
    "swarm_start_x": 0.0,
    "swarm_start_y": 0.0,
    "forward_reward_weight": 10.0,
    "goal_radius": 1.0,
    "goal": RelativePolarGoalConfig(
        distance=UniformDistParams(2.0, 4.0),
        angle=UniformDistParams(0.0, 2.0 * math.pi),
    ),
}

PO_WALL_SCENARIO_KWARGS = make_scenario_kwargs(
    MEDIUM_WALL_SCENARIO_KWARGS,
    {
        "swarm_start_y": UniformDistParams(-0.25, 0.0),
        "wall_distance": UniformDistParams.from_midpoint_and_width(1.0, 1.5),
    },
)

PO_WALL_DIFFICULTY_UPDATES: dict[PoWallDifficulty, dict[str, object]] = {
    "easy": {
        "wall_height": 0.25,
    },
    "medium": {
        "wall_height": 0.3,
    },
}


def po_wall_scenario_kwargs_with_difficulty(
    difficulty: PoWallDifficulty = "easy",
) -> dict[str, object]:
    return make_scenario_kwargs(PO_WALL_SCENARIO_KWARGS, PO_WALL_DIFFICULTY_UPDATES[difficulty])


PO_WALL_EASY_SCENARIO_KWARGS = po_wall_scenario_kwargs_with_difficulty("easy")
PO_WALL_MEDIUM_SCENARIO_KWARGS = po_wall_scenario_kwargs_with_difficulty("medium")
