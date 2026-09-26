from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal, TypeVar
import importlib.util
import shutil
import sys

import numpy as np
import torch

from swarmbots.mj_env.swarm.unit_config import (
    UNIT_CONFIG_TETRAHEDRON_XY,
    UNIT_CONFIG_TETRAHEDRON_XYZ,
    UNIT_CONFIG_TETRAHEDRON_ZX,
)
from swarmbots.mjw_env.scenarios.mjw_bridge_scenario import MJWBridgeScenario
from swarmbots.mjw_env.scenarios.mjw_climb_scenario import MJWClimbScenario
from swarmbots.mjw_env.scenarios.mjw_dual_payload_plane_scenario import MJWDualPayloadPlaneScenario
from swarmbots.mjw_env.scenarios.mjw_find_opening_scenario import MJWFindOpeningScenario
from swarmbots.mjw_env.scenarios.mjw_move_to_scenario import MJWMoveToScenario
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import MJWMultiPayloadGoalScenario
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneScenario
from swarmbots.mjw_env.scenarios.mjw_payload_step_scenario import MJWPayloadStepScenario
from swarmbots.mjw_env.scenarios.mjw_vertical_reach_scenario import MJWVerticalReachScenario
from swarmbots.mjw_env.scenarios.mjw_wall_scenario import MJWWallScenario
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm, MJWPreConnectedUnitLocationsConfig
from swarmbots.scenario_presets import scenario_presets_kwargs as shared_kwargs
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    WallDifficulty,
    make_scenario_kwargs,
    wall_scenario_kwargs_with_difficulty,
)

ScenarioT = TypeVar("ScenarioT")
JointPreset = Literal["zx", "xy", "xyz"]

DEFAULT_KWARGS = make_scenario_kwargs(shared_kwargs.COMMON_SCENARIO_KWARGS)
WALL_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.WALL_SCENARIO_KWARGS)
EASY_WALL_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.EASY_WALL_SCENARIO_KWARGS)
MEDIUM_WALL_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.MEDIUM_WALL_SCENARIO_KWARGS)
HARD_WALL_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.HARD_WALL_SCENARIO_KWARGS)
PO_WALL_EASY_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.PO_WALL_EASY_SCENARIO_KWARGS)
PO_WALL_MEDIUM_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.PO_WALL_MEDIUM_SCENARIO_KWARGS)
BRIDGE_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.BRIDGE_SCENARIO_KWARGS)
FIND_OPENING_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.FIND_OPENING_SCENARIO_KWARGS)
CLIMB_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.CLIMB_SCENARIO_KWARGS)
VERTICAL_REACH_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.VERTICAL_REACH_SCENARIO_KWARGS)
PAYLOAD_PLANE_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.PAYLOAD_PLANE_SCENARIO_KWARGS)
PAYLOAD_STEP_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.PAYLOAD_STEP_SCENARIO_KWARGS)
DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS)
MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS)
MOVE_TO_SCENARIO_KWARGS = make_scenario_kwargs(shared_kwargs.MOVE_TO_SCENARIO_KWARGS)

_JOINT_CONFIGS: dict[JointPreset, dict[str, object]] = {
    "zx": {
        "unit_config": UNIT_CONFIG_TETRAHEDRON_ZX,
        "hinge_range": (None, np.pi / 3),
        "hinge_armature": (0.025, 0.015),
        "hinge_damping": (0.3, 0.15),
        "hinge_frictionloss": (0.3, 0.15),
    },
    "xy": {
        "unit_config": UNIT_CONFIG_TETRAHEDRON_XY,
        "hinge_range": (np.pi / 3, np.pi / 3),
        "hinge_armature": (0.015, 0.015),
        "hinge_damping": (0.15, 0.15),
        "hinge_frictionloss": (0.15, 0.15),
    },
    "xyz": {
        "unit_config": UNIT_CONFIG_TETRAHEDRON_XYZ,
        "hinge_range": (np.pi / 3, np.pi / 3, None),
        "hinge_armature": (0.01, 0.01, 0.025),
        "hinge_damping": (0.1, 0.1, 0.5),
        "hinge_frictionloss": (0.1, 0.1, 0.5),
    },
}


def should_compile_reward_kernel_by_default() -> bool:
    if not hasattr(torch, "compile"):
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    if sys.platform == "win32" and shutil.which("cl") is None:
        return False
    return True


def _default_unit_start_locations() -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={
            4: 1.0,
            5: 1.0,
        },
        max_radius=1.5,
        unconnected_prob=0.02,
        z_pos=0.5,
        pool_seeds=tuple(range(42_000, 42_050)),
    )


def _resolve_swarm(
    *,
    swarm: MJWHomogeneousSwarm | None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None,
    quantize_connection_twist: int,
    joints: JointPreset,
) -> MJWHomogeneousSwarm:
    if swarm is not None:
        if unit_start_locations is not None:
            raise ValueError("unit_start_locations can only be used when presets construct the swarm")
        return swarm

    return MJWHomogeneousSwarm(
        unit_start_locations=unit_start_locations or _default_unit_start_locations(),
        body_radius=0.1,
        leg_length=0.2,
        leg_radius=0.025,
        connection_torquescale=50.0,
        quantize_connection_twist=quantize_connection_twist,
        **_JOINT_CONFIGS[joints],
    )


def _reward_kernel_kwargs() -> dict[str, object]:
    return {
        "compile_reward_kernel": should_compile_reward_kernel_by_default(),
        "reward_kernel_compile_mode": "default",
    }


def _preset_kwargs(*parts: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    return make_scenario_kwargs(DEFAULT_KWARGS, *parts, _reward_kernel_kwargs(), overrides)


def _make_scenario(
    scenario_cls: Callable[..., ScenarioT],
    *,
    scenario_kwargs: Mapping[str, object],
    seed: int | None,
    swarm: MJWHomogeneousSwarm | None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None,
    quantize_connection_twist: int,
    joints: JointPreset,
) -> ScenarioT:
    return scenario_cls(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        seed=seed,
        **scenario_kwargs,
    )


def default_wall(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    difficulty: WallDifficulty | None = None,
    **kwargs: object,
) -> MJWWallScenario:
    return _make_scenario(
        MJWWallScenario,
        scenario_kwargs=_preset_kwargs(wall_scenario_kwargs_with_difficulty(difficulty), overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def easy_wall(**kwargs: object) -> MJWWallScenario:
    return default_wall(difficulty="easy", **kwargs)


def medium_wall(**kwargs: object) -> MJWWallScenario:
    return default_wall(difficulty="medium", **kwargs)


def hard_wall(**kwargs: object) -> MJWWallScenario:
    return default_wall(difficulty="hard", **kwargs)


def default_bridge(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWBridgeScenario:
    return _make_scenario(
        MJWBridgeScenario,
        scenario_kwargs=_preset_kwargs(BRIDGE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_find_opening(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWFindOpeningScenario:
    return _make_scenario(
        MJWFindOpeningScenario,
        scenario_kwargs=_preset_kwargs(FIND_OPENING_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_climb(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWClimbScenario:
    return _make_scenario(
        MJWClimbScenario,
        scenario_kwargs=_preset_kwargs(CLIMB_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_vertical_reach(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWVerticalReachScenario:
    return _make_scenario(
        MJWVerticalReachScenario,
        scenario_kwargs=_preset_kwargs(VERTICAL_REACH_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_payload_plane(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWPayloadPlaneScenario:
    return _make_scenario(
        MJWPayloadPlaneScenario,
        scenario_kwargs=_preset_kwargs(PAYLOAD_PLANE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_dual_payload_plane(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWDualPayloadPlaneScenario:
    return _make_scenario(
        MJWDualPayloadPlaneScenario,
        scenario_kwargs=_preset_kwargs(DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_multi_payload_goal(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWMultiPayloadGoalScenario:
    return _make_scenario(
        MJWMultiPayloadGoalScenario,
        scenario_kwargs=_preset_kwargs(MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_payload_step(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWPayloadStepScenario:
    return _make_scenario(
        MJWPayloadStepScenario,
        scenario_kwargs=_preset_kwargs(PAYLOAD_STEP_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_move_to(
    *,
    seed: int | None = None,
    swarm: MJWHomogeneousSwarm | None = None,
    unit_start_locations: MJWPreConnectedUnitLocationsConfig | None = None,
    quantize_connection_twist: int = 8,
    joints: JointPreset = "zx",
    **kwargs: object,
) -> MJWMoveToScenario:
    return _make_scenario(
        MJWMoveToScenario,
        scenario_kwargs=_preset_kwargs(MOVE_TO_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )
