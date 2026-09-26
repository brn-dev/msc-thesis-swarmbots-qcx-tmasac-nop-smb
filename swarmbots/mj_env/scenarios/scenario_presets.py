from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypeVar

import numpy as np

from swarmbots.mj_env.scenarios.bridge_scenario import BridgeScenario
from swarmbots.mj_env.scenarios.climb_scenario import ClimbScenario
from swarmbots.mj_env.scenarios.dual_payload_plane_scenario import DualPayloadPlaneScenario
from swarmbots.mj_env.scenarios.find_opening_scenario import FindOpeningScenario
from swarmbots.mj_env.scenarios.move_to_scenario import MoveToScenario
from swarmbots.mj_env.scenarios.multi_payload_goal_scenario import MultiPayloadGoalScenario
from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario
from swarmbots.mj_env.scenarios.payload_step_scenario import PayloadStepScenario
from swarmbots.mj_env.scenarios.vertical_reach_scenario import VerticalReachScenario
from swarmbots.mj_env.scenarios.wall_scenario import WallScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PreConnectedUnitLocationsConfig, UnitStartLocations
from swarmbots.mj_env.swarm.unit_config import (
    UNIT_CONFIG_TETRAHEDRON_XY,
    UNIT_CONFIG_TETRAHEDRON_XYZ,
    UNIT_CONFIG_TETRAHEDRON_ZX,
)
from swarmbots.scenario_presets import scenario_presets_kwargs as shared_kwargs
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    WallDifficulty,
    make_scenario_kwargs,
    wall_scenario_kwargs_with_difficulty,
)

ScenarioT = TypeVar("ScenarioT")
JointPreset = Literal["zx", "xy", "xyz"]
UnitStartLocationsArg = UnitStartLocations | str

DEFAULT_KWARGS = make_scenario_kwargs(shared_kwargs.COMMON_SCENARIO_KWARGS, {"force_elliptic_cone": False})
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


def _default_unit_start_locations() -> PreConnectedUnitLocationsConfig:
    return PreConnectedUnitLocationsConfig(
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
    swarm: BaseSwarm | None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = None,
    joints: JointPreset = "zx",
) -> BaseSwarm:
    if swarm is not None:
        if unit_start_locations is not None:
            raise ValueError("unit_start_locations can only be used when presets construct the swarm")
        return swarm

    return HomogeneousSwarm(
        **_JOINT_CONFIGS[joints],
        unit_start_locations=unit_start_locations or _default_unit_start_locations(),
        body_radius=0.1,
        leg_length=0.2,
        leg_radius=0.025,
        connection_torquescale=50.0,
        quantize_connection_twist=quantize_connection_twist,
        randomize_unit_orientations=randomize_unit_orientations,
    )


def _preset_kwargs(*parts: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    return make_scenario_kwargs(DEFAULT_KWARGS, *parts, overrides)


def _make_scenario(
    scenario_cls: Callable[..., ScenarioT],
    *,
    scenario_kwargs: Mapping[str, object],
    seed: int | None,
    swarm: BaseSwarm | None,
    unit_start_locations: UnitStartLocationsArg | None,
    randomize_unit_orientations: bool,
    quantize_connection_twist: int | None,
    joints: JointPreset,
) -> ScenarioT:
    return scenario_cls(
        swarm=_resolve_swarm(
            swarm=swarm,
            unit_start_locations=unit_start_locations,
            randomize_unit_orientations=randomize_unit_orientations,
            quantize_connection_twist=quantize_connection_twist,
            joints=joints,
        ),
        **scenario_kwargs,
        seed=seed,
    )


def default_wall(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    difficulty: WallDifficulty | None = None,
    **kwargs: Any,
) -> WallScenario:
    return _make_scenario(
        WallScenario,
        scenario_kwargs=_preset_kwargs(wall_scenario_kwargs_with_difficulty(difficulty), overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def easy_wall(**kwargs: Any) -> WallScenario:
    return default_wall(difficulty="easy", **kwargs)


def medium_wall(**kwargs: Any) -> WallScenario:
    return default_wall(difficulty="medium", **kwargs)


def hard_wall(**kwargs: Any) -> WallScenario:
    return default_wall(difficulty="hard", **kwargs)


def default_bridge(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = None,
    **kwargs: Any,
) -> BridgeScenario:
    return _make_scenario(
        BridgeScenario,
        scenario_kwargs=_preset_kwargs(BRIDGE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints="zx",
    )


def default_find_opening(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> FindOpeningScenario:
    return _make_scenario(
        FindOpeningScenario,
        scenario_kwargs=_preset_kwargs(FIND_OPENING_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_climb(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> ClimbScenario:
    return _make_scenario(
        ClimbScenario,
        scenario_kwargs=_preset_kwargs(CLIMB_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_vertical_reach(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> VerticalReachScenario:
    return _make_scenario(
        VerticalReachScenario,
        scenario_kwargs=_preset_kwargs(VERTICAL_REACH_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_payload_plane(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> PayloadPlaneScenario:
    return _make_scenario(
        PayloadPlaneScenario,
        scenario_kwargs=_preset_kwargs(PAYLOAD_PLANE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_dual_payload_plane(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> DualPayloadPlaneScenario:
    return _make_scenario(
        DualPayloadPlaneScenario,
        scenario_kwargs=_preset_kwargs(DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_multi_payload_goal(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> MultiPayloadGoalScenario:
    return _make_scenario(
        MultiPayloadGoalScenario,
        scenario_kwargs=_preset_kwargs(MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_payload_step(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> PayloadStepScenario:
    return _make_scenario(
        PayloadStepScenario,
        scenario_kwargs=_preset_kwargs(PAYLOAD_STEP_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )


def default_move_to(
    seed: int | None = None,
    swarm: BaseSwarm | None = None,
    unit_start_locations: UnitStartLocationsArg | None = None,
    randomize_unit_orientations: bool = False,
    quantize_connection_twist: int | None = 8,
    joints: JointPreset = "zx",
    **kwargs: Any,
) -> MoveToScenario:
    return _make_scenario(
        MoveToScenario,
        scenario_kwargs=_preset_kwargs(MOVE_TO_SCENARIO_KWARGS, overrides=kwargs),
        seed=seed,
        swarm=swarm,
        unit_start_locations=unit_start_locations,
        randomize_unit_orientations=randomize_unit_orientations,
        quantize_connection_twist=quantize_connection_twist,
        joints=joints,
    )
