from typing import Any

from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.scenario_presets.scenario_obs_layouts import (
    CLIMB_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
    DUAL_PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
    MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT,
    PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
    VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
)


def _hinges_per_limb(limb_type: str) -> int:
    name = limb_type.removeprefix("LimbType.")
    if name in ("xy", "zx"):
        return 2
    if name == "xyz":
        return 3
    raise ValueError(f"Unsupported limb type: {limb_type}")


def _global_scalar_indices(scenario_settings: dict[str, Any], global_obs_dim: int) -> list[int]:
    if global_obs_dim == 0:
        return []

    if scenario_settings.get("global_obs_adapter") == PAYLOAD_GLOBAL_OBS_ADAPTER_NAME:
        if global_obs_dim != 9:
            raise ValueError(f"Unexpected adapted move-to global_obs_dim for obs indices: {global_obs_dim}")
        return [0, 1, 2]

    if scenario_settings.get("global_obs_adapter") == DUAL_PAYLOAD_GLOBAL_OBS_ADAPTER_NAME:
        if global_obs_dim != 18:
            raise ValueError(f"Unexpected adapted move-to dual-payload global_obs_dim for obs indices: {global_obs_dim}")
        return [0, 1, 2, 9, 10, 11]

    if scenario_settings.get("global_obs_layout") == MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT:
        num_payloads = int(scenario_settings["num_payloads"])
        expected_global_obs_dim = num_payloads * 12
        if global_obs_dim != expected_global_obs_dim:
            raise ValueError(f"Unexpected multi-payload goal global_obs_dim for obs indices: {global_obs_dim}")
        scalar_indices = []
        for payload_idx in range(num_payloads):
            offset = payload_idx * 12
            scalar_indices.extend(
                [offset + 1, offset + 2, offset + 3, offset + 10, offset + 11]
            )
        return scalar_indices

    if "payload_shape" in scenario_settings:
        expected_global_obs_dim = 18 if scenario_settings.get("num_payloads") == 2 else 9
        if global_obs_dim != expected_global_obs_dim:
            raise ValueError(f"Unexpected payload global_obs_dim for obs indices: {global_obs_dim}")
        if expected_global_obs_dim == 18:
            return [0, 1, 2, 9, 10, 11]
        return [0, 1, 2]

    if "goal" in scenario_settings:
        if global_obs_dim != 2:
            raise ValueError(f"Unexpected move-to global_obs_dim for obs indices: {global_obs_dim}")
        return [0, 1]

    if scenario_settings.get("global_obs_layout") in (
        CLIMB_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
        VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT,
    ):
        if global_obs_dim != 3:
            raise ValueError(f"Unexpected xyz goal global_obs_dim for obs indices: {global_obs_dim}")
        return [0, 1, 2]

    raise ValueError(f"Unsupported non-empty global_obs layout for obs indices: {global_obs_dim}")


def _global_rot6d_indices(scenario_settings: dict[str, Any], global_obs_dim: int) -> list[int]:
    if global_obs_dim == 0:
        return []
    if scenario_settings.get("global_obs_adapter") in (
        PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
        DUAL_PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
    ):
        return []
    if scenario_settings.get("global_obs_layout") == MULTI_PAYLOAD_GOAL_GLOBAL_OBS_LAYOUT:
        num_payloads = int(scenario_settings["num_payloads"])
        expected_global_obs_dim = num_payloads * 12
        if global_obs_dim != expected_global_obs_dim:
            raise ValueError(f"Unexpected multi-payload goal global_obs_dim for obs indices: {global_obs_dim}")
        return [payload_idx * 12 + 4 for payload_idx in range(num_payloads)]
    if "payload_shape" not in scenario_settings:
        return []

    expected_global_obs_dim = 18 if scenario_settings.get("num_payloads") == 2 else 9
    if global_obs_dim != expected_global_obs_dim:
        raise ValueError(f"Unexpected payload global_obs_dim for obs indices: {global_obs_dim}")
    if expected_global_obs_dim == 18:
        return [3, 12]
    return [3]


def _hidden_global_scalar_indices(scenario_settings: dict[str, Any], hidden_global_vars_dim: int) -> list[int]:
    if hidden_global_vars_dim == 0:
        return []
    is_find_opening = (
        scenario_settings.get("scenario_type") == "find_opening"
        or ("opening_x" in scenario_settings and "wall_exploration_cell_count" in scenario_settings)
    )
    if is_find_opening:
        expected_hidden_global_vars_dim = 1 + int(scenario_settings["wall_exploration_cell_count"])
        if hidden_global_vars_dim != expected_hidden_global_vars_dim:
            raise ValueError(
                f"Unexpected find-opening hidden global vars dim for obs indices: {hidden_global_vars_dim}"
            )
        return [0]
    if "payload_shape" in scenario_settings and not scenario_settings.get("payload_pos_observable", True):
        expected_hidden_global_vars_dim = 18 if scenario_settings.get("num_payloads") == 2 else 9
        if hidden_global_vars_dim != expected_hidden_global_vars_dim:
            raise ValueError(
                f"Unexpected hidden payload global vars dim for obs indices: {hidden_global_vars_dim}"
            )
        if expected_hidden_global_vars_dim == 18:
            return [0, 1, 2, 9, 10, 11]
        return [0, 1, 2]
    return list(range(hidden_global_vars_dim))


def build_obs_indices(
    env_settings: dict[str, Any],
    local_obs_dim: int,
    global_obs_dim: int,
    hidden_local_vars_dim: int,
    hidden_global_vars_dim: int,
) -> ObsIndices:
    scenario_settings = env_settings["scenario"]
    quat_rot6d_representation = bool(scenario_settings.get("quat_rot6d_representation", False))

    swarm_config = scenario_settings["swarm"]["config"]
    unit_config = swarm_config["unit_config"]
    limbs_per_unit = len(unit_config)
    include_connectors_xpos_in_obs = bool(scenario_settings["include_connectors_xpos_in_obs"])
    include_connectors_xquat_in_obs = bool(scenario_settings["include_connectors_xquat_in_obs"])

    num_hinges = sum(_hinges_per_limb(limb["type"]) for limb in unit_config)
    free_joint_rot_dim = 6 if quat_rot6d_representation else 4
    qpos_obs_dim = 3 + free_joint_rot_dim + 2 * num_hinges
    qvel_dim = 6 + num_hinges
    connector_obs_dim = limbs_per_unit * 5
    connectors_xpos_dim = limbs_per_unit * 3 if include_connectors_xpos_in_obs else 0
    connectors_xquat_dim = (
        limbs_per_unit * free_joint_rot_dim if include_connectors_xquat_in_obs else 0
    )
    expected_local_dim = (
        qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim + connectors_xquat_dim
    )
    if local_obs_dim != expected_local_dim:
        raise ValueError(
            "Unexpected local_obs_dim for obs indices. "
            f"{local_obs_dim=} {expected_local_dim=} {limbs_per_unit=} {num_hinges=}"
        )

    connector_obs_offset = qpos_obs_dim + qvel_dim
    connectors_xpos_offset = connector_obs_offset + connector_obs_dim
    connectors_xquat_offset = connectors_xpos_offset + connectors_xpos_dim

    local_scalar_indices = (
        list(range(3))
        + list(range(qpos_obs_dim, qpos_obs_dim + qvel_dim))
        + list(range(connectors_xpos_offset, connectors_xpos_offset + connectors_xpos_dim))
        + [connector_obs_offset + i * 5 + 4 for i in range(limbs_per_unit)]
    )
    local_quaternion_indices: list[int] = []
    if not quat_rot6d_representation:
        local_quaternion_indices.append(3)
        if include_connectors_xquat_in_obs:
            local_quaternion_indices.extend(
                connectors_xquat_offset + 4 * i for i in range(limbs_per_unit)
            )

    global_scalar_indices = _global_scalar_indices(scenario_settings, global_obs_dim)
    global_rot6d_indices = _global_rot6d_indices(scenario_settings, global_obs_dim)
    global_quaternion_indices = []

    # hidden_local_vars are currently used for per-unit binary threshold flags in wall scenarios.
    # Keep them unnormalized.
    hidden_local_vars_scalar_indices: list[int] = []
    hidden_local_vars_quaternion_indices: list[int] = []
    hidden_global_vars_scalar_indices = _hidden_global_scalar_indices(scenario_settings, hidden_global_vars_dim)
    hidden_global_vars_quaternion_indices: list[int] = []

    hinge_sin_start = 3 + free_joint_rot_dim
    local_angle_indices = list(range(hinge_sin_start, hinge_sin_start + 2 * num_hinges, 2))
    local_angle_indices.extend(connector_obs_offset + i * 5 + 2 for i in range(limbs_per_unit))

    local_rot6d_indices: list[int] = []
    if quat_rot6d_representation:
        local_rot6d_indices.append(3)
        if include_connectors_xquat_in_obs:
            local_rot6d_indices.extend(connectors_xquat_offset + 6 * i for i in range(limbs_per_unit))

    local_binary_indices = [connector_obs_offset + i * 5 + 0 for i in range(limbs_per_unit)]

    return ObsIndices(
        local_scalar_indices=local_scalar_indices,
        local_angle_indices=local_angle_indices,
        local_rot6d_indices=local_rot6d_indices,
        local_binary_indices=local_binary_indices,
        local_quaternion_indices=local_quaternion_indices,
        global_scalar_indices=global_scalar_indices,
        global_rot6d_indices=global_rot6d_indices,
        global_quaternion_indices=global_quaternion_indices,
        hidden_local_vars_scalar_indices=hidden_local_vars_scalar_indices,
        hidden_local_vars_quaternion_indices=hidden_local_vars_quaternion_indices,
        hidden_global_vars_scalar_indices=hidden_global_vars_scalar_indices,
        hidden_global_vars_quaternion_indices=hidden_global_vars_quaternion_indices,
    )
