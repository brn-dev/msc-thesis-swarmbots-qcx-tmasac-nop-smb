from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

import swarmbots.mj_env.mujoco_utils as mj_utils


@dataclass
class MJWModelMetadata:
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    ctrl_indices: np.ndarray
    connector_body_indices: np.ndarray
    eq_indices: np.ndarray
    unit_qpos_adr: np.ndarray
    unit_dof_adr: np.ndarray


def build_model_metadata(model: mujoco.MjModel, scenario: Any) -> MJWModelMetadata:
    unit_prefixes = scenario.swarm.config.unit_prefixes
    qpos_indices = np.asarray(
        [mj_utils.qpos_indices_for_prefix(model, prefix) for prefix in unit_prefixes],
        dtype=np.int64,
    )
    qvel_indices = np.asarray(
        [mj_utils.dof_indices_for_prefix(model, prefix) for prefix in unit_prefixes],
        dtype=np.int64,
    )
    ctrl_indices = np.asarray(
        [mj_utils.ctrl_indices_for_prefix(model, prefix) for prefix in unit_prefixes],
        dtype=np.int64,
    )

    connector_body_indices = np.zeros((scenario.swarm.num_units, scenario.swarm.config.limbs_per_unit), dtype=np.int64)
    for unit_idx in range(scenario.swarm.num_units):
        for connector_idx in range(scenario.swarm.config.limbs_per_unit):
            connector_body_indices[unit_idx, connector_idx] = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                scenario.swarm.config.get_connector_name(unit_idx, connector_idx),
            )

    num_twists = len(scenario.swarm.config.connection_twist_values)
    eq_indices = np.full(
        (
            scenario.swarm.num_units,
            scenario.swarm.config.limbs_per_unit,
            scenario.swarm.num_units,
            scenario.swarm.config.limbs_per_unit,
            num_twists,
        ),
        -1,
        dtype=np.int64,
    )
    for u1 in range(scenario.swarm.num_units - 1):
        for u2 in range(u1 + 1, scenario.swarm.num_units):
            for c1 in range(scenario.swarm.config.limbs_per_unit):
                for c2 in range(scenario.swarm.config.limbs_per_unit):
                    for twist_idx in range(num_twists):
                        eq_id = mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_EQUALITY,
                            scenario.swarm.config.get_eq_variant_name(u1, c1, u2, c2, twist_idx),
                        )
                        eq_indices[u1, c1, u2, c2, twist_idx] = eq_id
                        eq_indices[u2, c2, u1, c1, twist_idx] = eq_id

    unit_qpos_adr = np.zeros((scenario.swarm.num_units,), dtype=np.int64)
    unit_dof_adr = np.zeros((scenario.swarm.num_units,), dtype=np.int64)
    for unit_idx in range(scenario.swarm.num_units):
        body_name = f"{scenario.swarm.config.unit_prefixes[unit_idx]}-main_body"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        jnt_adr = model.body_jntadr[body_id]
        unit_qpos_adr[unit_idx] = model.jnt_qposadr[jnt_adr]
        unit_dof_adr[unit_idx] = model.jnt_dofadr[jnt_adr]

    return MJWModelMetadata(
        qpos_indices=qpos_indices,
        qvel_indices=qvel_indices,
        ctrl_indices=ctrl_indices,
        connector_body_indices=connector_body_indices,
        eq_indices=eq_indices,
        unit_qpos_adr=unit_qpos_adr,
        unit_dof_adr=unit_dof_adr,
    )
