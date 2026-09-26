from typing import Union

import mujoco
import numpy as np


def quat_z2vec(vec: Union[list, tuple, np.array]):
    quat = np.empty(4)
    mujoco.mju_quatZ2Vec(quat, vec)
    return quat

def body_ids_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """Return all body ids whose names start with prefix."""
    ids: list[int] = []
    for i in range(model.nbody):
        name = model.body(i).name
        if name and name.startswith(prefix):
            ids.append(i)
    return ids


def actuator_ids_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """Return all actuator ids whose names start with prefix."""
    ids: list[int] = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if name and name.startswith(prefix):
            ids.append(i)
    return ids

def qpos_indices_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    """
    Indices into data.qpos for all joints attached to body_name.
    If the body has no joints, returns [].
    """
    body_id = model.body(body_name).id
    jadr = model.body_jntadr[body_id]
    jnum = model.body_jntnum[body_id]

    if jadr < 0 or jnum == 0:
        return []

    # MuJoCo joint types: free=0, ball=1, slide=2, hinge=3
    # qpos sizes: free=7, ball=4, slide=1, hinge=1
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}

    out: set[int] = set()
    for jnt_id in range(jadr, jadr + jnum):
        qadr = model.jnt_qposadr[jnt_id]
        w = qpos_width[int(model.jnt_type[jnt_id])]
        out.update(range(qadr, qadr + w))

    return sorted(out)


def qpos_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """
    Union of qpos indices for all bodies whose names start with prefix.
    Useful if you name subtrees like 'robot1/'.
    """
    out: set[int] = set()
    for bid in body_ids_for_prefix(model, prefix):
        out.update(qpos_indices_for_body(model, model.body(bid).name))
    return sorted(out)


def dof_indices_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    """
    Indices into data.qvel (degrees of freedom) for all joints attached to body_name.
    """
    body_id = model.body(body_name).id
    dof_adr = model.body_dofadr[body_id]
    dof_num = model.body_dofnum[body_id]

    if dof_adr < 0 or dof_num == 0:
        return []

    return list(range(dof_adr, dof_adr + dof_num))


def dof_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """
    Union of dof (qvel) indices for all bodies whose names start with prefix.
    """
    out: set[int] = set()
    for bid in body_ids_for_prefix(model, prefix):
        out.update(dof_indices_for_body(model, model.body(bid).name))
    return sorted(out)

def ctrl_index_for_actuator(model: mujoco.MjModel, actuator_name: str) -> int:
    """Single ctrl index for a named actuator."""
    return model.actuator(actuator_name).id


def ctrl_indices_for_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    """All ctrl indices for actuators whose names start with prefix."""
    return actuator_ids_for_prefix(model, prefix)

def apply_twist(model: mujoco.MjModel, eq_idx: int, twist: float):
    model.eq_data[eq_idx, 7] = np.cos(twist / 2)
    model.eq_data[eq_idx, 8] = np.sin(twist / 2)

