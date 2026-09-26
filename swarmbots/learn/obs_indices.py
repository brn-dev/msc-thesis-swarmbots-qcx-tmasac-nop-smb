
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObsIndices:
    local_scalar_indices: list[int]
    local_angle_indices: list[int]
    local_rot6d_indices: list[int]
    local_binary_indices: list[int]
    local_quaternion_indices: list[int]
    global_scalar_indices: list[int]
    global_rot6d_indices: list[int]
    global_quaternion_indices: list[int]
    hidden_local_vars_scalar_indices: list[int]
    hidden_local_vars_quaternion_indices: list[int]
    hidden_global_vars_scalar_indices: list[int]
    hidden_global_vars_quaternion_indices: list[int]
