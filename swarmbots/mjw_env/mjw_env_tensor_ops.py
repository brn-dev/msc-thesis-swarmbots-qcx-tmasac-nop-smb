from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

from swarmbots.mjw_env.mjw_torch_quat import quat_to_rot6d_torch


MaskedObservations = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
FinalizedStep = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
CompiledReturn = TypeVar("CompiledReturn")


@dataclass(frozen=True, slots=True)
class MJWObservationLayout:
    num_envs: int
    num_agents: int
    num_connectors: int
    free_joint_position: slice
    free_joint_rotation: slice
    hinge: slice
    qvel: slice
    connector: slice
    connector_position: slice
    use_rot6d: bool
    include_connector_positions: bool


@dataclass(frozen=True, slots=True)
class MJWActionLayout:
    num_envs: int
    continuous_connectors: bool


def _build_local_obs(
    local_obs: torch.Tensor,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    partner_unit: torch.Tensor,
    connection_twist_idx: torch.Tensor,
    disconnect_potentials: torch.Tensor,
    twist_values: torch.Tensor,
    xpos: torch.Tensor,
    qpos_flat_indices: torch.Tensor,
    qvel_flat_indices: torch.Tensor,
    connector_body_indices: torch.Tensor,
    *,
    layout: MJWObservationLayout,
) -> torch.Tensor:
    unit_qpos = qpos[:, qpos_flat_indices].reshape(layout.num_envs, layout.num_agents, -1)
    unit_qvel = qvel[:, qvel_flat_indices].reshape(layout.num_envs, layout.num_agents, -1)
    local_obs[:, :, layout.free_joint_position] = unit_qpos[:, :, :3]
    free_joint_quat = unit_qpos[:, :, 3:7]
    if layout.use_rot6d:
        local_obs[:, :, layout.free_joint_rotation] = quat_to_rot6d_torch(free_joint_quat)
    else:
        local_obs[:, :, layout.free_joint_rotation] = free_joint_quat

    hinge_qpos = unit_qpos[:, :, 7:]
    hinge_obs = local_obs[:, :, layout.hinge].view(layout.num_envs, layout.num_agents, -1, 2)
    hinge_obs[..., 0] = torch.sin(hinge_qpos)
    hinge_obs[..., 1] = torch.cos(hinge_qpos)
    local_obs[:, :, layout.qvel] = unit_qvel

    connector_active = partner_unit >= 0
    valid_twist = connector_active & (connection_twist_idx >= 0)
    connector_twists = twist_values[connection_twist_idx.clamp_min(0)]
    connector_obs = local_obs[:, :, layout.connector].view(
        layout.num_envs,
        layout.num_agents,
        layout.num_connectors,
        5,
    )
    valid_twist_float = valid_twist.to(dtype=local_obs.dtype)
    connector_obs[..., 0] = (~connector_active).to(dtype=local_obs.dtype)
    connector_obs[..., 1] = connector_active.to(dtype=local_obs.dtype)
    connector_obs[..., 2] = torch.sin(connector_twists) * valid_twist_float
    connector_obs[..., 3] = torch.cos(connector_twists) * valid_twist_float
    connector_obs[..., 4] = disconnect_potentials

    if layout.include_connector_positions:
        local_obs[:, :, layout.connector_position] = xpos[:, connector_body_indices].reshape(
            layout.num_envs,
            layout.num_agents,
            -1,
        )
    return local_obs


def _prepare_actions(
    actuators: torch.Tensor,
    connectors: torch.Tensor,
    units_active_mask: torch.Tensor,
    ctrl: torch.Tensor,
    ctrl_flat_indices: torch.Tensor,
    actuator_strength: float,
    *,
    layout: MJWActionLayout,
) -> torch.Tensor:
    active_agents = units_active_mask.unsqueeze(-1)
    masked_actuators = actuators.masked_fill(~active_agents, 0.0)
    if ctrl.numel() > 0:
        ctrl[:, ctrl_flat_indices] = (
            masked_actuators.reshape(layout.num_envs, -1) * actuator_strength
        )
    if layout.continuous_connectors:
        return connectors.masked_fill(~active_agents, -1.0)
    return connectors & active_agents


def _mask_error_observations(
    local_obs: torch.Tensor,
    global_obs: torch.Tensor,
    hidden_local_obs: torch.Tensor,
    hidden_global_obs: torch.Tensor,
    unstable_mask: torch.Tensor,
) -> MaskedObservations:
    def mask_rows(tensor: torch.Tensor) -> torch.Tensor:
        row_mask = unstable_mask.reshape(
            unstable_mask.shape[0],
            *((1,) * (tensor.ndim - 1)),
        )
        return tensor.masked_fill_(row_mask, 0.0)

    return (
        mask_rows(local_obs),
        mask_rows(global_obs),
        mask_rows(hidden_local_obs),
        mask_rows(hidden_global_obs),
    )


def _begin_step(
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    current_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    unstable_mask = ~torch.isfinite(qpos).all(dim=1) | ~torch.isfinite(qvel).all(dim=1)
    stable_mask = ~unstable_mask
    current_step.add_(stable_mask.to(dtype=current_step.dtype))
    return unstable_mask, stable_mask


def _finalize_step(
    rewards: torch.Tensor,
    unstable_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    scenario_terminations: torch.Tensor,
    current_step: torch.Tensor,
    truncation_limit: torch.Tensor,
    simulation_unstable_reward: float,
) -> FinalizedStep:
    scenario_terminations = scenario_terminations & stable_mask
    terminations = unstable_mask | scenario_terminations
    truncations = stable_mask & ~scenario_terminations & (current_step >= truncation_limit)
    dones = terminations | truncations
    rewards = torch.where(
        unstable_mask,
        torch.as_tensor(simulation_unstable_reward, dtype=rewards.dtype, device=rewards.device),
        rewards,
    )
    return rewards, terminations, truncations, dones


@dataclass(frozen=True, slots=True)
class MJWEnvTensorOperations:
    build_local_obs: Callable[..., torch.Tensor]
    prepare_actions: Callable[..., torch.Tensor]
    mask_error_observations: Callable[..., MaskedObservations]
    begin_step: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    finalize_step: Callable[..., FinalizedStep]


def should_compile_mjw_env_tensor_operations_by_default(device: torch.device) -> bool:
    return device.type == "cuda" and hasattr(torch, "compile") and callable(torch.compile)


def build_mjw_env_tensor_operations(
    *,
    observation_layout: MJWObservationLayout,
    action_layout: MJWActionLayout,
    compile_operations: bool,
    compile_mode: str,
) -> MJWEnvTensorOperations:
    def build_local_obs(*args: torch.Tensor) -> torch.Tensor:
        return _build_local_obs(*args, layout=observation_layout)

    def prepare_actions(
        actuators: torch.Tensor,
        connectors: torch.Tensor,
        units_active_mask: torch.Tensor,
        ctrl: torch.Tensor,
        ctrl_flat_indices: torch.Tensor,
        actuator_strength: float,
    ) -> torch.Tensor:
        return _prepare_actions(
            actuators,
            connectors,
            units_active_mask,
            ctrl,
            ctrl_flat_indices,
            actuator_strength,
            layout=action_layout,
        )

    operations = MJWEnvTensorOperations(
        build_local_obs=build_local_obs,
        prepare_actions=prepare_actions,
        mask_error_observations=_mask_error_observations,
        begin_step=_begin_step,
        finalize_step=_finalize_step,
    )
    if not compile_operations:
        return operations
    _validate_compile_configuration(compile_mode)

    def compile_operation(
        operation: Callable[..., CompiledReturn],
    ) -> Callable[..., CompiledReturn]:
        return torch.compile(
            operation,
            mode=compile_mode,
            fullgraph=True,
            dynamic=False,
        )

    return MJWEnvTensorOperations(
        build_local_obs=compile_operation(operations.build_local_obs),
        prepare_actions=compile_operation(operations.prepare_actions),
        mask_error_observations=compile_operation(operations.mask_error_observations),
        begin_step=compile_operation(operations.begin_step),
        finalize_step=compile_operation(operations.finalize_step),
    )


def _validate_compile_configuration(compile_mode: str) -> None:
    if not hasattr(torch, "compile") or not callable(torch.compile):
        raise RuntimeError("Compiling MJW environment tensor operations requires torch.compile support.")
    if not compile_mode:
        raise ValueError("tensor_operations_compile_mode must be non-empty when MJW tensor compilation is enabled.")
