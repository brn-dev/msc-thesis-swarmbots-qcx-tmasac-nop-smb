from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
from torch.nn import functional as F


BootstrapObservations = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]
OptimizerStep = Callable[[], None]
CompiledReturn = TypeVar("CompiledReturn")


def _mask_terminal_observations(
        terminal_mask: torch.Tensor,
        next_local_obs: torch.Tensor,
        next_global_obs: torch.Tensor,
        next_hidden_local_vars: torch.Tensor,
        next_hidden_global_vars: torch.Tensor,
        next_agent_mask: torch.Tensor | None,
) -> BootstrapObservations:
    def mask_rows(tensor: torch.Tensor, value: bool | float) -> torch.Tensor:
        row_mask = terminal_mask.reshape(
            *terminal_mask.shape,
            *((1,) * (tensor.ndim - terminal_mask.ndim)),
        )
        return tensor.masked_fill(row_mask, value)

    return (
        mask_rows(next_local_obs, 0.0),
        mask_rows(next_global_obs, 0.0),
        mask_rows(next_hidden_local_vars, 0.0),
        mask_rows(next_hidden_global_vars, 0.0),
        None if next_agent_mask is None else mask_rows(next_agent_mask, True),
    )


def _mean_agent_log_probs(
        log_probs: torch.Tensor,
        agent_mask: torch.Tensor | None,
) -> torch.Tensor:
    if agent_mask is None:
        return log_probs.mean(dim=-1)
    log_prob_sums = log_probs.masked_fill(~agent_mask, 0.0).sum(dim=-1)
    active_agent_counts = agent_mask.to(dtype=log_probs.dtype).sum(dim=-1)
    return log_prob_sums / active_agent_counts


def _entropy_coefficient_loss(
        log_ent_coef: torch.Tensor,
        log_prob_mean: torch.Tensor,
        target_entropy: torch.Tensor,
) -> torch.Tensor:
    return -(log_ent_coef * (log_prob_mean + target_entropy).detach()).mean()


def _bellman_target(
        rewards: torch.Tensor,
        terminal_mask: torch.Tensor,
        target_q1: torch.Tensor,
        target_q2: torch.Tensor,
        next_log_prob_mean: torch.Tensor,
        ent_coef: torch.Tensor,
        gamma: float,
) -> torch.Tensor:
    next_q = torch.minimum(target_q1, target_q2) - ent_coef * next_log_prob_mean
    return torch.where(
        terminal_mask,
        rewards,
        rewards + gamma * next_q,
    )


def _critic_loss(
        current_q1: torch.Tensor,
        current_q2: torch.Tensor,
        target_q: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        F.mse_loss(current_q1, target_q)
        + F.mse_loss(current_q2, target_q)
    )


def _actor_loss(
        q1_pi: torch.Tensor,
        q2_pi: torch.Tensor,
        log_prob_mean: torch.Tensor,
        ent_coef: torch.Tensor,
) -> torch.Tensor:
    return (ent_coef * log_prob_mean - torch.minimum(q1_pi, q2_pi)).mean()


@dataclass(frozen=True, slots=True)
class SACTensorOperations:
    mask_terminal_observations: Callable[..., BootstrapObservations]
    mean_agent_log_probs: Callable[..., torch.Tensor]
    entropy_coefficient_loss: Callable[..., torch.Tensor]
    bellman_target: Callable[..., torch.Tensor]
    critic_loss: Callable[..., torch.Tensor]
    actor_loss: Callable[..., torch.Tensor]


def build_sac_tensor_operations(
        *,
        compile_operations: bool,
        compile_mode: str,
) -> SACTensorOperations:
    operations = SACTensorOperations(
        mask_terminal_observations=_mask_terminal_observations,
        mean_agent_log_probs=_mean_agent_log_probs,
        entropy_coefficient_loss=_entropy_coefficient_loss,
        bellman_target=_bellman_target,
        critic_loss=_critic_loss,
        actor_loss=_actor_loss,
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

    return SACTensorOperations(
        mask_terminal_observations=compile_operation(operations.mask_terminal_observations),
        mean_agent_log_probs=compile_operation(operations.mean_agent_log_probs),
        entropy_coefficient_loss=compile_operation(operations.entropy_coefficient_loss),
        bellman_target=compile_operation(operations.bellman_target),
        critic_loss=compile_operation(operations.critic_loss),
        actor_loss=compile_operation(operations.actor_loss),
    )


def build_optimizer_step(
        optimizer: torch.optim.Optimizer,
        *,
        compile_step: bool,
        compile_mode: str,
) -> OptimizerStep:
    if not compile_step:
        return optimizer.step
    _validate_compile_configuration(compile_mode)

    def optimizer_step() -> None:
        optimizer.step()

    return torch.compile(
        optimizer_step,
        mode=compile_mode,
        fullgraph=False,
        dynamic=False,
    )


def _validate_compile_configuration(compile_mode: str) -> None:
    if not hasattr(torch, "compile") or not callable(torch.compile):
        raise RuntimeError("Compiling SAC training operations requires torch.compile support.")
    if not compile_mode:
        raise ValueError("sac_compile_mode must be a non-empty string when SAC compilation is enabled.")
