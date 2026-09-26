from dataclasses import dataclass
from typing import Optional

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor


@dataclass(slots=True)
class PPORolloutBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: MaybeTensor
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    bootstrap_local_obs: torch.Tensor
    bootstrap_global_obs: torch.Tensor
    bootstrap_hidden_local_vars: torch.Tensor
    bootstrap_hidden_global_vars: torch.Tensor
    bootstrap_agent_mask: MaybeTensor
    bootstrap_values: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    dones: torch.Tensor
    episode_start_mask: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    @property
    def n_envs(self) -> int:
        return int(self.local_obs.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.local_obs.shape[1])

    @property
    def n_samples(self) -> int:
        return self.n_envs * self.n_steps

    def to(
            self,
            *,
            device: torch.device | str,
            dtype: torch.dtype,
    ) -> "PPORolloutBatch":
        return PPORolloutBatch(
            local_obs=self.local_obs.to(device=device, dtype=dtype),
            global_obs=self.global_obs.to(device=device, dtype=dtype),
            hidden_local_vars=self.hidden_local_vars.to(device=device, dtype=dtype),
            hidden_global_vars=self.hidden_global_vars.to(device=device, dtype=dtype),
            agent_mask=None if self.agent_mask is None else self.agent_mask.to(device=device),
            previous_actions=(
                None if self.previous_actions is None else self.previous_actions.to(device=device, dtype=dtype)
            ),
            actions=self.actions.to(device=device, dtype=dtype),
            rewards=self.rewards.to(device=device, dtype=dtype),
            log_probs=self.log_probs.to(device=device, dtype=dtype),
            values=self.values.to(device=device, dtype=dtype),
            bootstrap_local_obs=self.bootstrap_local_obs.to(device=device, dtype=dtype),
            bootstrap_global_obs=self.bootstrap_global_obs.to(device=device, dtype=dtype),
            bootstrap_hidden_local_vars=self.bootstrap_hidden_local_vars.to(device=device, dtype=dtype),
            bootstrap_hidden_global_vars=self.bootstrap_hidden_global_vars.to(device=device, dtype=dtype),
            bootstrap_agent_mask=(
                None if self.bootstrap_agent_mask is None else self.bootstrap_agent_mask.to(device=device)
            ),
            bootstrap_values=self.bootstrap_values.to(device=device, dtype=dtype),
            terminations=self.terminations.to(device=device),
            truncations=self.truncations.to(device=device),
            dones=self.dones.to(device=device),
            episode_start_mask=self.episode_start_mask.to(device=device),
            returns=self.returns.to(device=device, dtype=dtype),
            advantages=self.advantages.to(device=device, dtype=dtype),
        )


class PPORolloutBatchBuilder:
    def __init__(
            self,
            *,
            n_envs: int,
            n_steps: int,
            n_agents: int,
            agent_obs_shape: tuple[int, ...],
            global_obs_shape: tuple[int, ...],
            hidden_local_vars_shape: tuple[int, ...],
            hidden_global_vars_shape: tuple[int, ...],
            n_agent_actions: int,
            has_agent_mask: bool,
            has_previous_actions: bool,
            gamma: float,
            gae_lambda: float,
            storage_device: torch.device | str,
            storage_dtype: torch.dtype,
    ) -> None:
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.local_obs = torch.empty(
            (n_envs, n_steps, n_agents, *agent_obs_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.global_obs = torch.empty(
            (n_envs, n_steps, *global_obs_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.hidden_local_vars = torch.empty(
            (n_envs, n_steps, n_agents, *hidden_local_vars_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.hidden_global_vars = torch.empty(
            (n_envs, n_steps, *hidden_global_vars_shape),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.agent_mask: Optional[torch.Tensor] = (
            torch.empty((n_envs, n_steps, n_agents), dtype=torch.bool, device=storage_device)
            if has_agent_mask
            else None
        )
        self.previous_actions: Optional[torch.Tensor] = (
            torch.empty(
                (n_envs, n_steps, n_agents, n_agent_actions),
                dtype=storage_dtype,
                device=storage_device,
            )
            if has_previous_actions
            else None
        )
        self.actions = torch.empty(
            (n_envs, n_steps, n_agents, n_agent_actions),
            dtype=storage_dtype,
            device=storage_device,
        )
        self.rewards = torch.empty((n_envs, n_steps), dtype=storage_dtype, device=storage_device)
        self.log_probs = torch.empty((n_envs, n_steps, n_agents), dtype=storage_dtype, device=storage_device)
        self.values = torch.empty((n_envs, n_steps), dtype=storage_dtype, device=storage_device)

        self.bootstrap_local_obs = torch.empty_like(self.local_obs)
        self.bootstrap_global_obs = torch.empty_like(self.global_obs)
        self.bootstrap_hidden_local_vars = torch.empty_like(self.hidden_local_vars)
        self.bootstrap_hidden_global_vars = torch.empty_like(self.hidden_global_vars)
        self.bootstrap_agent_mask: Optional[torch.Tensor] = (
            torch.empty_like(self.agent_mask)
            if self.agent_mask is not None
            else None
        )
        self.bootstrap_values = torch.empty_like(self.values)

        self.terminations = torch.empty((n_envs, n_steps), dtype=torch.bool, device=storage_device)
        self.truncations = torch.empty_like(self.terminations)
        self.dones = torch.empty_like(self.terminations)
        self.episode_start_mask = torch.empty_like(self.terminations)

    def add(
            self,
            *,
            step_idx: int,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: MaybeTensor,
            previous_actions: MaybeTensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            bootstrap_obs: dict[str, torch.Tensor],
            bootstrap_values: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            dones: torch.Tensor,
            episode_start_mask: torch.Tensor,
    ) -> None:
        self.local_obs[:, step_idx] = local_obs
        self.global_obs[:, step_idx] = global_obs
        self.hidden_local_vars[:, step_idx] = hidden_local_vars
        self.hidden_global_vars[:, step_idx] = hidden_global_vars
        if self.agent_mask is not None:
            if agent_mask is None:
                raise ValueError("agent_mask must be provided when the rollout batch has agent masks")
            self.agent_mask[:, step_idx] = agent_mask
        if self.previous_actions is not None:
            if previous_actions is None:
                self.previous_actions[:, step_idx].zero_()
            else:
                self.previous_actions[:, step_idx] = previous_actions
        self.actions[:, step_idx] = actions
        self.rewards[:, step_idx] = rewards
        self.log_probs[:, step_idx] = log_probs
        self.values[:, step_idx] = values

        self.bootstrap_local_obs[:, step_idx] = bootstrap_obs["local_obs"]
        self.bootstrap_global_obs[:, step_idx] = bootstrap_obs["global_obs"]
        self.bootstrap_hidden_local_vars[:, step_idx] = bootstrap_obs["hidden_local_vars"]
        self.bootstrap_hidden_global_vars[:, step_idx] = bootstrap_obs["hidden_global_vars"]
        if self.bootstrap_agent_mask is not None:
            bootstrap_agent_mask = bootstrap_obs.get("agent_mask", None)
            if bootstrap_agent_mask is None:
                raise ValueError("bootstrap agent_mask must be provided when the rollout batch has agent masks")
            self.bootstrap_agent_mask[:, step_idx] = bootstrap_agent_mask
        self.bootstrap_values[:, step_idx] = bootstrap_values

        self.terminations[:, step_idx] = terminations
        self.truncations[:, step_idx] = truncations
        self.dones[:, step_idx] = dones
        self.episode_start_mask[:, step_idx] = episode_start_mask

    def build(self) -> PPORolloutBatch:
        advantages = compute_step_rollout_gae(
            rewards=self.rewards,
            values=self.values,
            bootstrap_values=self.bootstrap_values,
            dones=self.dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        returns = advantages + self.values
        return PPORolloutBatch(
            local_obs=self.local_obs,
            global_obs=self.global_obs,
            hidden_local_vars=self.hidden_local_vars,
            hidden_global_vars=self.hidden_global_vars,
            agent_mask=self.agent_mask,
            previous_actions=self.previous_actions,
            actions=self.actions,
            rewards=self.rewards,
            log_probs=self.log_probs,
            values=self.values,
            bootstrap_local_obs=self.bootstrap_local_obs,
            bootstrap_global_obs=self.bootstrap_global_obs,
            bootstrap_hidden_local_vars=self.bootstrap_hidden_local_vars,
            bootstrap_hidden_global_vars=self.bootstrap_hidden_global_vars,
            bootstrap_agent_mask=self.bootstrap_agent_mask,
            bootstrap_values=self.bootstrap_values,
            terminations=self.terminations,
            truncations=self.truncations,
            dones=self.dones,
            episode_start_mask=self.episode_start_mask,
            returns=returns,
            advantages=advantages,
        )


def compute_step_rollout_gae(
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
) -> torch.Tensor:
    advantages = torch.empty_like(rewards)
    last_gae_lam = torch.zeros_like(rewards[:, 0])
    n_steps = rewards.shape[1]

    for step_idx in reversed(range(n_steps)):
        if step_idx == n_steps - 1:
            next_values = bootstrap_values[:, step_idx]
        else:
            next_values = torch.where(dones[:, step_idx], bootstrap_values[:, step_idx], values[:, step_idx + 1])
        next_advantage_mask = (~dones[:, step_idx]).to(dtype=rewards.dtype)
        delta = rewards[:, step_idx] + gamma * next_values - values[:, step_idx]
        last_gae_lam = delta + gamma * gae_lambda * next_advantage_mask * last_gae_lam
        advantages[:, step_idx] = last_gae_lam

    return advantages
