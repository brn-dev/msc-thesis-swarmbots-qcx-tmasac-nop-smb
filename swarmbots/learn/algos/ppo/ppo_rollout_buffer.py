from dataclasses import dataclass
from typing import Any, Optional, Iterator

import torch
from gymnasium import spaces

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.temporal_state import (
    clone_detach_temporal_state,
    copy_temporal_state_rows_,
    index_temporal_state,
    move_temporal_state,
)
from swarmbots.learn.torch_device import as_device

MaybeTensor = Optional[torch.Tensor]


@dataclass
class PPOEpisodeSegment:
    local_obs: torch.Tensor  # (n_steps, n_agents, n_local_obs)
    global_obs: torch.Tensor  # (n_steps, n_global_obs)
    hidden_local_vars: torch.Tensor  # (n_steps, n_agents, n_hidden_local_vars)
    hidden_global_vars: torch.Tensor  # (n_steps, n_hidden_global_vars)
    agent_mask: MaybeTensor  # (n_steps, n_agents)
    actions: torch.Tensor  # (n_steps, n_agents, n_actions_per_agent)
    rewards: MaybeTensor  # (n_steps)
    log_probs: torch.Tensor  # (n_steps, n_agents)
    values: MaybeTensor  # (n_steps)

    final_local_obs: MaybeTensor  # (n_agents, n_local_obs)
    final_global_obs: MaybeTensor  # (n_global_obs)
    final_hidden_local_vars: MaybeTensor  # (n_agents, n_hidden_local_vars)
    final_hidden_global_vars: MaybeTensor  # (n_hidden_global_vars)
    final_agent_mask: MaybeTensor  # (n_agents)
    final_value: MaybeTensor  # (1)
    initial_previous_actions: MaybeTensor = None  # (n_agents, n_actions_per_agent)
    is_true_episode_start: bool = True

    returns: MaybeTensor = None  # (n_steps)
    advantages: MaybeTensor = None  # (n_steps)
    initial_temporal_state: Any = None
    rollout_env_idx: int | None = None
    rollout_start_step: int = 0

    def compute_gae(self, gamma: float, gae_lambda: float) -> None:
        assert self.rewards is not None
        assert self.values is not None
        assert self.final_value is not None

        self.advantages = torch.zeros_like(self.rewards)
        last_gae_lam = 0.0
        num_steps = len(self.rewards)

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_val = self.final_value
            else:
                next_val = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_val - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
            self.advantages[step] = last_gae_lam

        self.returns = self.advantages + self.values


class PPOEpisodeAccumulator:

    def __init__(
            self,
            n_envs: int,
            max_episode_length: int,
            n_agents: int,
            agent_obs_shape: tuple[int, ...],
            global_obs_shape: tuple[int, ...],
            hidden_local_vars_shape: tuple[int, ...],
            hidden_global_vars_shape: tuple[int, ...],
            n_agent_actions: int,
            has_agent_mask: bool,
            storage_device: torch.device | str,
            storage_dtype: torch.dtype
    ):
        self.max_episode_length = max_episode_length
        self.local_obs = torch.zeros(
            (n_envs, max_episode_length, n_agents, *agent_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.global_obs = torch.zeros(
            (n_envs, max_episode_length, *global_obs_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.hidden_local_vars = torch.zeros(
            (n_envs, max_episode_length, n_agents, *hidden_local_vars_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.hidden_global_vars = torch.zeros(
            (n_envs, max_episode_length, *hidden_global_vars_shape),
            dtype=storage_dtype, device=storage_device
        )
        self.agent_mask: MaybeTensor = torch.zeros(
            (n_envs, max_episode_length, n_agents),
            dtype=torch.bool, device=storage_device
        ) if has_agent_mask else None
        self.actions = torch.zeros(
            (n_envs, max_episode_length, n_agents, n_agent_actions),
            dtype=storage_dtype, device=storage_device
        )
        self.log_probs = torch.zeros(
            (n_envs, max_episode_length, n_agents),
            dtype=storage_dtype, device=storage_device
        )
        self.initial_previous_actions = torch.zeros(
            (n_envs, n_agents, n_agent_actions),
            dtype=storage_dtype, device=storage_device
        )
        self.is_true_episode_start = torch.zeros(
            n_envs,
            dtype=torch.bool,
            device=storage_device,
        )
        self.rewards = torch.zeros(
            (n_envs, max_episode_length),
            dtype=storage_dtype, device=storage_device
        )
        self.values = torch.zeros(
            (n_envs, max_episode_length),
            dtype=storage_dtype, device=storage_device
        )
        self.step = torch.zeros(n_envs, dtype=torch.long, device=storage_device)
        self.initial_temporal_state: Any = None
        self.rollout_start_steps = torch.zeros(n_envs, dtype=torch.long, device=storage_device)

    @property
    def total_steps(self) -> int:
        return self.step.sum().item()

    def add(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: MaybeTensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            previous_actions: MaybeTensor,
            episode_start_mask: torch.Tensor,
            temporal_state: Any,
            rollout_step_idx: int,
            bootstrap_obs: dict[str, torch.Tensor],
            bootstrap_values: torch.Tensor,
            dones: torch.Tensor,
    ) -> Iterator[PPOEpisodeSegment]:
        env_indices = torch.arange(self.step.shape[0], device=self.step.device)
        step_indices = self.step
        new_episode_env_indices = torch.where(step_indices == 0)[0]
        if len(new_episode_env_indices) > 0:
            if previous_actions is None:
                self.initial_previous_actions[new_episode_env_indices].zero_()
            else:
                self.initial_previous_actions[new_episode_env_indices] = previous_actions[new_episode_env_indices]
            self.is_true_episode_start[new_episode_env_indices] = episode_start_mask[new_episode_env_indices]
            self.rollout_start_steps[new_episode_env_indices] = rollout_step_idx
            if self.initial_temporal_state is None:
                self.initial_temporal_state = clone_detach_temporal_state(temporal_state)
            else:
                copy_temporal_state_rows_(
                    self.initial_temporal_state,
                    temporal_state,
                    new_episode_env_indices,
                )

        self.local_obs[env_indices, step_indices] = local_obs
        self.global_obs[env_indices, step_indices] = global_obs
        self.hidden_local_vars[env_indices, step_indices] = hidden_local_vars
        self.hidden_global_vars[env_indices, step_indices] = hidden_global_vars
        if agent_mask is not None:
            self.agent_mask[env_indices, step_indices] = agent_mask
        self.actions[env_indices, step_indices] = actions
        self.rewards[env_indices, step_indices] = rewards
        self.log_probs[env_indices, step_indices] = log_probs
        self.values[env_indices, step_indices] = values

        self.step += 1

        bootstrap_agent_mask = bootstrap_obs.get("agent_mask", None)
        done_env_indices = torch.where(dones)[0]
        for done_env_idx in done_env_indices.tolist():
            yield self.construct_episode(
                done_env_idx,
                final_local_obs=bootstrap_obs["local_obs"][done_env_idx],
                final_global_obs=bootstrap_obs["global_obs"][done_env_idx],
                final_hidden_local_vars=bootstrap_obs["hidden_local_vars"][done_env_idx],
                final_hidden_global_vars=bootstrap_obs["hidden_global_vars"][done_env_idx],
                final_agent_mask=None if bootstrap_agent_mask is None else bootstrap_agent_mask[done_env_idx],
                final_value=bootstrap_values[done_env_idx],
            )
            self.initial_previous_actions[done_env_idx].zero_()
            self.is_true_episode_start[done_env_idx] = False
            self.step[done_env_idx] = 0

    def construct_episode(
            self,
            env: int,
            final_local_obs: torch.Tensor,
            final_global_obs: torch.Tensor,
            final_hidden_local_vars: torch.Tensor,
            final_hidden_global_vars: torch.Tensor,
            final_agent_mask: MaybeTensor,
            final_value: torch.Tensor,
            clone_tensors: bool = True,
    ) -> PPOEpisodeSegment:
        step = int(self.step[env].item())
        def maybe_clone(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.clone() if clone_tensors else tensor

        agent_mask = None if self.agent_mask is None else maybe_clone(self.agent_mask[env, :step])
        return PPOEpisodeSegment(
            local_obs=maybe_clone(self.local_obs[env, :step]),
            global_obs=maybe_clone(self.global_obs[env, :step]),
            hidden_local_vars=maybe_clone(self.hidden_local_vars[env, :step]),
            hidden_global_vars=maybe_clone(self.hidden_global_vars[env, :step]),
            agent_mask=agent_mask,
            actions=maybe_clone(self.actions[env, :step]),
            rewards=maybe_clone(self.rewards[env, :step]),
            log_probs=maybe_clone(self.log_probs[env, :step]),
            values=maybe_clone(self.values[env, :step]),
            final_local_obs=maybe_clone(final_local_obs),
            final_global_obs=maybe_clone(final_global_obs),
            final_hidden_local_vars=maybe_clone(final_hidden_local_vars),
            final_hidden_global_vars=maybe_clone(final_hidden_global_vars),
            final_agent_mask=None if final_agent_mask is None else maybe_clone(final_agent_mask),
            final_value=maybe_clone(final_value),
            initial_previous_actions=maybe_clone(self.initial_previous_actions[env]),
            is_true_episode_start=bool(self.is_true_episode_start[env].item()),
            initial_temporal_state=clone_detach_temporal_state(
                index_temporal_state(self.initial_temporal_state, slice(env, env + 1))
            ),
            rollout_env_idx=env,
            rollout_start_step=int(self.rollout_start_steps[env].item()),
        )

    def reset(self) -> None:
        self.initial_previous_actions.zero_()
        self.is_true_episode_start.zero_()
        self.step[:] = 0
        self.initial_temporal_state = None
        self.rollout_start_steps.zero_()


class PPORolloutBuffer:

    def __init__(
            self,
            max_episode_length: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            gamma: float,
            gae_lambda: float,
            rollout_device: torch.device | str = 'cpu',
            rollout_dtype: torch.dtype = torch.float32,
            train_device: torch.device | str = 'auto',
            train_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.max_episode_length = max_episode_length

        self.observation_space = observation_space
        self.action_space = action_space

        self.local_obs_space = observation_space['local_obs']
        self.n_agents = self.local_obs_space.shape[1]
        self.agent_obs_shape = self.local_obs_space.shape[2:]

        self.global_obs_space = observation_space['global_obs']
        self.global_obs_shape = self.global_obs_space.shape[1:]
        self.hidden_local_vars_space = observation_space['hidden_local_vars']
        self.hidden_local_vars_shape = self.hidden_local_vars_space.shape[2:]
        self.hidden_global_vars_space = observation_space['hidden_global_vars']
        self.hidden_global_vars_shape = self.hidden_global_vars_space.shape[1:]
        self.has_agent_mask = 'agent_mask' in observation_space.keys() and observation_space['agent_mask'] is not None

        self.n_envs = self.local_obs_space.shape[0]

        assert action_space.n_agents == self.n_agents
        self.n_agent_actions = action_space.total_agent_action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rollout_device = as_device(rollout_device)
        self.rollout_dtype = rollout_dtype
        self.train_device = as_device(train_device)
        self.train_dtype = train_dtype

        self.episodes = list()
        self.accumulator = PPOEpisodeAccumulator(
            n_envs=self.n_envs,
            max_episode_length=self.max_episode_length,
            n_agents=self.n_agents,
            agent_obs_shape=self.agent_obs_shape,
            global_obs_shape=self.global_obs_shape,
            hidden_local_vars_shape=self.hidden_local_vars_shape,
            hidden_global_vars_shape=self.hidden_global_vars_shape,
            n_agent_actions=self.n_agent_actions,
            has_agent_mask=self.has_agent_mask,
            storage_device=self.rollout_device,
            storage_dtype=self.rollout_dtype,
        )

    def add(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: MaybeTensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            previous_actions: MaybeTensor,
            episode_start_mask: torch.Tensor,
            temporal_state: Any,
            rollout_step_idx: int,
            bootstrap_obs: dict[str, torch.Tensor],
            bootstrap_values: torch.Tensor,
            dones: torch.Tensor,
    ) -> None:
        new_episodes = self.accumulator.add(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            previous_actions=previous_actions,
            episode_start_mask=episode_start_mask,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            bootstrap_obs=bootstrap_obs,
            bootstrap_values=bootstrap_values,
            dones=dones,
        )
        for new_ep in new_episodes:
            new_ep.compute_gae(self.gamma, self.gae_lambda)
            self.episodes.append(new_ep)

    def reset(self) -> None:
        self.accumulator.reset()
        self.episodes = []

    def get_whole_episodes(self) -> list[PPOEpisodeSegment]:
        return self._episodes_to_train_dev(self.episodes)

    def dump_partial_episodes(
            self,
            final_obs: dict[str, torch.Tensor],
            final_values: torch.Tensor,
            clone_tensors: bool = True,
    ) -> list[PPOEpisodeSegment]:
        final_agent_mask = final_obs.get("agent_mask", None)
        partial_episodes: list[PPOEpisodeSegment] = []
        for env_idx in range(self.n_envs):
            step = int(self.accumulator.step[env_idx].item())
            if step == 0:
                continue
            episode = self.accumulator.construct_episode(
                env=env_idx,
                final_local_obs=final_obs["local_obs"][env_idx],
                final_global_obs=final_obs["global_obs"][env_idx],
                final_hidden_local_vars=final_obs["hidden_local_vars"][env_idx],
                final_hidden_global_vars=final_obs["hidden_global_vars"][env_idx],
                final_agent_mask=None if final_agent_mask is None else final_agent_mask[env_idx],
                final_value=final_values[env_idx],
                clone_tensors=clone_tensors,
            )
            episode.compute_gae(self.gamma, self.gae_lambda)
            partial_episodes.append(episode)
        return self._episodes_to_train_dev(partial_episodes)

    def _episodes_to_train_dev(self, episodes: list[PPOEpisodeSegment]) -> list[PPOEpisodeSegment]:
        return [
            PPOEpisodeSegment(
                local_obs=ep.local_obs.to(device=self.train_device, dtype=self.train_dtype),
                global_obs=ep.global_obs.to(device=self.train_device, dtype=self.train_dtype),
                hidden_local_vars=ep.hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
                hidden_global_vars=ep.hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
                agent_mask=None if ep.agent_mask is None else ep.agent_mask.to(device=self.train_device),
                actions=ep.actions.to(device=self.train_device, dtype=self.train_dtype),
                rewards=ep.rewards.to(device=self.train_device, dtype=self.train_dtype),
                log_probs=ep.log_probs.to(device=self.train_device, dtype=self.train_dtype),
                values=ep.values.to(device=self.train_device, dtype=self.train_dtype),
                final_local_obs=ep.final_local_obs.to(device=self.train_device, dtype=self.train_dtype),
                final_global_obs=ep.final_global_obs.to(device=self.train_device, dtype=self.train_dtype),
                final_hidden_local_vars=ep.final_hidden_local_vars.to(device=self.train_device, dtype=self.train_dtype),
                final_hidden_global_vars=ep.final_hidden_global_vars.to(device=self.train_device, dtype=self.train_dtype),
                final_agent_mask=None if ep.final_agent_mask is None else ep.final_agent_mask.to(device=self.train_device),
                final_value=ep.final_value.to(device=self.train_device, dtype=self.train_dtype),
                initial_previous_actions=(
                    None if ep.initial_previous_actions is None
                    else ep.initial_previous_actions.to(device=self.train_device, dtype=self.train_dtype)
                ),
                is_true_episode_start=ep.is_true_episode_start,
                returns=ep.returns.to(device=self.train_device, dtype=self.train_dtype),
                advantages=ep.advantages.to(device=self.train_device, dtype=self.train_dtype),
                initial_temporal_state=move_temporal_state(
                    ep.initial_temporal_state,
                    device=self.train_device,
                    dtype=self.train_dtype,
                ),
                rollout_env_idx=ep.rollout_env_idx,
                rollout_start_step=ep.rollout_start_step,
            )
            for ep in episodes
        ]


