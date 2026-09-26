from dataclasses import dataclass
from typing import TypeVar

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch
from swarmbots.learn.base_sampler import BaseSampler, BaseSamplerConfig, BatchIndices


@dataclass
class PPOSamples:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: MaybeTensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


@dataclass(frozen=True)
class PPOSamplerConfig(BaseSamplerConfig):
    pass


PPOSamplesType = TypeVar('PPOSamplesType', bound=PPOSamples, covariant=True)
PPOSamplerConfigType = TypeVar('PPOSamplerConfigType', bound=PPOSamplerConfig, covariant=True)


class PPOSampler(BaseSampler[PPOSamplesType, PPOSamplerConfigType]):

    def __init__(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOSamplerConfigType,
            requires_previous_actions: bool = False,
    ):
        self.local_obs = torch.concatenate(tuple(ep.local_obs for ep in episodes), dim=0)
        self.global_obs = torch.concatenate(tuple(ep.global_obs for ep in episodes), dim=0)
        self.hidden_local_vars = torch.concatenate(tuple(ep.hidden_local_vars for ep in episodes), dim=0)
        self.hidden_global_vars = torch.concatenate(tuple(ep.hidden_global_vars for ep in episodes), dim=0)
        has_agent_mask = any(ep.agent_mask is not None for ep in episodes)
        has_missing_agent_mask = any(ep.agent_mask is None for ep in episodes)
        if has_agent_mask and has_missing_agent_mask:
            raise ValueError("agent_mask must be provided for all episodes or none")
        if has_missing_agent_mask:
            self.agent_mask = None
        else:
            self.agent_mask = torch.concatenate(tuple(ep.agent_mask for ep in episodes), dim=0)
        if requires_previous_actions:
            previous_actions_per_episode = tuple(
                torch.cat((ep.initial_previous_actions.unsqueeze(0), ep.actions[:-1]), dim=0)
                for ep in episodes
            )
            self.previous_actions = torch.concatenate(previous_actions_per_episode, dim=0)
        else:
            self.previous_actions = None
        self.actions = torch.concatenate(tuple(ep.actions for ep in episodes), dim=0)
        self.log_probs = torch.concatenate(tuple(ep.log_probs for ep in episodes), dim=0)
        self.values = torch.concatenate(tuple(ep.values for ep in episodes), dim=0)
        self.returns = torch.concatenate(tuple(ep.returns for ep in episodes), dim=0)
        self.advantages = torch.concatenate(tuple(ep.advantages for ep in episodes), dim=0)

        super().__init__(
            config=config,
            n_samples=self.local_obs.shape[0],
            index_device=self.local_obs.device,
        )

    def _fetch_samples(self, batch_indices: BatchIndices) -> PPOSamples:
        return PPOSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=None if self.previous_actions is None else self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
        )


class PPOBatchSampler(BaseSampler[PPOSamplesType, PPOSamplerConfigType]):

    def __init__(
            self,
            rollout_batch: PPORolloutBatch,
            config: PPOSamplerConfigType,
            requires_previous_actions: bool = False,
    ):
        self.local_obs = _flatten_env_time(rollout_batch.local_obs)
        self.global_obs = _flatten_env_time(rollout_batch.global_obs)
        self.hidden_local_vars = _flatten_env_time(rollout_batch.hidden_local_vars)
        self.hidden_global_vars = _flatten_env_time(rollout_batch.hidden_global_vars)
        self.agent_mask = (
            None if rollout_batch.agent_mask is None else _flatten_env_time(rollout_batch.agent_mask)
        )
        self.previous_actions = (
            None
            if not requires_previous_actions or rollout_batch.previous_actions is None
            else _flatten_env_time(rollout_batch.previous_actions)
        )
        self.actions = _flatten_env_time(rollout_batch.actions)
        self.log_probs = _flatten_env_time(rollout_batch.log_probs)
        self.values = _flatten_env_time(rollout_batch.values)
        self.returns = _flatten_env_time(rollout_batch.returns)
        self.advantages = _flatten_env_time(rollout_batch.advantages)

        super().__init__(
            config=config,
            n_samples=self.local_obs.shape[0],
            index_device=self.local_obs.device,
        )

    def _fetch_samples(self, batch_indices: BatchIndices) -> PPOSamples:
        return PPOSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=None if self.previous_actions is None else self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
        )


def _flatten_env_time(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.flatten(0, 1).contiguous()
