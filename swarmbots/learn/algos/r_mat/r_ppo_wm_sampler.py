from dataclasses import dataclass
from typing import Any

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler, BaseWMSamples
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import (
    build_wm_episode_windows,
    ensure_wm_window_helper_compile_available,
    pad_time_axis,
)
from swarmbots.learn.base_sampler import BaseSampler, BatchIndices
from swarmbots.learn.temporal_state import concatenate_temporal_states, index_temporal_state


@dataclass
class RPPOWMSamples(BaseWMSamples):
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: MaybeTensor
    actions: torch.Tensor
    wm_actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    time_mask: torch.Tensor
    episode_start_mask: torch.Tensor
    initial_temporal_state: Any

    next_local_obs: torch.Tensor
    wm_target_time_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: MaybeTensor
    wm_loss_agent_mask: MaybeTensor


@dataclass(frozen=True, kw_only=True)
class RPPOWMSamplerConfig(PPOWMSamplerConfig):
    sequence_length: int


class RPPOWMSampler(
    BaseSampler[RPPOWMSamples, RPPOWMSamplerConfig],
    BaseWMSampler[RPPOWMSamples, RPPOWMSamplerConfig],
):

    def __init__(
            self,
            episodes: list[PPOEpisodeSegment],
            config: RPPOWMSamplerConfig,
            requires_previous_actions: bool = False,
    ) -> None:
        if config.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {config.sequence_length}")
        if config.num_next_steps < 1:
            raise ValueError(f"num_next_steps must be >= 1, got {config.num_next_steps}")
        if config.compile_wm_window_helper:
            ensure_wm_window_helper_compile_available(
                compile_mode=config.wm_window_helper_compile_mode,
            )

        non_empty_episodes = [episode for episode in episodes if episode.local_obs.shape[0] > 0]
        if not non_empty_episodes:
            raise ValueError("RPPOWMSampler requires at least one non-empty episode segment")
        if any(episode.initial_temporal_state is None for episode in non_empty_episodes):
            raise ValueError("Recurrent episode segments must contain initial_temporal_state")

        has_agent_mask = any(episode.agent_mask is not None for episode in non_empty_episodes)
        if has_agent_mask and any(episode.agent_mask is None for episode in non_empty_episodes):
            raise ValueError("agent_mask must be provided for all episodes or none")

        grouped_episodes = _group_episode_segments(non_empty_episodes)
        sequence_rows = [
            _build_sequence_row(
                episode_group,
                config=config,
                requires_previous_actions=requires_previous_actions,
                has_agent_mask=has_agent_mask,
            )
            for episode_group in grouped_episodes
        ]
        if config.batch_size > len(sequence_rows):
            raise ValueError(
                f"Recurrent batch_size counts TBPTT rows and must not exceed the available row count: "
                f"got batch_size={config.batch_size} for {len(sequence_rows)} rows"
            )

        self.local_obs = torch.stack([row.local_obs for row in sequence_rows]).contiguous()
        self.global_obs = torch.stack([row.global_obs for row in sequence_rows]).contiguous()
        self.hidden_local_vars = torch.stack([row.hidden_local_vars for row in sequence_rows]).contiguous()
        self.hidden_global_vars = torch.stack([row.hidden_global_vars for row in sequence_rows]).contiguous()
        self.agent_mask = (
            torch.stack([row.agent_mask for row in sequence_rows]).contiguous()
            if has_agent_mask
            else None
        )
        self.previous_actions = (
            torch.stack([row.previous_actions for row in sequence_rows]).contiguous()
            if requires_previous_actions
            else None
        )
        self.actions = torch.stack([row.actions for row in sequence_rows]).contiguous()
        self.wm_actions = torch.stack([row.wm_actions for row in sequence_rows]).contiguous()
        self.log_probs = torch.stack([row.log_probs for row in sequence_rows]).contiguous()
        self.values = torch.stack([row.values for row in sequence_rows]).contiguous()
        self.returns = torch.stack([row.returns for row in sequence_rows]).contiguous()
        self.advantages = torch.stack([row.advantages for row in sequence_rows]).contiguous()
        self.time_mask = torch.stack([row.time_mask for row in sequence_rows]).contiguous()
        self.episode_start_mask = torch.stack([row.episode_start_mask for row in sequence_rows]).contiguous()
        self.initial_temporal_state = concatenate_temporal_states(
            [row.initial_temporal_state for row in sequence_rows]
        )
        self.next_local_obs = torch.stack([row.next_local_obs for row in sequence_rows]).contiguous()
        self.wm_target_time_mask = torch.stack(
            [row.wm_target_time_mask for row in sequence_rows]
        ).contiguous()
        self.next_global_obs = torch.stack([row.next_global_obs for row in sequence_rows]).contiguous()
        self.wm_agent_mask = (
            torch.stack([row.wm_agent_mask for row in sequence_rows]).contiguous()
            if has_agent_mask
            else None
        )
        self.wm_loss_agent_mask = (
            torch.stack([row.wm_loss_agent_mask for row in sequence_rows]).contiguous()
            if has_agent_mask
            else None
        )

        super().__init__(
            config=config,
            n_samples=len(sequence_rows),
            index_device=self.local_obs.device,
        )

    def _fetch_samples(self, batch_indices: BatchIndices) -> RPPOWMSamples:
        return RPPOWMSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=None if self.previous_actions is None else self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            wm_actions=self.wm_actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            time_mask=self.time_mask[batch_indices],
            episode_start_mask=self.episode_start_mask[batch_indices],
            initial_temporal_state=index_temporal_state(self.initial_temporal_state, batch_indices),
            next_local_obs=self.next_local_obs[batch_indices],
            wm_target_time_mask=self.wm_target_time_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            wm_agent_mask=None if self.wm_agent_mask is None else self.wm_agent_mask[batch_indices],
            wm_loss_agent_mask=None if self.wm_loss_agent_mask is None else self.wm_loss_agent_mask[batch_indices],
        )


@dataclass
class _SequenceRow:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    previous_actions: torch.Tensor | None
    actions: torch.Tensor
    wm_actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    time_mask: torch.Tensor
    episode_start_mask: torch.Tensor
    initial_temporal_state: Any
    next_local_obs: torch.Tensor
    wm_target_time_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: torch.Tensor | None
    wm_loss_agent_mask: torch.Tensor | None


def _group_episode_segments(
        episodes: list[PPOEpisodeSegment],
) -> list[list[PPOEpisodeSegment]]:
    groups: dict[tuple[str, int], list[PPOEpisodeSegment]] = {}
    for episode_idx, episode in enumerate(episodes):
        key = (
            ("env", episode.rollout_env_idx)
            if episode.rollout_env_idx is not None
            else ("episode", episode_idx)
        )
        groups.setdefault(key, []).append(episode)

    grouped_episodes = []
    for key in sorted(groups):
        group = groups[key]
        group.sort(key=lambda episode: episode.rollout_start_step)
        grouped_episodes.append(group)
    return grouped_episodes


def _build_sequence_row(
        episodes: list[PPOEpisodeSegment],
        *,
        config: RPPOWMSamplerConfig,
        requires_previous_actions: bool,
        has_agent_mask: bool,
) -> _SequenceRow:
    sequence_length = config.sequence_length
    for previous_episode, episode in zip(episodes, episodes[1:]):
        expected_start = previous_episode.rollout_start_step + int(previous_episode.local_obs.shape[0])
        if episode.rollout_start_step != expected_start:
            raise ValueError(
                f"Non-contiguous rollout segments for env {episode.rollout_env_idx}: "
                f"expected start {expected_start}, got {episode.rollout_start_step}"
            )
    num_steps = sum(int(episode.local_obs.shape[0]) for episode in episodes)
    if num_steps > sequence_length:
        env_idx = episodes[0].rollout_env_idx
        raise ValueError(
            f"TBPTT row for env {env_idx} has {num_steps} steps, exceeding sequence_length={sequence_length}. "
            "Use a fixed step rollout no longer than the TBPTT sequence."
        )

    episode_windows = [
        build_wm_episode_windows(
            episode,
            num_next_steps=config.num_next_steps,
            compile_modules=config.compile_wm_window_helper,
            compile_mode=config.wm_window_helper_compile_mode,
        )
        for episode in episodes
    ]
    episode_start_mask = torch.zeros(
        sequence_length,
        dtype=torch.bool,
        device=episodes[0].local_obs.device,
    )
    offset = 0
    for episode in episodes:
        if episode.is_true_episode_start:
            episode_start_mask[offset] = True
        offset += int(episode.local_obs.shape[0])

    def concatenate_and_pad(
            tensors: list[torch.Tensor],
            *,
            pad_value: bool | int | float,
    ) -> torch.Tensor:
        tensor = torch.cat(tensors, dim=0)
        return pad_time_axis(
            tensor,
            padding_len=sequence_length - tensor.shape[0],
            pad_value=pad_value,
        )

    previous_actions = None
    if requires_previous_actions:
        previous_actions = concatenate_and_pad(
            [_build_previous_actions(episode) for episode in episodes],
            pad_value=0,
        )

    agent_mask = None
    wm_agent_mask = None
    wm_loss_agent_mask = None
    if has_agent_mask:
        if any(episode.agent_mask is None for episode in episodes):
            raise ValueError("agent_mask must be provided when agent masks are enabled")
        if any(windows.wm_agent_mask is None for windows in episode_windows):
            raise ValueError("wm_agent_mask must be provided when agent masks are enabled")
        if any(windows.wm_loss_agent_mask is None for windows in episode_windows):
            raise ValueError("wm_loss_agent_mask must be provided when agent masks are enabled")
        agent_mask = concatenate_and_pad(
            [episode.agent_mask for episode in episodes],
            pad_value=True,
        )
        wm_agent_mask = concatenate_and_pad(
            [windows.wm_agent_mask for windows in episode_windows],
            pad_value=True,
        )
        wm_loss_agent_mask = concatenate_and_pad(
            [windows.wm_loss_agent_mask for windows in episode_windows],
            pad_value=True,
        )

    time_mask = torch.zeros(sequence_length, dtype=torch.bool, device=episodes[0].local_obs.device)
    time_mask[:num_steps] = True
    return _SequenceRow(
        local_obs=concatenate_and_pad([episode.local_obs for episode in episodes], pad_value=0),
        global_obs=concatenate_and_pad([episode.global_obs for episode in episodes], pad_value=0),
        hidden_local_vars=concatenate_and_pad(
            [episode.hidden_local_vars for episode in episodes],
            pad_value=0,
        ),
        hidden_global_vars=concatenate_and_pad(
            [episode.hidden_global_vars for episode in episodes],
            pad_value=0,
        ),
        agent_mask=agent_mask,
        previous_actions=previous_actions,
        actions=concatenate_and_pad([episode.actions for episode in episodes], pad_value=0),
        wm_actions=concatenate_and_pad(
            [windows.multi_step_actions for windows in episode_windows],
            pad_value=0,
        ),
        log_probs=concatenate_and_pad([episode.log_probs for episode in episodes], pad_value=0),
        values=concatenate_and_pad([episode.values for episode in episodes], pad_value=0),
        returns=concatenate_and_pad([episode.returns for episode in episodes], pad_value=0),
        advantages=concatenate_and_pad([episode.advantages for episode in episodes], pad_value=0),
        time_mask=time_mask,
        episode_start_mask=episode_start_mask,
        initial_temporal_state=episodes[0].initial_temporal_state,
        next_local_obs=concatenate_and_pad(
            [windows.next_local_obs for windows in episode_windows],
            pad_value=0,
        ),
        wm_target_time_mask=concatenate_and_pad(
            [windows.wm_target_time_mask for windows in episode_windows],
            pad_value=False,
        ),
        next_global_obs=concatenate_and_pad(
            [windows.next_global_obs for windows in episode_windows],
            pad_value=0,
        ),
        wm_agent_mask=wm_agent_mask,
        wm_loss_agent_mask=wm_loss_agent_mask,
    )


def _build_previous_actions(episode: PPOEpisodeSegment) -> torch.Tensor:
    if episode.initial_previous_actions is None:
        return torch.zeros_like(episode.actions)
    return torch.cat((episode.initial_previous_actions.unsqueeze(0), episode.actions[:-1]), dim=0)
