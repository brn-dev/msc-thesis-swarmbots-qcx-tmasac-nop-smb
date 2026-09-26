from typing import Optional

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderState
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSampler, RPPOWMSamplerConfig
from swarmbots.learn.losses import LossDict


class RMATPolicyMixin:
    encoder: RMATEncoder
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int

    def _generate_rmat_actions(
            self,
            *,
            augmented_observations: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations, _ = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
        )
        actions, log_probs = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        assert log_probs is not None
        return actions, log_probs, values

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        augmented_observations, _ = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
        )
        return self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = hidden_local_vars
        _ = hidden_global_vars
        augmented_observations, _ = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
        )
        actions, _ = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: RPPOWMSamplerConfig,
    ) -> RPPOWMSampler:
        return RPPOWMSampler(
            episodes=episodes,
            config=config,
            requires_previous_actions=self.requires_previous_actions(),
        )

    def supports_rollout_batch_sampler(self, config: object) -> bool:
        _ = config
        return False

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATEncoderState:
        return self.encoder.initial_state(
            batch_size=batch_size,
            n_agents=n_agents,
            device=device,
            dtype=dtype,
        )

    def forward_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATEncoderState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATEncoderState]:
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
        actions, log_probs = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        assert log_probs is not None
        return actions, log_probs, values, next_state

    def predict_values_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            *,
            temporal_state: RMATEncoderState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        _ = previous_actions
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        return values, next_state

    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATEncoderState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = scenario_ids
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
        actions, _ = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions, next_state

    def _encode_observations(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            temporal_state: RMATEncoderState | None,
            episode_start_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RMATEncoderState]:
        if temporal_state is None:
            temporal_state = self.initial_temporal_state(
                batch_size=local_obs.shape[0],
                n_agents=local_obs.shape[-2],
                device=local_obs.device,
                dtype=local_obs.dtype,
            )
        return self.encoder(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
            initial_state=temporal_state,
            reset_mask=episode_start_mask,
        )


def flatten_time_agent_mask(
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor,
        n_agents: int,
) -> torch.Tensor:
    if agent_mask is None:
        return time_mask.unsqueeze(-1).expand(*time_mask.shape, n_agents).reshape(-1, n_agents)
    return (agent_mask & time_mask.unsqueeze(-1)).reshape(-1, n_agents)


def reshape_extra_losses(
        *,
        extra_losses: LossDict,
        batch_size: int,
        sequence_length: int,
        n_agents: int,
        time_mask: torch.Tensor,
        loss_agent_mask: torch.Tensor,
) -> LossDict:
    reshaped_losses: LossDict = {}
    flat_time_mask = time_mask.reshape(batch_size * sequence_length)
    for name, value in extra_losses.items():
        if value.ndim == 2 and value.shape == (batch_size * sequence_length, n_agents):
            reshaped_value = value.masked_fill(~loss_agent_mask, 0.0)
            reshaped_losses[name] = reshaped_value.reshape(batch_size, sequence_length, n_agents)
            continue
        if value.ndim == 1 and value.shape == (batch_size * sequence_length,):
            reshaped_value = value.masked_fill(~flat_time_mask, 0.0)
            reshaped_losses[name] = reshaped_value.reshape(batch_size, sequence_length)
            continue
        reshaped_losses[name] = value
    return reshaped_losses
