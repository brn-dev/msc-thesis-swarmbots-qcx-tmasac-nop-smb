from dataclasses import dataclass, field, replace

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_policy_mixin import (
    RMATPolicyMixin,
    flatten_time_agent_mask,
    reshape_extra_losses,
)
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamples
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import (
    BaseLearnEnvWrapper,
)
from swarmbots.learn.losses import LossDict, LossMetrics


@dataclass(frozen=True)
class RMATIndPolicyConfig(MATIndPolicyConfig):
    encoder_config: RMATEncoderConfig = field(default_factory=RMATEncoderConfig)


class RMATIndPolicy(RMATPolicyMixin, MATIndPolicy):

    encoder_config: RMATEncoderConfig

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: RMATIndPolicyConfig | None = None,
    ) -> None:
        super().__init__(env=env, config=RMATIndPolicyConfig() if config is None else config)

    def _build_encoder_config(self) -> RMATEncoderConfig:
        return replace(
            self.config.encoder_config,
            d_model=self.d_model_encoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_encoder(self) -> nn.Module:
        return RMATEncoder(
            config=self.encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )

    def _generate_rmat_actions(
            self,
            *,
            augmented_observations: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._generate_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=return_log_probs,
        )

    def _evaluate_actions(
            self,
            batch: RPPOWMSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        local_obs = batch.local_obs
        batch_size, sequence_length, n_agents, _local_obs_dim = local_obs.shape
        augmented_observations, _ = self.encoder(
            local_obs,
            batch.global_obs,
            agent_mask=batch.agent_mask,
            time_mask=batch.time_mask,
            initial_state=batch.initial_temporal_state,
            reset_mask=batch.episode_start_mask,
        )

        flat_batch_size = batch_size * sequence_length
        flat_augmented_observations = augmented_observations.reshape(
            flat_batch_size,
            n_agents,
            self.d_model_encoder,
        )
        flat_agent_mask = None if batch.agent_mask is None else batch.agent_mask.reshape(flat_batch_size, n_agents)
        flat_loss_agent_mask = flatten_time_agent_mask(
            agent_mask=batch.agent_mask,
            time_mask=batch.time_mask,
            n_agents=n_agents,
        )
        flat_previous_actions = (
            None
            if batch.previous_actions is None
            else batch.previous_actions.reshape(flat_batch_size, n_agents, self.agent_action_dim)
        )
        flat_actions = batch.actions.reshape(flat_batch_size, n_agents, self.agent_action_dim)

        latent_pi = self.actor_head(flat_augmented_observations).contiguous()
        self.action_dist.update_latent_features(latent_pi)
        flat_log_probs = self.action_dist.log_prob(flat_actions, previous_actions=flat_previous_actions)
        flat_log_probs = flat_log_probs.masked_fill(~flat_loss_agent_mask, 0.0)

        flat_values = self._critic_with_hidden_vars(
            flat_augmented_observations,
            batch.hidden_local_vars.reshape(flat_batch_size, n_agents, self.hidden_local_vars_dim),
            batch.hidden_global_vars.reshape(flat_batch_size, self.hidden_global_vars_dim),
            agent_mask=flat_agent_mask,
        )
        flat_time_mask = batch.time_mask.reshape(flat_batch_size)
        flat_values = flat_values.masked_fill(~flat_time_mask, 0.0)

        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(
            agent_mask=flat_loss_agent_mask,
            action_splitter=action_splitter,
        )
        reshaped_extra_losses = reshape_extra_losses(
            extra_losses=extra_losses,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            time_mask=batch.time_mask,
            loss_agent_mask=flat_loss_agent_mask,
        )

        return (
            flat_log_probs.reshape(batch_size, sequence_length, n_agents),
            flat_values.reshape(batch_size, sequence_length),
            reshaped_extra_losses,
            extra_loss_metrics,
            augmented_observations,
        )
