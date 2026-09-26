from collections.abc import Callable
from dataclasses import dataclass, field
import shutil
import sys
from typing import Any, Literal, Optional

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    bernoulli_config_to_dict,
    continuous_config_to_dicts,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoder, MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_encoder import _init_linear
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.popart import PopArtLinear
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass


def _masked_agent_pool(
        values: torch.Tensor,
        agent_mask: torch.Tensor | None,
        *,
        mode: Literal["mean", "sum"],
) -> torch.Tensor:
    if agent_mask is None:
        if mode == "mean":
            return values.mean(dim=1)
        if mode == "sum":
            return values.sum(dim=1)
        raise ValueError(f"Unknown pool mode: {mode}")

    if agent_mask.dtype != torch.bool:
        raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
    if agent_mask.shape != values.shape:
        raise ValueError(f"Expected agent_mask shape {tuple(values.shape)}, got {tuple(agent_mask.shape)}")

    masked_values = values * agent_mask.to(dtype=values.dtype)
    if mode == "sum":
        return masked_values.sum(dim=1)
    if mode == "mean":
        denom = agent_mask.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
        return masked_values.sum(dim=1) / denom
    raise ValueError(f"Unknown pool mode: {mode}")


class MATOrigCritic(nn.Module):

    def __init__(
            self,
            *,
            encoder_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            use_popart: bool,
            popart_config: PopArtConfig,
            pool_mode: Literal["mean", "sum"],
            value_head_hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_local_vars_dim = hidden_local_vars_dim
        self.hidden_global_vars_dim = hidden_global_vars_dim
        self.pool_mode = pool_mode

        per_agent_input_dim = encoder_dim + hidden_local_vars_dim + hidden_global_vars_dim
        value_head_layers: list[nn.Module] = [
            _init_linear(nn.Linear(per_agent_input_dim, encoder_dim), activate=True),
            nn.GELU(),
            nn.LayerNorm(encoder_dim),
        ]
        head_input_dim = encoder_dim
        for hidden_dim in [] if value_head_hidden_dims is None else value_head_hidden_dims:
            value_head_layers.extend([
                _init_linear(nn.Linear(head_input_dim, hidden_dim), activate=True),
                nn.GELU(),
            ])
            head_input_dim = hidden_dim
        value_head_layers.append(_init_linear(nn.Linear(head_input_dim, 1)))
        self.value_head = nn.Sequential(*value_head_layers)

        self.popart_head: PopArtLinear | None
        if use_popart:
            self.popart_head = PopArtLinear(
                in_features=1,
                out_features=1,
                beta=popart_config.beta,
                eps=popart_config.eps,
                min_std=popart_config.min_std,
                init_sigma=popart_config.init_sigma,
            )
        else:
            self.popart_head = None

    def forward(
            self,
            encoder_rep: torch.Tensor,
            *,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        value_input = encoder_rep
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            value_input = torch.cat((value_input, hidden_local_vars), dim=-1)
        if self.hidden_global_vars_dim > 0:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
            expanded_hidden_global = hidden_global_vars.unsqueeze(1).expand(-1, encoder_rep.shape[1], -1)
            value_input = torch.cat((value_input, expanded_hidden_global), dim=-1)

        per_agent_values = self.value_head(value_input).squeeze(-1)
        pooled_values = _masked_agent_pool(per_agent_values, agent_mask, mode=self.pool_mode)
        if self.popart_head is None:
            return pooled_values
        return self.popart_head(pooled_values.unsqueeze(-1)).squeeze(-1)

    @property
    def has_popart(self) -> bool:
        return self.popart_head is not None

    def update_popart(self, targets: torch.Tensor) -> None:
        if self.popart_head is None:
            raise RuntimeError("PopArt is not enabled for this MATOrigCritic")
        self.popart_head.update(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.popart_head is None:
            return values
        return self.popart_head.normalize(values)

    def get_popart_metrics(self) -> dict[str, float]:
        if self.popart_head is None:
            return {}
        sigma = self.popart_head.sigma()
        return {
            "popart_mu": self.popart_head.mu.mean().item(),
            "popart_sigma": sigma.mean().item(),
        }


@dataclass(frozen=True)
class MATOrigCriticConfig:
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    pool_mode: Literal["mean", "sum"] = "mean"
    value_head_hidden_dims: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MATOrigPolicyConfig:
    encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    decoder_config: MATOrigDecoderConfig = field(default_factory=MATOrigDecoderConfig)
    critic_config: MATOrigCriticConfig = field(default_factory=MATOrigCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    max_agents: int | None = None
    compile_modules: bool = False
    compile_mode: str = "default"
    encoder_decoder_projection_init_gain: float = 1.0
    action_net_init_gain: float = 0.01


def _ensure_torch_compile_available(*, compile_mode: str) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("MATOrigPolicyConfig.compile_modules=True requires torch.compile support.")
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "MATOrigPolicyConfig.compile_modules=True on this Windows setup requires cl.exe on PATH for torch.compile."
        )
    if not compile_mode:
        raise ValueError("MATOrigPolicyConfig.compile_mode must be a non-empty string when compile_modules=True.")


class MATOrigPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATOrigPolicyConfig = MATOrigPolicyConfig(),
    ) -> None:
        super().__init__()
        self.config = config

        self.n_agents = env.n_agents
        self.max_agents = self.n_agents if config.max_agents is None else config.max_agents
        if self.max_agents < self.n_agents:
            raise ValueError(f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}")

        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.agent_action_dim = env.action_space.total_agent_action_dim

        self.encoder = MATEncoder(
            config=config.encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )
        self.decoder = MATOrigDecoder(
            config=config.decoder_config,
            max_agents=self.max_agents,
            action_input_dim=self.agent_action_dim,
        )
        if config.encoder_config.d_model == self.decoder.d_model:
            self.encoder_decoder_projection: nn.Module = nn.Identity()
        else:
            encoder_decoder_projection = nn.Linear(config.encoder_config.d_model, self.decoder.d_model)
            make_init_linear_orthogonal(config.encoder_decoder_projection_init_gain)(encoder_decoder_projection)
            self.encoder_decoder_projection = encoder_decoder_projection
        self.action_dist = HybridActionDistribution(
            latent_dim=self.decoder.latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
            action_net_initialization=make_init_linear_orthogonal(config.action_net_init_gain),
        )
        self.critic = MATOrigCritic(
            encoder_dim=config.encoder_config.d_model,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            use_popart=config.critic_config.use_popart,
            popart_config=config.critic_config.popart_config,
            pool_mode=config.critic_config.pool_mode,
            value_head_hidden_dims=config.critic_config.value_head_hidden_dims,
        )
        self._generate_actions_fn: Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]] = self._generate_actions_impl
        self._evaluate_latent_and_values_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_latent_and_values_impl
        )
        self._evaluate_actions_core_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_actions_core_impl
        )
        self._apply_optional_compile()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mat_orig_policy_config": {
                "encoder_config": serialize_dataclass(self.config.encoder_config),
                "decoder_config": serialize_dataclass(self.config.decoder_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "max_agents": self.max_agents,
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
                "encoder_decoder_projection_init_gain": self.config.encoder_decoder_projection_init_gain,
                "action_net_init_gain": self.config.action_net_init_gain,
                "action_dist_compile_friendly": self.action_dist.compile_friendly,
            }
        }

    def requires_previous_actions(self) -> bool:
        return self.action_dist.requires_previous_actions()

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
        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0], n_agents=local_obs.shape[1])
        encoder_rep = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        decoder_obs_rep = self._project_encoder_rep_for_decoder(encoder_rep)
        actions, log_probs = self._generate_actions(
            decoder_obs_rep=decoder_obs_rep,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self.critic(
            encoder_rep,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return actions, log_probs, values

    def _evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        actions = self._policy_actions(batch.actions)
        self._validate_agent_mask(batch.agent_mask, batch_size=batch.local_obs.shape[0], n_agents=batch.local_obs.shape[1])
        if self.action_dist.compile_friendly:
            log_probs, values, encoder_rep = self._evaluate_actions_core(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                previous_actions=batch.previous_actions,
                actions=actions,
            )
        else:
            latent_pi, values, encoder_rep = self._evaluate_latent_and_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                actions=actions,
            )
            self.action_dist.update_latent_features(latent_pi)
            log_probs = self.action_dist.log_prob(actions, previous_actions=batch.previous_actions)
        if batch.agent_mask is not None:
            log_probs = log_probs.masked_fill(~batch.agent_mask, 0.0)
        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(
            agent_mask=batch.agent_mask,
            action_splitter=action_splitter,
        )
        return log_probs, values, extra_losses, extra_loss_metrics, encoder_rep

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
        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0], n_agents=local_obs.shape[1])
        encoder_rep = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        return self.critic(
            encoder_rep,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
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
        self._validate_agent_mask(agent_mask, batch_size=local_obs.shape[0], n_agents=local_obs.shape[1])
        encoder_rep = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        decoder_obs_rep = self._project_encoder_rep_for_decoder(encoder_rep)
        actions, _ = self._generate_actions(
            decoder_obs_rep=decoder_obs_rep,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOSamplerConfig,
    ) -> PPOSampler:
        return PPOSampler(
            episodes=episodes,
            config=config,
            requires_previous_actions=self.requires_previous_actions(),
        )

    def _actor_latents_from_actions(
            self,
            encoder_rep: torch.Tensor,
            actions: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        shifted_actions = self._build_shifted_actions(actions, agent_mask=agent_mask)
        decoder_obs_rep = self._project_encoder_rep_for_decoder(encoder_rep)
        return self.decoder(shifted_actions, decoder_obs_rep, agent_mask=agent_mask)

    def _project_encoder_rep_for_decoder(
            self,
            encoder_rep: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder_decoder_projection(encoder_rep)

    def _evaluate_latent_and_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_latent_and_values_fn(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )

    def _evaluate_latent_and_values_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_rep = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        latent_pi = self._actor_latents_from_actions(encoder_rep, actions, agent_mask=agent_mask)
        values = self.critic(
            encoder_rep,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return latent_pi, values, encoder_rep

    def _evaluate_actions_core(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_actions_core_fn(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            actions=actions,
        )

    def _evaluate_actions_core_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_pi, values, encoder_rep = self._evaluate_latent_and_values_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions, previous_actions=previous_actions)
        return log_probs, values, encoder_rep

    @staticmethod
    def _build_shifted_actions(
            actions: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        MATOrigPolicy._validate_agent_mask(agent_mask, batch_size=actions.shape[0], n_agents=actions.shape[1])
        shifted_actions = torch.zeros_like(actions)
        if actions.shape[1] <= 1:
            return shifted_actions

        source_actions = actions
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.shape != actions.shape[:2]:
                raise ValueError(f"Expected agent_mask shape {tuple(actions.shape[:2])}, got {tuple(agent_mask.shape)}")
            source_actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        shifted_actions[:, 1:, :] = source_actions[:, :-1, :]
        return shifted_actions

    @staticmethod
    def _validate_agent_mask(
            agent_mask: torch.Tensor | None,
            *,
            batch_size: int,
            n_agents: int,
    ) -> None:
        if agent_mask is None:
            return
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.shape != (batch_size, n_agents):
            raise ValueError(f"Expected agent_mask shape {(batch_size, n_agents)}, got {tuple(agent_mask.shape)}")

        mask_int = agent_mask.to(dtype=torch.int8)
        if torch.any(mask_int[:, 1:] > mask_int[:, :-1]):
            raise ValueError("MATOrigPolicy expects agent_mask to be a contiguous true-prefix per batch item")

    def _generate_actions(
            self,
            *,
            decoder_obs_rep: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._generate_actions_fn(
            decoder_obs_rep=decoder_obs_rep,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=return_log_probs,
        )

    def _generate_actions_impl(
            self,
            *,
            decoder_obs_rep: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = decoder_obs_rep.shape[0]
        shifted_actions = decoder_obs_rep.new_zeros((batch_size, self.n_agents, self.agent_action_dim))
        output_actions = decoder_obs_rep.new_zeros((batch_size, self.n_agents, self.agent_action_dim))
        output_log_probs = decoder_obs_rep.new_zeros((batch_size, self.n_agents)) if return_log_probs else None

        for agent_idx in range(self.n_agents):
            latent_pi = self.decoder(shifted_actions, decoder_obs_rep, agent_mask=agent_mask)[
                :, agent_idx:agent_idx + 1, :
            ]
            sample_agent_idx = agent_idx if self.action_dist.sampling_depends_on_agent else None
            previous_action_i = None if previous_actions is None else previous_actions[:, agent_idx:agent_idx + 1, :]

            if return_log_probs:
                action_i, log_prob_i = self.action_dist.get_on_policy_actions_with_log_probs(
                    latent_pi,
                    deterministic=deterministic,
                    agent=sample_agent_idx,
                    previous_actions=previous_action_i,
                )
                output_log_probs[:, agent_idx] = log_prob_i.squeeze(1)
            else:
                action_i = self.action_dist.update_latent_features(latent_pi).get_actions(
                    deterministic=deterministic,
                    agent=sample_agent_idx,
                    previous_actions=previous_action_i,
                )

            output_actions[:, agent_idx:agent_idx + 1, :] = action_i
            if agent_idx + 1 < self.n_agents:
                shifted_actions[:, agent_idx + 1:agent_idx + 2, :] = action_i

        if agent_mask is not None:
            output_actions = output_actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
            if output_log_probs is not None:
                output_log_probs = output_log_probs.masked_fill(~agent_mask, 0.0)
        return output_actions, output_log_probs

    @property
    def has_popart(self) -> bool:
        return self.critic.has_popart

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        if not self.has_popart:
            return
        self.critic.update_popart(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.critic.normalize_values(values)

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return self.critic.get_popart_metrics()

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "encoder": self._module_grad_norm(self.encoder),
            "encoder_decoder_projection": self._module_grad_norm(self.encoder_decoder_projection),
            "decoder": self._module_grad_norm(self.decoder),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)
        per_action_entropy_weights = self._pop_per_action_entropy_weights(remaining_weights)
        for idx, value in per_action_entropy_weights.items():
            if value < 0:
                raise ValueError(f"act{idx}_ent_loss_coef must be >= 0, got {value}")
            self.action_dist.set_sub_ent_loss_coef(idx, value)

        entropy_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("ent_loss_coef", "entropy", "ent"),
        )
        if entropy_weight is not None:
            alias, value = entropy_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.action_dist.set_all_ent_loss_coefs(value)

        super().update_loss_weights(**remaining_weights)

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return

        _ensure_torch_compile_available(compile_mode=self.config.compile_mode)

        self.encoder = self._compile_module(self.encoder)
        self.encoder_decoder_projection = self._compile_module(self.encoder_decoder_projection)
        self.decoder = self._compile_module(self.decoder)
        self.critic = self._compile_module(self.critic)
        self._evaluate_latent_and_values_fn = self._compile_callable(self._evaluate_latent_and_values_impl)
        if self.action_dist.compile_friendly:
            self._generate_actions_fn = self._compile_callable(self._generate_actions_impl)
            self._evaluate_actions_core_fn = self._compile_callable(self._evaluate_actions_core_impl)

    def _compile_module(
            self,
            module: nn.Module,
    ) -> nn.Module:
        if isinstance(module, nn.Identity):
            return module
        return torch.compile(
            module,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    def _compile_callable(
            self,
            fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        return torch.compile(
            fn,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )
