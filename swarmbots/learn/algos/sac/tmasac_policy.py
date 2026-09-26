import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, Self

import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.action_sampling import ActionSampleStrategy, expand_action_samples
from swarmbots.learn.action_dists.hybrid_action_dist import (
    ContinuousActionDistConfigInput,
    HybridActionDistribution,
    continuous_action_gradient_estimator,
    continuous_config_to_dicts,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig, MATEncoderLayer
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
from swarmbots.learn.algos.sac.sac_nop import (
    SACNOPConfig,
    SACNOPLatentSource,
    SACNOPModule,
    normalize_nop_latent_source,
)
from swarmbots.learn.algos.sac.scenario_obs_encoder import (
    TMASACScenarioEncoderConfig,
    TMASACScenarioObservationEncoder,
)
from swarmbots.learn.algos.sac.tmasac_actor_heads import (
    TMASACActorHead,
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
    TMASACIndependentActorHead,
    TMASACQCXActorHead,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.feed_forward import MLP, feedforward_linear_layers, make_feedforward
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal, reinitialize_multihead_attention
from swarmbots.learn.polyak_update import polyak_update
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_value


@dataclass(frozen=True)
class TMASACCriticConfig:
    n_local_projection_hidden_layers: int = 1
    n_value_regressor_hidden_layers: int = 2
    action_coembed_hidden_dims: list[int] | None = None
    separate_observation_action_encoders: bool = False
    action_encoder_dim: int | None = None
    independent_encoders: bool = False
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    action_coembed_init_gain: float = 1.0
    action_coembed_output_init_gain: float = 1.0
    local_projection_init_gain: float = 1.0
    value_regressor_init_gain: float = 1.0
    value_head_init_gain: float = 0.01


@dataclass(frozen=True)
class TMASACPolicyConfig:
    actor_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    critic_encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    shared_encoder_config: MATEncoderConfig | None = None
    share_observation_encoder: bool = False
    actor_head_config: TMASACActorHeadConfig = field(default_factory=TMASACActorHeadConfig)
    critic_config: TMASACCriticConfig = field(default_factory=TMASACCriticConfig)
    act_fn_cls: ActivationFactory = nn.GELU
    dropout: float = 0.0
    continuous_config: ContinuousActionDistConfigInput = field(
        default_factory=lambda: PredictedStdConfig(base_std=1.0)
    )
    max_agents: int | None = None
    nop_config: SACNOPConfig = field(default_factory=SACNOPConfig)
    compile_modules: bool = False
    compile_mode: str = "default"
    action_net_init_gain: float = 0.01
    scenario_encoder_config: TMASACScenarioEncoderConfig | None = None


class TMASACObservationActionEncoder(nn.Module):
    def __init__(
            self,
            *,
            observation_input_dim: int,
            action_dim: int,
            d_model: int,
            action_encoder_dim: int | None,
            linear_init_gain: float,
            act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.action_encoder_dim = (
            max(1, d_model // 2)
            if action_encoder_dim is None
            else int(action_encoder_dim)
        )
        linear_init = make_init_linear_orthogonal(linear_init_gain)
        self.observation_encoder = MLP(
            input_dim=observation_input_dim,
            hidden_dims=[d_model],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fn_cls,
        )
        self.action_encoder = MLP(
            input_dim=action_dim,
            hidden_dims=[self.action_encoder_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fn_cls,
        )
        self.output_dim = d_model + self.action_encoder_dim

    def forward(
            self,
            observation_inputs: torch.Tensor,
            actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self.observation_encoder(observation_inputs),
                self.action_encoder(actions),
            ),
            dim=-1,
        )


class TMASACActionConditionedEncoder(nn.Module):
    def __init__(
            self,
            *,
            max_agents: int,
            local_input_dim: int,
            global_input_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            action_dim: int,
            encoder_config: MATEncoderConfig,
            critic_config: TMASACCriticConfig,
            act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.max_agents = int(max_agents)
        self.local_input_dim = int(local_input_dim)
        self.global_input_dim = int(global_input_dim)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.global_encoder_input_dim = self.global_input_dim + self.hidden_global_vars_dim
        self.has_global_input = self.global_encoder_input_dim > 0

        local_observation_input_dim = self.local_input_dim + self.hidden_local_vars_dim
        self.observation_action_encoder = (
            TMASACObservationActionEncoder(
                observation_input_dim=local_observation_input_dim,
                action_dim=self.action_dim,
                d_model=self.d_model,
                action_encoder_dim=critic_config.action_encoder_dim,
                linear_init_gain=critic_config.action_coembed_init_gain,
                act_fn_cls=act_fn_cls,
            )
            if critic_config.separate_observation_action_encoders
            else None
        )
        local_action_input_dim = (
            local_observation_input_dim + self.action_dim
            if self.observation_action_encoder is None
            else self.observation_action_encoder.output_dim
        )
        self.local_action_input_norm = (
            nn.LayerNorm(local_action_input_dim)
            if encoder_config.normalize_obs_inputs
            else nn.Identity()
        )
        action_coembed_hidden_dims = (
            [self.d_model]
            if critic_config.action_coembed_hidden_dims is None
            else [*critic_config.action_coembed_hidden_dims]
        )
        self.local_action_encoder = MLP(
            input_dim=local_action_input_dim,
            hidden_dims=[*action_coembed_hidden_dims, self.d_model],
            end_with_act_fn=False,
            linear_init=make_init_linear_orthogonal(critic_config.action_coembed_init_gain),
            final_linear_init=make_init_linear_orthogonal(critic_config.action_coembed_output_init_gain),
            act_fn_cls=act_fn_cls,
        )

        self.global_input_norm = (
            nn.LayerNorm(self.global_encoder_input_dim)
            if encoder_config.normalize_obs_inputs and self.has_global_input
            else nn.Identity()
        )
        if self.has_global_input:
            linear_init = make_init_linear_orthogonal(encoder_config.linear_init_gain)
            projection_linear_init = (
                linear_init
                if encoder_config.linear_projection_init_gain is None
                else make_init_linear_orthogonal(encoder_config.linear_projection_init_gain)
            )
            self.global_encoder = make_feedforward(
                input_dim=self.global_encoder_input_dim,
                output_dim=self.d_model,
                config=encoder_config.global_obs_encoder_config,
                linear_init=linear_init,
                output_linear_init=projection_linear_init,
                act_fn_cls=act_fn_cls,
            )
        else:
            self.global_encoder = None

        self.token_norm = nn.LayerNorm(self.d_model) if encoder_config.normalize_tokens else nn.Identity()
        prototype_layer = MATEncoderLayer(encoder_config)
        self.layers = nn.ModuleList([
            copy.deepcopy(prototype_layer)
            for _ in range(encoder_config.num_layers)
        ])
        self.norm = nn.LayerNorm(
            self.d_model,
            eps=encoder_config.layer_norm_eps,
            bias=encoder_config.bias,
        )
        if encoder_config.transformer_ff_init_gain is not None:
            self._reinitialize_layers(feedforward_init_gain=encoder_config.transformer_ff_init_gain)

        self.agent_embeddings: nn.Parameter | None = None
        if encoder_config.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, self.max_agents, self.d_model),
                requires_grad=True,
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def forward(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        n_agents = local_inputs.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_inputs second dim <= {self.max_agents}, got {n_agents}")

        observation_parts = [local_inputs]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            observation_parts.append(hidden_local_vars)
        observation_inputs = (
            observation_parts[0]
            if len(observation_parts) == 1
            else torch.cat(observation_parts, dim=-1)
        )
        token_inputs = (
            torch.cat((observation_inputs, actions), dim=-1)
            if self.observation_action_encoder is None
            else self.observation_action_encoder(observation_inputs, actions)
        )
        token_inputs = self.local_action_input_norm(token_inputs)
        tokens = self.local_action_encoder(token_inputs)
        if self.agent_embeddings is not None:
            tokens = tokens + self.agent_embeddings[:, :n_agents, :]

        if self.global_encoder is not None:
            global_parts = [global_inputs] if self.global_input_dim > 0 else []
            if self.hidden_global_vars_dim > 0:
                if hidden_global_vars is None:
                    raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
                global_parts.append(hidden_global_vars)
            global_encoder_inputs = global_parts[0] if len(global_parts) == 1 else torch.cat(global_parts, dim=-1)
            global_tokens = self.global_encoder(self.global_input_norm(global_encoder_inputs))
            tokens = tokens + global_tokens.unsqueeze(1).expand(-1, n_agents, -1)
        tokens = self.token_norm(tokens)

        src_key_padding_mask = None if agent_mask is None else ~agent_mask
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.norm(tokens)

    def _reinitialize_layers(self, *, feedforward_init_gain: float) -> None:
        hidden_linear_init = make_init_linear_orthogonal(feedforward_init_gain)
        output_linear_init = make_init_linear_orthogonal(1.0)
        for layer in self.layers:
            if layer.self_attn is not None:
                reinitialize_multihead_attention(layer.self_attn)
            hidden_layers, output_layers = feedforward_linear_layers(layer.feedforward)
            for linear in hidden_layers:
                hidden_linear_init(linear)
            for linear in output_layers:
                output_linear_init(linear)


class TMASACTwinCritic(nn.Module):
    def __init__(
            self,
            *,
            n_agents: int,
            max_agents: int,
            local_input_dim: int,
            global_input_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            action_dim: int,
            encoder_config: MATEncoderConfig,
            critic_config: TMASACCriticConfig,
            act_fn_cls: ActivationFactory,
            dropout: float,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.encoder_config = replace(
            encoder_config,
            act_fn_cls=act_fn_cls,
            dropout=dropout,
        )
        def build_encoder() -> TMASACActionConditionedEncoder:
            return TMASACActionConditionedEncoder(
                max_agents=max_agents,
                local_input_dim=local_input_dim,
                global_input_dim=global_input_dim,
                hidden_local_vars_dim=self.hidden_local_vars_dim,
                hidden_global_vars_dim=self.hidden_global_vars_dim,
                action_dim=self.action_dim,
                encoder_config=self.encoder_config,
                critic_config=critic_config,
                act_fn_cls=act_fn_cls,
            )

        self.encoder = build_encoder()
        self.encoder2 = build_encoder() if critic_config.independent_encoders else None
        self.nop_source_latent_dim = self.d_model * (2 if self.encoder2 is not None else 1)
        q_local_dim = self.d_model
        self.q1 = self._build_q_network(
            q_local_dim=q_local_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )
        self.q2 = self._build_q_network(
            q_local_dim=q_local_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )

    def encode(
            self,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        primary_latents = self._encode_with(
            self.encoder,
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        if self.encoder2 is None:
            return primary_latents
        secondary_latents = self._encode_with(
            self.encoder2,
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return self._combine_nop_source_latents(primary_latents, secondary_latents)

    def forward(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_latents = self._encode_with(
            self.encoder,
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        q2_latents = q1_latents if self.encoder2 is None else self._encode_with(
            self.encoder2,
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return (
            self.q1(q1_latents, agent_mask=agent_mask),
            self.q2(q2_latents, agent_mask=agent_mask),
            self._combine_nop_source_latents(q1_latents, q2_latents),
        )

    def _combine_nop_source_latents(
            self,
            primary_latents: torch.Tensor,
            secondary_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.encoder2 is None:
            return primary_latents
        return torch.cat((primary_latents, secondary_latents), dim=-1)

    @staticmethod
    def _encode_with(
            encoder: TMASACActionConditionedEncoder,
            *,
            local_inputs: torch.Tensor,
            global_inputs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return encoder(
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    def _load_from_state_dict(
            self,
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
    ) -> None:
        if self.encoder2 is not None:
            primary_prefix = f"{prefix}encoder."
            secondary_prefix = f"{prefix}encoder2."
            has_primary_encoder = any(key.startswith(primary_prefix) for key in state_dict)
            has_secondary_encoder = any(key.startswith(secondary_prefix) for key in state_dict)
            if has_primary_encoder and not has_secondary_encoder:
                for key, value in list(state_dict.items()):
                    if key.startswith(primary_prefix):
                        state_dict[f"{secondary_prefix}{key.removeprefix(primary_prefix)}"] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _build_q_network(
            *,
            q_local_dim: int,
            critic_config: TMASACCriticConfig,
            act_fn_cls: ActivationFactory,
    ) -> DeepSetCritic:
        return DeepSetCritic(
            num_local_features=q_local_dim,
            local_projection_hidden_dims=[
                q_local_dim
            ] * critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[
                q_local_dim
            ] * critic_config.n_value_regressor_hidden_layers,
            act_fn_cls=act_fn_cls,
            local_projection_linear_init_gain=critic_config.local_projection_init_gain,
            value_regressor_linear_init_gain=critic_config.value_regressor_init_gain,
            value_head_linear_init_gain=critic_config.value_head_init_gain,
            use_popart=critic_config.use_popart,
            popart_beta=critic_config.popart_config.beta,
            popart_eps=critic_config.popart_config.eps,
            popart_min_std=critic_config.popart_config.min_std,
            popart_init_sigma=critic_config.popart_config.init_sigma,
        )


class TMASACPolicy(BaseSACPolicy):
    _compiled_action_log_prob: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None
    _compiled_actor_actions_and_log_probs: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None
    _compiled_actor_action_samples: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None
    _compiled_actor_encoder: nn.Module | None

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: TMASACPolicyConfig = TMASACPolicyConfig(),
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_compiled_action_log_prob", None)
        object.__setattr__(self, "_compiled_actor_actions_and_log_probs", None)
        object.__setattr__(self, "_compiled_actor_action_samples", None)
        object.__setattr__(self, "_compiled_actor_encoder", None)
        self.config = config
        self._validate_continuous_action_space(env)
        self._validate_config(config)

        self.n_agents = int(env.n_agents)
        self.max_agents = self.n_agents if config.max_agents is None else int(config.max_agents)
        if self.max_agents < self.n_agents:
            raise ValueError(f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}")
        self.local_obs_dim = int(env.local_obs_dim)
        self.global_obs_dim = int(env.global_obs_dim)
        self.hidden_local_vars_dim = int(env.hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(env.hidden_global_vars_dim)
        self.agent_action_dim = int(env.action_space.total_agent_action_dim)
        self.act_fn_cls = config.act_fn_cls
        self.dropout = float(config.dropout)
        self.actor_head_kind = self._normalize_actor_head_kind(config.actor_head_config.kind)
        self.nop_latent_source = normalize_nop_latent_source(config.nop_config.latent_source)
        self.share_observation_encoder = (
            bool(config.share_observation_encoder)
            or config.shared_encoder_config is not None
        )
        if (
                self.actor_head_kind is TMASACActorHeadKind.DECENTRALIZED
                and self.share_observation_encoder
        ):
            raise ValueError(
                "The decentralized TMASAC actor cannot share an observation encoder because "
                "the shared critic encoder mixes agents."
            )
        self.scenario_encoder_config = config.scenario_encoder_config
        if self.scenario_encoder_config is not None:
            if not getattr(env, "has_scenario_id", False):
                raise ValueError(
                    "TMASAC scenario encoding requires an environment observation_space with scenario_id."
                )
            if self.share_observation_encoder:
                raise ValueError(
                    "TMASAC scenario encoding does not yet support shared observation encoding."
                )
            self._validate_scenario_env_config(env)
            self._validate_scenario_nop_config(config)
        elif getattr(env, "has_scenario_id", False):
            raise ValueError(
                "An environment with scenario_id requires TMASACPolicyConfig.scenario_encoder_config."
            )

        self.actor_scenario_encoder = self._build_scenario_encoder(include_hidden_fields=False)
        self.critic_scenario_encoder = self._build_scenario_encoder(include_hidden_fields=True)
        self.critic_scenario_encoder_target = (
            None
            if self.critic_scenario_encoder is None
            else copy.deepcopy(self.critic_scenario_encoder)
        )
        if self.critic_scenario_encoder_target is not None:
            for parameter in self.critic_scenario_encoder_target.parameters():
                parameter.requires_grad_(False)

        actor_global_obs_dim = (
            self.global_obs_dim
            if self.actor_scenario_encoder is None
            else self.actor_scenario_encoder.global_output_dim
        )
        critic_global_obs_dim = (
            self.global_obs_dim
            if self.critic_scenario_encoder is None
            else self.critic_scenario_encoder.global_output_dim
        )
        self.critic_hidden_local_vars_dim = (
            self.hidden_local_vars_dim
            if self.critic_scenario_encoder is None
            else self.critic_scenario_encoder.hidden_local_output_dim
        )
        self.critic_hidden_global_vars_dim = (
            self.hidden_global_vars_dim
            if self.critic_scenario_encoder is None
            else self.critic_scenario_encoder.hidden_global_output_dim
        )

        self.actor_encoder_config = replace(
            config.actor_encoder_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
            use_agent_attention=(
                False
                if self.actor_head_kind is TMASACActorHeadKind.DECENTRALIZED
                else config.actor_encoder_config.use_agent_attention
            ),
        )
        self.critic_encoder_config = replace(
            config.critic_encoder_config,
            act_fn_cls=config.act_fn_cls,
            dropout=config.dropout,
        )
        self.shared_encoder_config = (
            None
            if not self.share_observation_encoder
            else replace(
                config.shared_encoder_config or config.actor_encoder_config,
                act_fn_cls=config.act_fn_cls,
                dropout=config.dropout,
            )
        )
        if self.nop_latent_source is SACNOPLatentSource.SHARED_ENCODER and self.shared_encoder_config is None:
            raise ValueError("SACNOPLatentSource.SHARED_ENCODER requires a shared observation encoder.")

        self.shared_observation_encoder = (
            None
            if self.shared_encoder_config is None
            else self._build_observation_encoder(
                self.shared_encoder_config,
                global_obs_dim=critic_global_obs_dim,
            )
        )
        self.shared_observation_encoder_target = (
            None
            if self.shared_encoder_config is None
            else self._build_observation_encoder(
                self.shared_encoder_config,
                global_obs_dim=critic_global_obs_dim,
            )
        )
        if self.shared_observation_encoder is not None and self.shared_observation_encoder_target is not None:
            self.shared_observation_encoder_target.load_state_dict(self.shared_observation_encoder.state_dict())
            for parameter in self.shared_observation_encoder_target.parameters():
                parameter.requires_grad_(False)

        actor_local_input_dim = (
            self.local_obs_dim
            if self.shared_encoder_config is None
            else self.shared_encoder_config.d_model
        )
        actor_global_input_dim = 0 if self.shared_encoder_config is not None else actor_global_obs_dim
        critic_local_input_dim = (
            self.local_obs_dim
            if self.shared_encoder_config is None
            else self.shared_encoder_config.d_model
        )
        critic_global_input_dim = 0 if self.shared_encoder_config is not None else critic_global_obs_dim

        self._actor_encoder = self._build_actor_encoder(
            local_obs_dim=actor_local_input_dim,
            global_obs_dim=actor_global_input_dim,
        )
        self.actor_head = self._build_actor_head()
        self.action_dist = HybridActionDistribution(
            latent_dim=self.actor_head.latent_dim,
            action_space=env.action_space,
            continuous_config=self._validated_continuous_config(env),
            bernoulli_config=None,
            action_net_initialization=make_init_linear_orthogonal(config.action_net_init_gain),
        )

        self.critic = self._build_critic(
            local_input_dim=critic_local_input_dim,
            global_input_dim=critic_global_input_dim,
        )
        self.critic_target = self._build_critic(
            local_input_dim=critic_local_input_dim,
            global_input_dim=critic_global_input_dim,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

        self.actor_nop = self._build_nop_module("actor")
        self.critic_nop = self._build_nop_module("critic")
        self._apply_optional_compile()
        if self.config.compile_modules and self.action_dist.compile_friendly:
            self._compiled_actor_action_samples = self._compile_callable(
                self._actor_action_samples_impl,
                fullgraph=True,
            )
        self._keep_target_modules_in_eval_mode()

    @property
    def actor_encoder(self) -> nn.Module:
        return self._actor_encoder

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self._keep_target_modules_in_eval_mode()
        return self

    def _keep_target_modules_in_eval_mode(self) -> None:
        if self.shared_observation_encoder_target is not None:
            self.shared_observation_encoder_target.train(False)
        if self.critic_scenario_encoder_target is not None:
            self.critic_scenario_encoder_target.train(False)
        self.critic_target.train(False)

    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_action_samples == 1:
            sample_actions = self._compiled_action_log_prob or self._action_log_prob_impl
            return sample_actions(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                scenario_ids=scenario_ids,
                previous_actions=previous_actions,
                deterministic=deterministic,
                use_rsample=use_rsample,
            )
        actor_latents = self.encode_actor(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        return self.action_log_prob_from_latents(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
        )

    def action_log_prob_from_latents(
            self,
            *,
            actor_latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool = False,
            use_rsample: bool = True,
            num_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For K > 1 return [K, ..., agents, actions] and [K, ..., agents].

        Keep recurrence outside the candidate axis; each QCX row is a joint action.
        K = 1 retains the original shapes and compiled sampling entry point.
        """
        if num_action_samples == 1:
            return self._actor_actions_and_log_probs(
                actor_latents=actor_latents,
                agent_mask=agent_mask,
                previous_actions=previous_actions,
                deterministic=deterministic,
                use_rsample=use_rsample,
            )
        sample_actions = self._compiled_actor_action_samples or self._actor_action_samples_impl
        return sample_actions(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
            num_action_samples=num_action_samples,
            action_sample_strategy=action_sample_strategy,
        )

    def _actor_action_samples_impl(
            self,
            *,
            actor_latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
            num_action_samples: int,
            action_sample_strategy: ActionSampleStrategy,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_head.actions_and_log_probs(
            actor_latents=expand_action_samples(actor_latents, num_action_samples),
            action_dist=self.action_dist,
            agent_mask=expand_action_samples(agent_mask, num_action_samples),
            previous_actions=expand_action_samples(previous_actions, num_action_samples),
            deterministic=deterministic,
            use_rsample=use_rsample,
            stratified_sample_dim=0 if action_sample_strategy == "stratified" else None,
        )

    def _action_log_prob_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        actor_latents = self._encode_actor_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        return self._actor_actions_and_log_probs(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )

    def _actor_actions_and_log_probs(
            self,
            *,
            actor_latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._compiled_actor_actions_and_log_probs is not None:
            return self._compiled_actor_actions_and_log_probs(
                actor_latents=actor_latents,
                agent_mask=agent_mask,
                previous_actions=previous_actions,
                deterministic=deterministic,
                use_rsample=use_rsample,
            )
        return self._actor_actions_and_log_probs_impl(
            actor_latents=actor_latents,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )

    def _actor_actions_and_log_probs_impl(
            self,
            *,
            actor_latents: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            use_rsample: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_head.actions_and_log_probs(
            actor_latents=actor_latents,
            action_dist=self.action_dist,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )

    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            critic_local_inputs,
            critic_global_inputs,
            critic_hidden_local_vars,
            critic_hidden_global_vars,
        ) = self._critic_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        q1, q2, _nop_latents = self.critic(
            local_obs=critic_local_inputs,
            global_obs=critic_global_inputs,
            actions=actions,
            hidden_local_vars=critic_hidden_local_vars,
            hidden_global_vars=critic_hidden_global_vars,
            agent_mask=agent_mask,
        )
        return q1, q2

    def q_values_with_nop_latents(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        (
            critic_local_inputs,
            critic_global_inputs,
            critic_hidden_local_vars,
            critic_hidden_global_vars,
        ) = self._critic_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        q1, q2, critic_latents = self.critic(
            local_obs=critic_local_inputs,
            global_obs=critic_global_inputs,
            actions=actions,
            hidden_local_vars=critic_hidden_local_vars,
            hidden_global_vars=critic_hidden_global_vars,
            agent_mask=agent_mask,
        )
        if self.critic_nop is None:
            return q1, q2, None
        if self.nop_latent_source is SACNOPLatentSource.SHARED_ENCODER:
            return q1, q2, critic_local_inputs
        return q1, q2, critic_latents

    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            critic_local_inputs,
            critic_global_inputs,
            critic_hidden_local_vars,
            critic_hidden_global_vars,
        ) = self._critic_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=True,
        )
        q1, q2, _nop_latents = self.critic_target(
            local_obs=critic_local_inputs,
            global_obs=critic_global_inputs,
            actions=actions,
            hidden_local_vars=critic_hidden_local_vars,
            hidden_global_vars=critic_hidden_global_vars,
            agent_mask=agent_mask,
        )
        return q1, q2

    def encode_critic(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        (
            critic_local_inputs,
            critic_global_inputs,
            critic_hidden_local_vars,
            critic_hidden_global_vars,
        ) = self._critic_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            target=False,
        )
        return self._critic_module().encode(
            local_inputs=critic_local_inputs,
            global_inputs=critic_global_inputs,
            actions=actions,
            hidden_local_vars=critic_hidden_local_vars,
            hidden_global_vars=critic_hidden_global_vars,
            agent_mask=agent_mask,
        )

    def encode_actor(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor_local_inputs, actor_global_inputs = self._actor_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        actor_encoder = (
            self._actor_encoder
            if self._compiled_actor_encoder is None
            else self._compiled_actor_encoder
        )
        return actor_encoder(actor_local_inputs, actor_global_inputs, agent_mask=agent_mask)

    def _encode_actor_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actor_local_inputs, actor_global_inputs = self._actor_observation_inputs(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        return self._actor_encoder(actor_local_inputs, actor_global_inputs, agent_mask=agent_mask)

    def actor_parameters(self) -> list[nn.Parameter]:
        return list(self._iter_parameters(
            self.actor_scenario_encoder,
            self.actor_encoder,
            self.actor_head,
            self.action_dist,
            self.actor_nop,
        ))

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self._iter_parameters(
            self.critic_scenario_encoder,
            self.shared_observation_encoder,
            self.critic,
            self.critic_nop,
        ))

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.actor_nop is None:
            return None, {}
        local_obs, global_obs, agent_mask, scenario_ids = self._nop_initial_obs(batch)
        actor_latents = self.encode_actor(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        return self.actor_nop.compute_loss(source_latents=actor_latents, batch=batch)

    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
            *,
            source_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if self.critic_nop is None:
            return None, {}
        if source_latents is None:
            (
                local_obs,
                global_obs,
                hidden_local_vars,
                hidden_global_vars,
                actions,
                agent_mask,
                scenario_ids,
            ) = self._nop_initial_critic_inputs(batch)
            if self.nop_latent_source is SACNOPLatentSource.SHARED_ENCODER:
                source_latents = self.encode_shared_observations(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    agent_mask=agent_mask,
                    target=False,
                )
            else:
                source_latents = self.encode_critic(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    actions=actions,
                    hidden_local_vars=hidden_local_vars,
                    hidden_global_vars=hidden_global_vars,
                    agent_mask=agent_mask,
                    scenario_ids=scenario_ids,
                )
        return self.critic_nop.compute_loss(source_latents=source_latents, batch=batch)

    def has_nop_loss(self) -> bool:
        return self.actor_nop is not None or self.critic_nop is not None

    def get_nop_num_next_steps(self) -> int:
        return self.config.nop_config.num_next_steps

    def polyak_update_targets(self, tau: float) -> None:
        if self.critic_scenario_encoder is not None and self.critic_scenario_encoder_target is not None:
            polyak_update(
                self.critic_scenario_encoder,
                self.critic_scenario_encoder_target,
                tau,
            )
        if self.shared_observation_encoder is not None and self.shared_observation_encoder_target is not None:
            polyak_update(self.shared_observation_encoder, self.shared_observation_encoder_target, tau)
        polyak_update(self.critic, self.critic_target, tau)

    def get_hyper_parameters(self) -> dict[str, Any]:
        policy_config = {
            "actor_encoder_config": serialize_dataclass(self.actor_encoder_config),
            "critic_encoder_config": serialize_dataclass(self.critic_encoder_config),
            "shared_encoder_config": (
                None
                if self.shared_encoder_config is None
                else serialize_dataclass(self.shared_encoder_config)
            ),
            "share_observation_encoder": self.share_observation_encoder,
            "actor_head_config": serialize_dataclass(self.config.actor_head_config),
            "critic_config": serialize_dataclass(self.config.critic_config),
            "act_fn_cls": serialize_value(self.act_fn_cls),
            "dropout": self.dropout,
            "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
            "max_agents": self.max_agents,
            "nop_config": serialize_dataclass(self.config.nop_config),
            "nop_latent_source": self.nop_latent_source.value,
            "actor_nop": None if self.actor_nop is None else self.actor_nop.get_hyper_parameters(),
            "critic_nop": None if self.critic_nop is None else self.critic_nop.get_hyper_parameters(),
            "compile_modules": self.config.compile_modules,
            "compile_mode": self.config.compile_mode,
            "action_net_init_gain": self.config.action_net_init_gain,
            "action_dist_compile_friendly": self.action_dist.compile_friendly,
        }
        if self.scenario_encoder_config is not None:
            policy_config["scenario_encoder_config"] = serialize_dataclass(
                self.scenario_encoder_config
            )
        return {
            "tmasac_policy_config": policy_config
        }

    def get_grad_norms(self) -> dict[str, float]:
        metrics = {
            "shared_observation_encoder": self._module_grad_norm(self.shared_observation_encoder),
            "shared_observation_encoder_target": self._module_grad_norm(self.shared_observation_encoder_target),
            "actor_encoder": self._module_grad_norm(self.actor_encoder),
            "actor_head": self._module_grad_norm(self.actor_head),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "critic_target": self._module_grad_norm(self.critic_target),
            "actor_nop": self._module_grad_norm(self.actor_nop),
            "critic_nop": self._module_grad_norm(self.critic_nop),
            "total": self._module_grad_norm(self),
        }
        if self.scenario_encoder_config is not None:
            metrics.update({
                "actor_scenario_encoder": self._module_grad_norm(self.actor_scenario_encoder),
                "critic_scenario_encoder": self._module_grad_norm(self.critic_scenario_encoder),
                "critic_scenario_encoder_target": self._module_grad_norm(
                    self.critic_scenario_encoder_target
                ),
            })
        return metrics

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return
        remaining_weights = dict(weights)
        nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("nop_loss_coef", "world_model_loss_coef", "wm_loss_coef"),
        )
        if nop_weight is not None:
            alias, value = nop_weight
            self._set_all_nop_loss_coefs(alias=alias, value=value)

        actor_nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("actor_nop_loss_coef", "actor_world_model_loss_coef"),
        )
        if actor_nop_weight is not None:
            alias, value = actor_nop_weight
            self._set_nop_loss_coef(self.actor_nop, alias=alias, value=value)

        critic_nop_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("critic_nop_loss_coef", "critic_world_model_loss_coef"),
        )
        if critic_nop_weight is not None:
            alias, value = critic_nop_weight
            self._set_nop_loss_coef(self.critic_nop, alias=alias, value=value)

        if remaining_weights:
            raise ValueError(f"Unknown weights given: {remaining_weights}")

    def encode_shared_observations(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            target: bool,
    ) -> torch.Tensor:
        encoder = self.shared_observation_encoder_target if target else self.shared_observation_encoder
        if encoder is None:
            raise RuntimeError("TMASACPolicy has no shared observation encoder configured.")
        return encoder(local_obs, global_obs, agent_mask=agent_mask)

    def _actor_observation_inputs(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_obs = self._encode_scenario_global_obs(
            encoder=self.actor_scenario_encoder,
            global_obs=global_obs,
            scenario_ids=scenario_ids,
        )
        if self.shared_observation_encoder is None:
            return local_obs, global_obs
        shared_latents = self.encode_shared_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            target=False,
        )
        return shared_latents.detach(), global_obs

    def _critic_observation_inputs(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            scenario_ids: torch.Tensor | None,
            target: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        scenario_encoder = (
            self.critic_scenario_encoder_target
            if target
            else self.critic_scenario_encoder
        )
        global_obs = self._encode_scenario_global_obs(
            encoder=scenario_encoder,
            global_obs=global_obs,
            scenario_ids=scenario_ids,
        )
        if scenario_encoder is not None:
            assert scenario_ids is not None
            if hidden_local_vars is None or hidden_global_vars is None:
                raise ValueError(
                    "TMASAC scenario encoding requires hidden_local_vars and hidden_global_vars."
                )
            hidden_local_vars = scenario_encoder.encode_hidden_local_vars(
                hidden_local_vars,
                scenario_ids,
            )
            hidden_global_vars = scenario_encoder.encode_hidden_global_vars(
                hidden_global_vars,
                scenario_ids,
            )
        encoder = self.shared_observation_encoder_target if target else self.shared_observation_encoder
        if encoder is None:
            return local_obs, global_obs, hidden_local_vars, hidden_global_vars
        shared_latents = self.encode_shared_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            target=target,
        )
        return shared_latents, global_obs, hidden_local_vars, hidden_global_vars

    @staticmethod
    def _encode_scenario_global_obs(
            *,
            encoder: TMASACScenarioObservationEncoder | None,
            global_obs: torch.Tensor,
            scenario_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if encoder is None:
            if scenario_ids is not None:
                raise ValueError(
                    "scenario_ids were provided, but TMASACPolicy has no scenario encoder configured."
                )
            return global_obs
        if scenario_ids is None:
            raise ValueError(
                "TMASACPolicy scenario encoding is configured, but scenario_ids were not provided."
            )
        return encoder.encode_global_obs(global_obs, scenario_ids)

    def _build_actor_head(self) -> TMASACActorHead:
        if self.actor_head_kind is TMASACActorHeadKind.QCX:
            return TMASACQCXActorHead(
                actor_latent_dim=self.actor_encoder_config.d_model,
                action_dim=self.agent_action_dim,
                n_agents=self.n_agents,
                max_agents=self.max_agents,
                config=self.config.actor_head_config,
                act_fn_cls=self.config.act_fn_cls,
                dropout=self.config.dropout,
            )
        return TMASACIndependentActorHead(
            d_model=self.actor_encoder_config.d_model,
            config=self.config.actor_head_config,
            act_fn_cls=self.config.act_fn_cls,
        )

    def _build_actor_encoder(
            self,
            *,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> nn.Module:
        return self._build_observation_encoder(
            self.actor_encoder_config,
            local_obs_dim=local_obs_dim,
            global_obs_dim=global_obs_dim,
        )

    def _build_critic(
            self,
            *,
            local_input_dim: int,
            global_input_dim: int,
    ) -> TMASACTwinCritic:
        return TMASACTwinCritic(
            n_agents=self.n_agents,
            max_agents=self.max_agents,
            local_input_dim=local_input_dim,
            global_input_dim=global_input_dim,
            hidden_local_vars_dim=self.critic_hidden_local_vars_dim,
            hidden_global_vars_dim=self.critic_hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            encoder_config=self.critic_encoder_config,
            critic_config=self.config.critic_config,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_scenario_encoder(
            self,
            *,
            include_hidden_fields: bool,
    ) -> TMASACScenarioObservationEncoder | None:
        if self.scenario_encoder_config is None:
            return None
        return TMASACScenarioObservationEncoder(
            config=self.scenario_encoder_config,
            global_obs_dim=self.global_obs_dim,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim,
            act_fn_cls=self.config.act_fn_cls,
            include_hidden_fields=include_hidden_fields,
        )

    def _build_observation_encoder(
            self,
            config: MATEncoderConfig,
            *,
            local_obs_dim: int | None = None,
            global_obs_dim: int | None = None,
    ) -> MATEncoder:
        return MATEncoder(
            config=config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim if local_obs_dim is None else int(local_obs_dim),
            global_obs_dim=self.global_obs_dim if global_obs_dim is None else int(global_obs_dim),
        )

    def _build_nop_module(self, source_name: str) -> SACNOPModule | None:
        nop_config = self.config.nop_config
        if not nop_config.enabled:
            return None
        source = self.nop_latent_source
        module_name = source_name
        if source_name == "actor":
            if source not in (SACNOPLatentSource.ACTOR, SACNOPLatentSource.BOTH):
                return None
            source_latent_dim = self.actor_encoder_config.d_model
        elif source_name == "critic":
            if source is SACNOPLatentSource.SHARED_ENCODER:
                if self.shared_encoder_config is None:
                    raise ValueError(
                        "SACNOPLatentSource.SHARED_ENCODER requires a shared observation encoder."
                    )
                source_latent_dim = self.shared_encoder_config.d_model
                module_name = "shared_encoder"
            elif source in (SACNOPLatentSource.CRITIC, SACNOPLatentSource.BOTH):
                source_latent_dim = self._critic_module().nop_source_latent_dim
            else:
                return None
        else:
            raise ValueError(f"Unknown NOP source {source_name!r}")
        return SACNOPModule(
            n_agents=self.n_agents,
            source_latent_dim=source_latent_dim,
            action_dim=self.agent_action_dim,
            config=nop_config,
            name=module_name,
            skip_first_transition=(
                nop_config.skip_first_transition_for_critic
                and source_name == "critic"
                and source in (SACNOPLatentSource.CRITIC, SACNOPLatentSource.BOTH)
            ),
        )

    def _validated_continuous_config(self, env: BaseLearnEnvWrapper) -> ContinuousActionDistConfigInput:
        config = self.config.continuous_config
        configs = config if isinstance(config, list) else [config] * env.action_space.n_spaces
        if len(configs) != env.action_space.n_spaces:
            raise ValueError(
                f"Expected {env.action_space.n_spaces} continuous configs, got {len(configs)}"
            )
        for idx, sub_config in enumerate(configs):
            if sub_config is None:
                raise ValueError(f"TMASACPolicy requires a continuous action config for action sub-space {idx}.")
            gradient_estimator = continuous_action_gradient_estimator(sub_config)
            if not gradient_estimator.supports_actor_gradients:
                raise ValueError(
                    "TMASACPolicy requires pathwise or straight-through actor gradients for SAC; "
                    f"action sub-space {idx} got {type(sub_config).__name__} "
                    f"with gradient estimator {gradient_estimator.value!r}."
                )
        return config

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return
        if not hasattr(torch, "compile") or not callable(torch.compile):
            raise RuntimeError("TMASACPolicyConfig.compile_modules=True requires torch.compile support.")
        if not self.config.compile_mode:
            raise ValueError("TMASACPolicyConfig.compile_mode must be a non-empty string when compile_modules=True.")

        if self.shared_observation_encoder is not None:
            self.shared_observation_encoder = self._compile_module(self.shared_observation_encoder)
        if self.shared_observation_encoder_target is not None:
            self.shared_observation_encoder_target = self._compile_module(self.shared_observation_encoder_target)
        compile_actor_end_to_end = (
            self.action_dist.compile_friendly
            and self.shared_observation_encoder is None
        )
        if compile_actor_end_to_end:
            object.__setattr__(
                self,
                "_compiled_actor_encoder",
                self._compile_module(self._actor_encoder),
            )
            self._compiled_action_log_prob = self._compile_callable(
                self._action_log_prob_impl,
                fullgraph=True,
            )
        else:
            self._actor_encoder = self._compile_module(self._actor_encoder)
            if self.action_dist.compile_friendly:
                self._compiled_actor_actions_and_log_probs = self._compile_callable(
                    self._actor_actions_and_log_probs_impl,
                    fullgraph=True,
                )
            elif self.actor_head.supports_standalone_compile:
                self.actor_head = self._compile_module(self.actor_head)
        self.critic = self._compile_module(self.critic)
        self.critic_target = self._compile_module(self.critic_target)

    def _compile_callable(
            self,
            fn: Callable[..., Any],
            *,
            fullgraph: bool,
    ) -> Callable[..., Any]:
        return torch.compile(
            fn,
            mode=self.config.compile_mode,
            fullgraph=fullgraph,
            dynamic=False,
        )

    def _compile_module(self, module: nn.Module) -> nn.Module:
        return torch.compile(
            module,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    def _critic_module(self) -> TMASACTwinCritic:
        if isinstance(self.critic, TMASACTwinCritic):
            return self.critic
        orig_mod = getattr(self.critic, "_orig_mod", None)
        if isinstance(orig_mod, TMASACTwinCritic):
            return orig_mod
        raise RuntimeError(f"Cannot resolve TMASACTwinCritic from compiled module {type(self.critic).__name__}")

    @staticmethod
    def _nop_initial_obs(
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch):
            return (
                batch.local_obs[:, 0],
                batch.global_obs[:, 0],
                None if batch.agent_mask is None else batch.agent_mask[:, 0],
                None if batch.scenario_ids is None else batch.scenario_ids[:, 0],
            )
        return batch.local_obs, batch.global_obs, batch.agent_mask, batch.scenario_ids

    @staticmethod
    def _nop_initial_critic_inputs(
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if isinstance(batch, OffPolicyReplayEpisodeSegmentBatch):
            return (
                batch.local_obs[:, 0],
                batch.global_obs[:, 0],
                None if batch.hidden_local_vars is None else batch.hidden_local_vars[:, 0],
                None if batch.hidden_global_vars is None else batch.hidden_global_vars[:, 0],
                batch.actions[:, 0],
                None if batch.agent_mask is None else batch.agent_mask[:, 0],
                None if batch.scenario_ids is None else batch.scenario_ids[:, 0],
            )
        return (
            batch.local_obs,
            batch.global_obs,
            batch.hidden_local_vars,
            batch.hidden_global_vars,
            batch.actions,
            batch.agent_mask,
            batch.scenario_ids,
        )

    @staticmethod
    def _normalize_actor_head_kind(kind: TMASACActorHeadKind | str) -> TMASACActorHeadKind:
        if isinstance(kind, TMASACActorHeadKind):
            return kind
        try:
            return TMASACActorHeadKind(kind.lower())
        except ValueError as exc:
            valid = [item.value for item in TMASACActorHeadKind]
            raise ValueError(f"Unknown TMASAC actor head kind {kind!r}; expected one of {valid}") from exc

    @staticmethod
    def _validate_continuous_action_space(env: BaseLearnEnvWrapper) -> None:
        for key, sub_space in env.action_space.items():
            if not isinstance(sub_space, spaces.Box):
                raise ValueError(
                    "SAC/TMASAC currently supports continuous Box action sub-spaces only; "
                    f"sub-space {key!r} is {type(sub_space).__name__}."
                )

    @staticmethod
    def _validate_config(config: TMASACPolicyConfig) -> None:
        action_encoder_dim = config.critic_config.action_encoder_dim
        if action_encoder_dim is not None and action_encoder_dim <= 0:
            raise ValueError(f"action_encoder_dim must be positive, got {action_encoder_dim}")
        if config.critic_config.use_popart:
            raise NotImplementedError(
                "TMASACPolicy does not support PopArt critics yet because SAC must update PopArt target "
                "statistics during critic training. Set critic_config.use_popart=False."
            )

    @staticmethod
    def _validate_scenario_nop_config(config: TMASACPolicyConfig) -> None:
        if not config.nop_config.enabled:
            return
        target_config = config.nop_config.next_obs_pred_config
        if target_config.global_scalar_target_indices or target_config.global_rot6d_target_indices:
            raise ValueError(
                "TMASAC scenario encoding does not yet support global NOP targets. "
                "Use local NOP targets only."
            )

    def _validate_scenario_env_config(self, env: BaseLearnEnvWrapper) -> None:
        assert self.scenario_encoder_config is not None
        scenario_names = getattr(env, "scenario_names", None)
        if scenario_names is not None:
            configured_names = tuple(
                scenario.name
                for scenario in self.scenario_encoder_config.scenarios
            )
            if tuple(scenario_names) != configured_names:
                raise ValueError(
                    "Scenario encoder names/order must match the environment: "
                    f"configured={configured_names}, environment={tuple(scenario_names)}."
                )
        scenario_dims = getattr(env, "scenario_observation_dims", None)
        if scenario_dims is None:
            return
        for scenario in self.scenario_encoder_config.scenarios:
            expected_dims = {
                "global_obs": scenario.global_obs_dim,
                "hidden_local_vars": scenario.hidden_local_vars_dim,
                "hidden_global_vars": scenario.hidden_global_vars_dim,
            }
            if scenario_dims.get(scenario.name) != expected_dims:
                raise ValueError(
                    f"Scenario encoder dimensions for {scenario.name!r} do not match "
                    f"the environment: configured={expected_dims}, "
                    f"environment={scenario_dims.get(scenario.name)}."
                )

    @staticmethod
    def _iter_parameters(*modules: nn.Module | None) -> Iterable[nn.Parameter]:
        for module in modules:
            if module is None:
                continue
            yield from module.parameters()

    @staticmethod
    def _pop_loss_weight_alias(
            weights: dict[str, float],
            *,
            aliases: tuple[str, ...],
    ) -> tuple[str, float] | None:
        matching_aliases = [alias for alias in aliases if alias in weights]
        if not matching_aliases:
            return None
        if len(matching_aliases) > 1:
            raise ValueError(f"Multiple aliases for the same loss weight are not allowed: {matching_aliases}")
        alias = matching_aliases[0]
        value = float(weights.pop(alias))
        return alias, value

    def _set_all_nop_loss_coefs(self, *, alias: str, value: float) -> None:
        self._validate_nop_loss_coef(alias=alias, value=value)
        if self.actor_nop is not None:
            self.actor_nop.nop_loss_coef = value
        if self.critic_nop is not None:
            self.critic_nop.nop_loss_coef = value

    def _set_nop_loss_coef(self, module: SACNOPModule | None, *, alias: str, value: float) -> None:
        self._validate_nop_loss_coef(alias=alias, value=value)
        if module is None:
            raise ValueError(f"{alias} was provided but the corresponding NOP module is disabled.")
        module.nop_loss_coef = value

    @staticmethod
    def _validate_nop_loss_coef(*, alias: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{alias} must be >= 0, got {value}")
