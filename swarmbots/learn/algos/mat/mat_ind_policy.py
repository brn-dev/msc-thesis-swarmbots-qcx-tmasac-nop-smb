from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.hybrid_action_dist import (
    ContinuousActionDistConfigInput,
    HybridActionDistribution,
    bernoulli_config_to_dict,
    continuous_config_to_dicts,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import (
    MATQCSCriticConfig,
    _ensure_torch_compile_available,
)
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import (
    PPOSampler,
    PPOSamplerConfig,
    PPOSamples,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import (
    BaseLearnEnvWrapper,
)
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_value


@dataclass(frozen=True)
class MATIndPolicyConfig:
    encoder_config: MATEncoderConfig = field(default_factory=MATEncoderConfig)
    critic_config: MATQCSCriticConfig = field(default_factory=MATQCSCriticConfig)
    actor_head_hidden_dims: list[int] | None = None
    act_fn_cls: ActivationFactory = nn.GELU
    dropout: float = 0.0
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    max_agents: int | None = None
    compile_modules: bool = False
    compile_mode: str = "default"
    actor_head_init_gain: float = 1.0
    action_net_init_gain: float = 0.01


class MATIndPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATIndPolicyConfig | None = None,
    ) -> None:
        super().__init__()
        config = MATIndPolicyConfig() if config is None else config
        self.config = config

        self.n_agents: int = env.n_agents
        self.max_agents = self.n_agents if config.max_agents is None else config.max_agents
        if self.max_agents < self.n_agents:
            raise ValueError(
                f"max_agents must be >= env.n_agents ({self.n_agents}), got {self.max_agents}"
            )

        self.local_obs_dim: int = env.local_obs_dim
        self.global_obs_dim: int = env.global_obs_dim
        self.hidden_local_vars_dim: int = env.hidden_local_vars_dim
        self.hidden_global_vars_dim: int = env.hidden_global_vars_dim
        self.act_fn_cls = config.act_fn_cls
        self.dropout = config.dropout
        self.agent_action_dim = env.action_space.total_agent_action_dim

        self.d_model_encoder = config.encoder_config.d_model
        self.encoder_config = self._build_encoder_config()
        self.encoder = self._build_encoder()

        if config.actor_head_hidden_dims is not None and len(config.actor_head_hidden_dims) > 0:
            self.actor_head = MLP(
                input_dim=self._actor_input_dim(),
                hidden_dims=config.actor_head_hidden_dims,
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.actor_head_init_gain),
                act_fn_cls=config.act_fn_cls,
            )
            latent_pi_dim = config.actor_head_hidden_dims[-1]
        else:
            self.actor_head = nn.Identity()
            latent_pi_dim = self._actor_input_dim()

        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
            action_net_initialization=make_init_linear_orthogonal(config.action_net_init_gain),
        )

        self.critic = DeepSetCritic(
            num_local_features=self.d_model_encoder + self.hidden_local_vars_dim,
            local_projection_hidden_dims=[self.d_model_encoder] * config.critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[self.d_model_encoder] * config.critic_config.n_value_regressor_hidden_layers,
            num_global_features=self.hidden_global_vars_dim,
            act_fn_cls=config.act_fn_cls,
            context_in_elements=self.hidden_global_vars_dim > 0,
            local_projection_linear_init_gain=config.critic_config.local_projection_init_gain,
            value_regressor_linear_init_gain=config.critic_config.value_regressor_init_gain,
            value_head_linear_init_gain=config.critic_config.value_head_init_gain,
            use_popart=config.critic_config.use_popart,
            popart_beta=config.critic_config.popart_config.beta,
            popart_eps=config.critic_config.popart_config.eps,
            popart_min_std=config.critic_config.popart_config.min_std,
            popart_init_sigma=config.critic_config.popart_config.init_sigma,
        )

        self._generate_actions_fn: Callable[..., tuple[torch.Tensor, torch.Tensor | None]] = self._generate_actions_impl
        self._evaluate_latent_and_values_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_latent_and_values_impl
        )
        self._evaluate_actions_core_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            self._evaluate_actions_core_impl
        )
        self._apply_optional_compile()

    def _build_encoder_config(self) -> MATEncoderConfig:
        return replace(
            self.config.encoder_config,
            d_model=self.d_model_encoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_encoder(self) -> nn.Module:
        return MATEncoder(
            config=self.encoder_config,
            max_agents=self.max_agents,
            local_obs_dim=self.local_obs_dim,
            global_obs_dim=self.global_obs_dim,
        )

    def _actor_input_dim(self) -> int:
        return self.d_model_encoder

    def _actor_observations(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            augmented_observations: torch.Tensor,
    ) -> torch.Tensor:
        _ = local_obs
        _ = global_obs
        return augmented_observations

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mat_ind_policy_config": {
                "encoder_config": serialize_dataclass(self.encoder_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "actor_head_hidden_dims": self.config.actor_head_hidden_dims,
                "act_fn_cls": serialize_value(self.act_fn_cls),
                "dropout": self.dropout,
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "max_agents": self.max_agents,
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
                "actor_head_init_gain": self.config.actor_head_init_gain,
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
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, log_probs = self._generate_actions(
            augmented_observations=self._actor_observations(
                local_obs,
                global_obs,
                augmented_observations,
            ),
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
        return actions, log_probs, values

    def _generate_actions(
            self,
            *,
            augmented_observations: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._generate_actions_fn(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=return_log_probs,
        )

    def _generate_actions_impl(
            self,
            *,
            augmented_observations: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent_pi = self.actor_head(augmented_observations).contiguous()
        if return_log_probs:
            actions, log_probs = self.action_dist.get_on_policy_actions_with_log_probs(
                latent_pi,
                deterministic=deterministic,
                previous_actions=previous_actions,
            )
            return self._mask_actions(actions, agent_mask), self._mask_log_probs(log_probs, agent_mask)

        actions = self.action_dist.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            previous_actions=previous_actions,
        )
        return self._mask_actions(actions, agent_mask), None

    def _evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        actions = self._policy_actions(batch.actions)
        if self.action_dist.compile_friendly:
            log_probs, values, augmented_observations = self._evaluate_actions_core(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                previous_actions=batch.previous_actions,
                actions=actions,
            )
        else:
            latent_pi, values, augmented_observations = self._evaluate_latent_and_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
            )
            self.action_dist.update_latent_features(latent_pi)
            log_probs = self.action_dist.log_prob(actions, previous_actions=batch.previous_actions)
            log_probs = self._mask_log_probs(log_probs, batch.agent_mask)

        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(
            agent_mask=batch.agent_mask,
            action_splitter=action_splitter,
        )
        return log_probs, values, extra_losses, extra_loss_metrics, augmented_observations

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
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        return self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )

    def _evaluate_latent_and_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._evaluate_latent_and_values_fn(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    def _evaluate_latent_and_values_impl(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actor_observations = self._actor_observations(
            local_obs,
            global_obs,
            augmented_observations,
        )
        latent_pi = self.actor_head(actor_observations).contiguous()
        values = self._critic_with_hidden_vars(
            augmented_observations,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask=agent_mask,
        )
        return latent_pi, values, augmented_observations

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
        latent_pi, values, augmented_observations = self._evaluate_latent_and_values_impl(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions, previous_actions=previous_actions)
        return self._mask_log_probs(log_probs, agent_mask), values, augmented_observations

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
        augmented_observations = self.encoder(local_obs, global_obs, agent_mask=agent_mask)
        actions, _ = self._generate_actions(
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
            config: PPOSamplerConfig,
    ) -> PPOSampler:
        return PPOSampler(
            episodes=episodes,
            config=config,
            requires_previous_actions=self.requires_previous_actions(),
        )

    def _critic_with_hidden_vars(
            self,
            augmented_observations: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
            hidden_global_vars: torch.Tensor | None,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic_local_obs = augmented_observations
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            critic_local_obs = torch.cat((augmented_observations, hidden_local_vars), dim=-1)

        if self.hidden_global_vars_dim <= 0:
            return self.critic(critic_local_obs, agent_mask=agent_mask)
        if hidden_global_vars is None:
            raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
        return self.critic(critic_local_obs, hidden_global_vars, agent_mask=agent_mask)

    @property
    def has_popart(self) -> bool:
        return getattr(self.critic, "has_popart", False)

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
            "actor_head": self._module_grad_norm(self.actor_head),
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
        self.actor_head = self._compile_module(self.actor_head)
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

    @staticmethod
    def _mask_actions(actions: torch.Tensor, agent_mask: torch.Tensor | None) -> torch.Tensor:
        if agent_mask is None:
            return actions
        return actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _mask_log_probs(log_probs: torch.Tensor, agent_mask: torch.Tensor | None) -> torch.Tensor:
        if agent_mask is None:
            return log_probs
        return log_probs.masked_fill(~agent_mask, 0.0)
