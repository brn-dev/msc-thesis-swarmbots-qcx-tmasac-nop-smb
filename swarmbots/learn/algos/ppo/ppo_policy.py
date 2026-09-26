from dataclasses import dataclass, field
from typing import Any
import shutil
import sys
import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    continuous_config_to_dicts,
    bernoulli_config_to_dict,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal, make_init_linear_orthogonal
from swarmbots.learn.nn_components.popart import PopArtLinear
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class PopArtConfig:
    beta: float = 3e-4
    eps: float = 1e-5
    min_std: float = 1e-4
    init_sigma: float = 1.0


@dataclass(frozen=True)
class PPOActorConfig:
    hidden_dims: list[int] = field(default_factory=list)
    shared_encoder_latent_dim_per_agent: int | None = None
    actor_head_hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim_per_agent: int = 64
    act_fun_class: type[nn.Module] = nn.Tanh
    actor_head_init_gain: float = 0.01


@dataclass(frozen=True)
class PPOCriticConfig:
    hidden_dims: list[int] = field(default_factory=list)
    act_fun_class: type[nn.Module] = nn.Tanh
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    value_head_init_gain: float = 0.01


@dataclass(frozen=True)
class PPOPolicyConfig:
    actor_config: PPOActorConfig = field(default_factory=PPOActorConfig)
    critic_config: PPOCriticConfig = field(default_factory=PPOCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    compile_modules: bool = False
    compile_mode: str = "default"


class PPOSharedEncoder(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            config: PPOActorConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        encoder_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = list(config.hidden_dims)
        self.local_latent_dim = (
            config.latent_pi_dim_per_agent
            if config.shared_encoder_latent_dim_per_agent is None
            else config.shared_encoder_latent_dim_per_agent
        )
        self.act_fun_class = encoder_act_fun_class

        self.mlp = MLP(
            input_dim=local_obs_dim * n_agents + global_obs_dim,
            hidden_dims=[*config.hidden_dims, self.local_latent_dim * n_agents],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=encoder_act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        encoder_input = torch.flatten(local_obs, start_dim=1)
        if self.has_global_obs:
            encoder_input = torch.cat((encoder_input, global_obs), dim=-1)
        return self.mlp(encoder_input).view((local_obs.shape[0], self.n_agents, self.local_latent_dim))


class PPOActor(nn.Module):

    def __init__(
            self,
            local_latent_dim: int,
            config: PPOActorConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        actor_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.local_latent_dim = local_latent_dim
        self.latent_pi_dim = config.latent_pi_dim_per_agent
        self.hidden_dims = list(config.actor_head_hidden_dims)
        self.act_fun_class = actor_act_fun_class

        self.mlp = MLP(
            input_dim=local_latent_dim,
            hidden_dims=[*config.actor_head_hidden_dims, config.latent_pi_dim_per_agent],
            end_with_act_fn=True,
            linear_init=linear_init,
            final_linear_init=make_init_linear_orthogonal(config.actor_head_init_gain),
            act_fn_cls=actor_act_fun_class,
        )

    def forward(self, local_latents: torch.Tensor) -> torch.Tensor:
        return self.mlp(local_latents)


class PPOCritic(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            config: PPOCriticConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        critic_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.n_agents = n_agents
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.has_global_obs = global_obs_dim > 0
        self.hidden_dims = list(config.hidden_dims)
        self.act_fun_class = critic_act_fun_class
        self.use_popart = bool(config.use_popart)
        self.popart_config = config.popart_config

        value_head_input_dim = local_obs_dim * n_agents + global_obs_dim
        if config.hidden_dims:
            self.value_features: nn.Module = MLP(
                input_dim=value_head_input_dim,
                hidden_dims=config.hidden_dims,
                end_with_act_fn=True,
                linear_init=linear_init,
                act_fn_cls=critic_act_fun_class
            )
            value_head_input_dim = int(config.hidden_dims[-1])
        else:
            self.value_features = nn.Identity()

        if self.use_popart:
            self.value_head: nn.Module = PopArtLinear(
                in_features=value_head_input_dim,
                out_features=1,
                beta=config.popart_config.beta,
                eps=config.popart_config.eps,
                min_std=config.popart_config.min_std,
                init_sigma=config.popart_config.init_sigma,
                init_gain=config.value_head_init_gain,
            )
        else:
            self.value_head = nn.Linear(value_head_input_dim, 1)
            make_init_linear_orthogonal(config.value_head_init_gain)(self.value_head)

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.shape != local_obs.shape[:2]:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(local_obs.shape[:2])}, got {tuple(agent_mask.shape)}"
                )
            local_obs = local_obs * agent_mask.to(dtype=local_obs.dtype).unsqueeze(-1)
        critic_input = torch.flatten(local_obs, start_dim=1)
        if self.has_global_obs:
            if global_obs is None:
                raise ValueError("global_obs is required when PPOCritic has global features")
            critic_input = torch.cat((critic_input, global_obs), dim=-1)
        critic_features = self.value_features(critic_input)
        return self.value_head(critic_features).squeeze(dim=-1)

    @property
    def has_popart(self) -> bool:
        return isinstance(self.value_head, PopArtLinear)

    def update_popart(self, targets: torch.Tensor) -> None:
        if not isinstance(self.value_head, PopArtLinear):
            raise RuntimeError("PopArt is not enabled for this PPOCritic")
        self.value_head.update(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.value_head, PopArtLinear):
            return values
        return self.value_head.normalize(values)

    def get_popart_metrics(self) -> dict[str, float]:
        if not isinstance(self.value_head, PopArtLinear):
            return {}
        sigma = self.value_head.sigma()
        return {
            "popart_mu": self.value_head.mu.mean().item(),
            "popart_sigma": sigma.mean().item(),
        }


class PPOPolicy(BasePPOPolicy[PPOSamples, PPOSamplerConfig]):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: PPOPolicyConfig = PPOPolicyConfig(),
    ):
        super().__init__()
        self.config = config

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.latent_pi_dim = config.actor_config.latent_pi_dim_per_agent
        self.local_latent_dim = (
            config.actor_config.latent_pi_dim_per_agent
            if config.actor_config.shared_encoder_latent_dim_per_agent is None
            else config.actor_config.shared_encoder_latent_dim_per_agent
        )

        self.shared_encoder = PPOSharedEncoder(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            config=config.actor_config,
        )
        self.actor = PPOActor(
            local_latent_dim=self.local_latent_dim,
            config=config.actor_config,
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=config.actor_config.latent_pi_dim_per_agent,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
        )

        self.critic = PPOCritic(
            n_agents=env.n_agents,
            local_obs_dim=self.local_latent_dim + self.hidden_local_vars_dim,
            global_obs_dim=self.hidden_global_vars_dim,
            config=config.critic_config,
        )
        self._apply_optional_compile()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "ppo_policy_config": {
                "actor_config": serialize_dataclass(self.config.actor_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
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
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        local_latents = self.shared_encoder(local_obs, global_obs)
        latent_pi = self.actor(local_latents)
        actions, log_probs = self.action_dist.get_on_policy_actions_with_log_probs(
            latent_pi,
            deterministic,
            previous_actions=previous_actions,
        )

        critic_local_obs = self._build_critic_local_obs(local_latents, hidden_local_vars)
        critic_global_obs = self._build_critic_global_obs(hidden_global_vars)
        values = self.critic(critic_local_obs, critic_global_obs, agent_mask=agent_mask)
        return actions, log_probs, values

    def _evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        local_obs = batch.local_obs
        global_obs = batch.global_obs
        hidden_local_vars = batch.hidden_local_vars
        hidden_global_vars = batch.hidden_global_vars
        agent_mask = batch.agent_mask
        previous_actions = batch.previous_actions
        actions = self._policy_actions(batch.actions)

        local_obs = self._mask_local_obs(local_obs, agent_mask)
        local_latents = self.shared_encoder(local_obs, global_obs)
        latent_pi = self.actor(local_latents)

        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions, previous_actions=previous_actions)

        critic_local_obs = self._build_critic_local_obs(local_latents, hidden_local_vars)
        critic_global_obs = self._build_critic_global_obs(hidden_global_vars)
        values = self.critic(critic_local_obs, critic_global_obs, agent_mask=agent_mask)

        extra_losses, extra_loss_metrics = self.action_dist.compute_extra_losses(
            agent_mask=agent_mask,
            action_splitter=action_splitter,
        )
        return log_probs, values, extra_losses, extra_loss_metrics, local_latents

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
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        local_latents = self.shared_encoder(local_obs, global_obs)
        critic_local_obs = self._build_critic_local_obs(local_latents, hidden_local_vars)
        critic_global_obs = self._build_critic_global_obs(hidden_global_vars)
        return self.critic(critic_local_obs, critic_global_obs, agent_mask=agent_mask)

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> torch.Tensor:
        _ = hidden_local_vars
        _ = hidden_global_vars
        local_obs = self._mask_local_obs(local_obs, agent_mask)
        local_latents = self.shared_encoder(local_obs, global_obs)
        latent_pi = self.actor(local_latents)
        actions = self.action_dist.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            previous_actions=previous_actions,
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

    @staticmethod
    def _mask_local_obs(
            local_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if agent_mask is None:
            return local_obs
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.shape != local_obs.shape[:2]:
            raise ValueError(
                f"Expected agent_mask shape {tuple(local_obs.shape[:2])}, got {tuple(agent_mask.shape)}"
            )
        return local_obs * agent_mask.to(dtype=local_obs.dtype).unsqueeze(-1)

    @staticmethod
    def _build_critic_local_obs(
            local_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_local_vars is None or hidden_local_vars.shape[-1] == 0:
            return local_obs
        if local_obs.shape[:2] != hidden_local_vars.shape[:2]:
            raise ValueError(
                "Expected local_obs and hidden_local_vars to match in first two dims, "
                f"got {tuple(local_obs.shape)} and {tuple(hidden_local_vars.shape)}"
            )
        return torch.cat((local_obs, hidden_local_vars), dim=-1)

    @staticmethod
    def _build_critic_global_obs(
            hidden_global_vars: torch.Tensor | None
    ) -> torch.Tensor | None:
        if hidden_global_vars is None or hidden_global_vars.shape[-1] == 0:
            return None
        return hidden_global_vars

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
            "shared_encoder": self._module_grad_norm(self.shared_encoder),
            "actor": self._module_grad_norm(self.actor),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return

        self._ensure_torch_compile_available()

        self.shared_encoder = self._compile_module(self.shared_encoder)
        self.actor = self._compile_module(self.actor)
        self.critic = self._compile_module(self.critic)

    def _compile_module(self, module: nn.Module) -> nn.Module:
        if isinstance(module, nn.Identity):
            return module
        return torch.compile(
            module,
            mode=self.config.compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    @staticmethod
    def _ensure_torch_compile_available() -> None:
        if not hasattr(torch, "compile"):
            raise RuntimeError("PPOPolicyConfig.compile_modules=True requires torch.compile support.")
        if sys.platform == "win32" and shutil.which("cl") is None:
            raise RuntimeError(
                "PPOPolicyConfig.compile_modules=True on this Windows setup requires cl.exe on PATH for torch.compile."
            )
        if not callable(torch.compile):
            raise RuntimeError("torch.compile is not callable in this environment.")

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
