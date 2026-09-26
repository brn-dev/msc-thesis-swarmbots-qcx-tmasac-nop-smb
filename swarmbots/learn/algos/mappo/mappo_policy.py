from dataclasses import dataclass, field
from typing import Any
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfigInput,
    continuous_config_to_dicts,
    bernoulli_config_to_dict,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor, MAPPOActorConfig, MAPPOSharedEncoder
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import PPOCritic, PPOPolicy, PPOCriticConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.deep_set import DeepSetCritic, DeepSetCriticConfig
from swarmbots.learn.serialization_utils import serialize_dataclass

MAPPOCritic = PPOCritic


@dataclass(frozen=True)
class MAPPOCriticConfig:
    mlp_hidden_dims: list[int] | None = field(default_factory=list)
    deep_set_config: DeepSetCriticConfig | None = None
    act_fun_class: type[nn.Module] = nn.Tanh
    use_popart: bool = False
    local_projection_init_gain: float = 1.0
    value_regressor_init_gain: float = 1.0
    value_head_init_gain: float = 0.01


@dataclass(frozen=True)
class MAPPOPolicyConfig:
    actor_config: MAPPOActorConfig = field(default_factory=MAPPOActorConfig)
    critic_config: MAPPOCriticConfig = field(default_factory=MAPPOCriticConfig)
    continuous_config: ContinuousActionDistConfigInput = None
    bernoulli_config: BernoulliConfig | None = None
    compile_modules: bool = False
    compile_mode: str = "default"


class MAPPOPolicy(PPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MAPPOPolicyConfig = MAPPOPolicyConfig(),
    ):
        BasePolicy.__init__(self)
        self.config = config

        self.n_agents = env.n_agents
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.latent_pi_dim = config.actor_config.latent_pi_dim
        self.local_latent_dim = (
            config.actor_config.latent_pi_dim
            if config.actor_config.shared_encoder_latent_dim is None
            else config.actor_config.shared_encoder_latent_dim
        )

        self.shared_encoder = MAPPOSharedEncoder(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            config=config.actor_config,
        )
        self.actor = MAPPOActor(
            local_latent_dim=self.local_latent_dim,
            config=config.actor_config,
        )
        self.action_dist = HybridActionDistribution(
            latent_dim=config.actor_config.latent_pi_dim,
            action_space=env.action_space,
            continuous_config=config.continuous_config,
            bernoulli_config=config.bernoulli_config,
        )

        if config.critic_config.deep_set_config is None:
            self.critic = MAPPOCritic(
                n_agents=env.n_agents,
                local_obs_dim=self.local_latent_dim + self.hidden_local_vars_dim,
                global_obs_dim=self.hidden_global_vars_dim,
                config=PPOCriticConfig(
                    hidden_dims=config.critic_config.mlp_hidden_dims or [],
                    act_fun_class=config.critic_config.act_fun_class,
                    use_popart=config.critic_config.use_popart,
                    value_head_init_gain=config.critic_config.value_head_init_gain,
                ),
            )
        else:
            self.critic = DeepSetCritic(
                num_local_features=self.local_latent_dim + self.hidden_local_vars_dim,
                local_projection_hidden_dims=config.critic_config.deep_set_config.local_projection_hidden_dims,
                value_regressor_hidden_dims=config.critic_config.deep_set_config.value_regressor_hidden_dims,
                num_global_features=self.hidden_global_vars_dim,
                set_dim=AGENTS_DIM,
                pool_mode="mean",
                use_popart=config.critic_config.use_popart,
                local_projection_linear_init_gain=config.critic_config.local_projection_init_gain,
                value_regressor_linear_init_gain=config.critic_config.value_regressor_init_gain,
                value_head_linear_init_gain=config.critic_config.value_head_init_gain,
                act_fn_cls=config.critic_config.act_fun_class,
            )
        self._apply_optional_compile()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "mappo_policy_config": {
                "actor_config": serialize_dataclass(self.config.actor_config),
                "critic_config": serialize_dataclass(self.config.critic_config),
                "continuous_config": continuous_config_to_dicts(self.action_dist.continuous_configs),
                "bernoulli_config": bernoulli_config_to_dict(self.action_dist.bernoulli_config),
                "compile_modules": self.config.compile_modules,
                "compile_mode": self.config.compile_mode,
            }
        }
