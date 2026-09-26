from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal, make_init_linear_orthogonal


@dataclass(frozen=True)
class MAPPOActorConfig:
    hidden_dims: list[int] = field(default_factory=list)
    shared_encoder_latent_dim: int | None = None
    actor_head_hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim: int = 64
    act_fun_class: type[nn.Module] = nn.Tanh
    actor_head_init_gain: float = 0.01


class MAPPOSharedEncoder(nn.Module):

    def __init__(
            self,
            n_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
            config: MAPPOActorConfig,
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
            config.latent_pi_dim if config.shared_encoder_latent_dim is None else config.shared_encoder_latent_dim
        )
        self.act_fun_class = encoder_act_fun_class

        self.mlp = MLP(
            input_dim=local_obs_dim + global_obs_dim,
            hidden_dims=[*config.hidden_dims, self.local_latent_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=encoder_act_fun_class
        )

    def forward(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        encoder_input = local_obs
        if self.has_global_obs:
            expanded_global_obs = global_obs.unsqueeze(1).expand(-1, self.n_agents, -1)
            encoder_input = torch.cat((encoder_input, expanded_global_obs), dim=-1)

        return self.mlp(encoder_input)


class MAPPOActor(nn.Module):

    def __init__(
            self,
            local_latent_dim: int,
            config: MAPPOActorConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ):
        super().__init__()
        actor_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.local_latent_dim = local_latent_dim
        self.latent_pi_dim = config.latent_pi_dim
        self.hidden_dims = list(config.actor_head_hidden_dims)
        self.act_fun_class = actor_act_fun_class

        self.mlp = MLP(
            input_dim=local_latent_dim,
            hidden_dims=[*config.actor_head_hidden_dims, config.latent_pi_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            final_linear_init=make_init_linear_orthogonal(config.actor_head_init_gain),
            act_fn_cls=actor_act_fun_class,
        )

    def forward(self, local_latents: torch.Tensor) -> torch.Tensor:
        return self.mlp(local_latents)
