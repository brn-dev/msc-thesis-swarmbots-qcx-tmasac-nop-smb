from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import (
    make_init_linear_orthogonal,
    reinitialize_transformer_stack,
)


@dataclass(frozen=True)
class TransformerTransitionModelConfig:
    n_agents: int
    latent_dim: int
    action_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.0
    act_fn_cls: ActivationFactory = nn.GELU
    add_agent_embeddings: bool = False
    predict_delta: bool = True
    coembed_mlp_hidden_dims: list[int] | None = None
    head_mlp_hidden_dims: list[int] | None = None
    coembed_init_gain: float = 1.0
    head_init_gain: float = 0.01
    transformer_ff_init_gain: float | None = 1.0
    norm_first: bool = True
    layer_norm_eps: float = 1e-5

class TransformerTransitionModel(nn.Module):
    """
    Multi-agent transition model predicting next local latents z_{t+1} from (z_t, a_t).

    Treats each agent as a token. Co-embeds (z_t, a_t) per agent, then runs self-attention
    across agents via a Transformer encoder.
    """

    def __init__(
        self,
        config: TransformerTransitionModelConfig,
    ) -> None:
        super().__init__()
        self.n_agents = config.n_agents
        self.latent_dim = config.latent_dim
        self.action_dim = config.action_dim
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.dim_feedforward = config.dim_feedforward
        self.dropout = config.dropout
        self.act_fn_cls = config.act_fn_cls
        self.add_agent_embeddings = config.add_agent_embeddings
        self.predict_delta = config.predict_delta
        self.coembed_mlp_hidden_dims = config.coembed_mlp_hidden_dims
        self.head_mlp_hidden_dims = config.head_mlp_hidden_dims
        self.coembed_init_gain = config.coembed_init_gain
        self.head_init_gain = config.head_init_gain
        self.transformer_ff_init_gain = config.transformer_ff_init_gain
        self.norm_first = config.norm_first
        self.layer_norm_eps = config.layer_norm_eps
        self.enable_nested_tensor = not config.norm_first

        in_dim = config.latent_dim + config.action_dim
        coembed_linear_init = make_init_linear_orthogonal(config.coembed_init_gain)
        if config.coembed_mlp_hidden_dims is None:
            self.coembed: nn.Module = nn.Linear(in_dim, config.d_model)
            coembed_linear_init(self.coembed)
        else:
            self.coembed = MLP(
                input_dim=in_dim,
                hidden_dims=[*config.coembed_mlp_hidden_dims, config.d_model],
                end_with_act_fn=True,
                linear_init=coembed_linear_init,
                act_fn_cls=config.act_fn_cls,
            )

        self.agent_embeddings: nn.Parameter | None = None
        if config.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(torch.zeros(1, config.n_agents, config.d_model), requires_grad=True)
            nn.init.orthogonal_(self.agent_embeddings)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=make_activation(config.act_fn_cls, num_features=config.dim_feedforward),
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
            norm_first=config.norm_first,
            bias=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
            enable_nested_tensor=self.enable_nested_tensor,
        )
        if config.transformer_ff_init_gain is not None:
            reinitialize_transformer_stack(
                self.encoder,
                feedforward_init_gain=config.transformer_ff_init_gain,
            )

        head_linear_init = make_init_linear_orthogonal(config.head_init_gain)
        if config.head_mlp_hidden_dims is None:
            self.head: nn.Module = nn.Linear(config.d_model, config.latent_dim)
            head_linear_init(self.head)
        else:
            self.head = MLP(
                input_dim=config.d_model,
                hidden_dims=[*config.head_mlp_hidden_dims, config.latent_dim],
                end_with_act_fn=False,
                linear_init=head_linear_init,
                act_fn_cls=config.act_fn_cls,
            )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "act_fn_cls": str(self.act_fn_cls),
            "add_agent_embeddings": self.add_agent_embeddings,
            "predict_delta": self.predict_delta,
            "coembed_mlp_hidden_dims": self.coembed_mlp_hidden_dims,
            "head_mlp_hidden_dims": self.head_mlp_hidden_dims,
            "coembed_init_gain": self.coembed_init_gain,
            "head_init_gain": self.head_init_gain,
            "transformer_ff_init_gain": self.transformer_ff_init_gain,
            "norm_first": self.norm_first,
            "layer_norm_eps": self.layer_norm_eps,
            "enable_nested_tensor": self.enable_nested_tensor,
        }

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor, agent_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z_t: (batch, n_agents, latent_dim)
            a_t: (batch, n_agents, action_dim)
            agent_mask: optional bool mask (batch, n_agents) where True means "valid / present".

        Returns:
            z_{t+1} prediction: (batch, n_agents, latent_dim)
        """
        src = torch.cat([z_t, a_t], dim=-1)
        x = self.coembed(src)
        if self.agent_embeddings is not None:
            x = x + self.agent_embeddings

        src_key_padding_mask = None
        if agent_mask is not None:
            if agent_mask.ndim != 2 or agent_mask.shape != (z_t.shape[0], self.n_agents):
                raise ValueError(
                    f"Expected agent_mask shape (B, N)=({z_t.shape[0]}, {self.n_agents}), got {tuple(agent_mask.shape)}"
                )
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            src_key_padding_mask = ~agent_mask

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        pred = self.head(h)
        if self.predict_delta:
            return z_t + pred
        return pred

    def predict_n_steps(
        self,
        z_0: torch.Tensor,
        action_seq: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Predict multiple steps into the future by iteratively feeding predictions back in.

        Args:
            z_0: (batch, n_agents, latent_dim)
            action_seq: (batch, steps, n_agents, action_dim)
            agent_mask:
                Optional bool mask either shaped (batch, n_agents) (static across steps) or
                (batch, steps, n_agents) (time-varying), where True means "valid / present".

        Returns:
            z_preds: (batch, steps, n_agents, latent_dim) where z_preds[:, t] predicts z_{t+1}.
        """
        steps = action_seq.shape[1]
        if steps == 0:
            return z_0.new_empty((z_0.shape[0], 0, self.n_agents, self.latent_dim))

        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.ndim == 2:
                if agent_mask.shape != (z_0.shape[0], self.n_agents):
                    raise ValueError(
                        f"Expected agent_mask shape (B, N)=({z_0.shape[0]}, {self.n_agents}), got {tuple(agent_mask.shape)}"
                    )
            elif agent_mask.ndim == 3:
                if agent_mask.shape != (z_0.shape[0], steps, self.n_agents):
                    raise ValueError(
                        f"Expected agent_mask shape (B, T, N)=({z_0.shape[0]}, {steps}, {self.n_agents}), got {tuple(agent_mask.shape)}"
                    )
            else:
                raise ValueError(f"Expected agent_mask ndim 2 or 3, got {agent_mask.ndim}")

        z_t = z_0
        z_preds: list[torch.Tensor] = []
        for t in range(steps):
            mask_t = agent_mask[:, t] if (agent_mask is not None and agent_mask.ndim == 3) else agent_mask
            z_t = self(z_t, action_seq[:, t], agent_mask=mask_t)
            z_preds.append(z_t)

        return torch.stack(z_preds, dim=1)


