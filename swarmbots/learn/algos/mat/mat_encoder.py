import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.nn_init import (
    make_init_linear_orthogonal,
    reinitialize_multihead_attention,
)
from swarmbots.learn.nn_components.feed_forward import (
    FeedForwardConfig,
    GLU,
    MLPConfig,
    StackedGLU,
    feedforward_linear_layers,
    make_feedforward,
)


@dataclass(frozen=True)
class MATEncoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: ActivationFactory = nn.GELU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = False
    linear_init_gain: float = 1.0
    linear_projection_init_gain: float | None = 1.0
    transformer_ff_init_gain: float | None = 1.0
    transformer_ff_config: FeedForwardConfig | None = None
    local_obs_encoder_config: FeedForwardConfig = field(default_factory=MLPConfig)
    global_obs_encoder_config: FeedForwardConfig = field(default_factory=MLPConfig)
    normalize_obs_inputs: bool = False
    normalize_tokens: bool = False
    use_agent_attention: bool = True


class MATEncoderLayer(nn.Module):

    def __init__(
            self,
            config: MATEncoderConfig,
    ) -> None:
        super().__init__()
        self.self_attn: nn.MultiheadAttention | None = (
            nn.MultiheadAttention(
                config.d_model,
                config.nhead,
                dropout=config.dropout,
                bias=config.bias,
                batch_first=True,
            )
            if config.use_agent_attention
            else None
        )
        self.norm_first = config.norm_first
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.feedforward = make_feedforward(
            input_dim=config.d_model,
            output_dim=config.d_model,
            config=resolve_transformer_ff_config(config),
            linear_init=_skip_init,
            output_linear_init=_skip_init,
            act_fn_cls=config.act_fn_cls,
            bias=config.bias,
            dropout=config.dropout,
        )

    @property
    def linear1(self) -> nn.Linear:
        return self._feedforward_linear_layers()[0]

    @property
    def linear2(self) -> nn.Linear:
        return self._feedforward_linear_layers()[-1]

    @property
    def activation(self) -> nn.Module:
        if isinstance(self.feedforward, GLU):
            return self.feedforward.activation
        if isinstance(self.feedforward, StackedGLU):
            first_layer = self.feedforward[0]
            if not isinstance(first_layer, GLU):
                raise TypeError(f"Unexpected StackedGLU child module: {type(first_layer).__name__}")
            return first_layer.activation
        for module in self.feedforward.modules():
            if module is self.feedforward:
                continue
            if not isinstance(module, (nn.Linear, nn.Dropout)):
                return module
        raise RuntimeError("MATEncoderLayer feedforward has no activation")

    def forward(
            self,
            src: torch.Tensor,
            src_mask: torch.Tensor | None = None,
            src_key_padding_mask: torch.Tensor | None = None,
            is_causal: bool = False,
    ) -> torch.Tensor:
        hidden = src
        if self.norm_first:
            hidden = hidden + self._self_attention_block(
                self.norm1(hidden),
                attention_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
            hidden = hidden + self._feedforward_block(self.norm2(hidden))
            return hidden

        hidden = self.norm1(
            hidden + self._self_attention_block(
                hidden,
                attention_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )
        )
        hidden = self.norm2(hidden + self._feedforward_block(hidden))
        return hidden

    def _self_attention_block(
            self,
            embeddings: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None,
            key_padding_mask: torch.Tensor | None,
            is_causal: bool,
    ) -> torch.Tensor:
        if self.self_attn is None:
            return torch.zeros_like(embeddings)
        attention_output = self.self_attn(
            embeddings,
            embeddings,
            embeddings,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(attention_output)

    def _feedforward_block(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.feedforward(embeddings))

    def _feedforward_linear_layers(self) -> list[nn.Linear]:
        hidden_layers, output_layers = feedforward_linear_layers(self.feedforward)
        return [*hidden_layers, *output_layers]


class MATEncoder(nn.Module):

    def __init__(
            self,
            config: MATEncoderConfig,
            *,
            max_agents: int,
            local_obs_dim: int,
            global_obs_dim: int,
    ) -> None:
        super().__init__()
        self.local_obs_dim: int = local_obs_dim
        self.global_obs_dim: int = global_obs_dim
        self.has_global_obs: bool = global_obs_dim > 0
        self.add_agent_embeddings = config.add_agent_embeddings
        self.max_agents = max_agents
        self.local_obs_input_norm = nn.LayerNorm(local_obs_dim) if config.normalize_obs_inputs else nn.Identity()
        self.global_obs_input_norm = (
            nn.LayerNorm(global_obs_dim)
            if config.normalize_obs_inputs and self.has_global_obs
            else nn.Identity()
        )
        self.token_norm = nn.LayerNorm(config.d_model) if config.normalize_tokens else nn.Identity()
        linear_init = make_init_linear_orthogonal(config.linear_init_gain)
        projection_linear_init = (
            linear_init
            if config.linear_projection_init_gain is None
            else make_init_linear_orthogonal(config.linear_projection_init_gain)
        )

        self.local_obs_encoder = make_feedforward(
            input_dim=self.local_obs_dim,
            output_dim=config.d_model,
            config=config.local_obs_encoder_config,
            linear_init=linear_init,
            output_linear_init=projection_linear_init,
            act_fn_cls=config.act_fn_cls,
        )

        if self.has_global_obs:
            self.global_obs_encoder = make_feedforward(
                input_dim=self.global_obs_dim,
                output_dim=config.d_model,
                config=config.global_obs_encoder_config,
                linear_init=linear_init,
                output_linear_init=projection_linear_init,
                act_fn_cls=config.act_fn_cls,
            )
        else:
            self.global_obs_encoder = None

        prototype_layer = MATEncoderLayer(config)
        self.layers = nn.ModuleList([
            copy.deepcopy(prototype_layer)
            for _ in range(config.num_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps, bias=config.bias)
        if config.transformer_ff_init_gain is not None:
            self._reinitialize_layers(feedforward_init_gain=config.transformer_ff_init_gain)

        self.agent_embeddings: nn.Parameter | None = None
        if self.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, self.max_agents, config.d_model), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_agents = local_obs.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_obs second dim <= {self.max_agents}, got {n_agents}")
        local_obs = self.local_obs_input_norm(local_obs)
        local_embeddings = self.local_obs_encoder(local_obs)
        if self.agent_embeddings is not None:
            local_embeddings = local_embeddings + self.agent_embeddings[:, :n_agents, :]

        if self.has_global_obs:
            global_obs = self.global_obs_input_norm(global_obs)
            global_embeddings = self.global_obs_encoder(global_obs)
            expanded_global_embeddings = global_embeddings.unsqueeze(1).expand(-1, n_agents, -1)
            local_embeddings = local_embeddings + expanded_global_embeddings
        local_embeddings = self.token_norm(local_embeddings)

        src_key_padding_mask = None
        if agent_mask is not None:
            src_key_padding_mask = ~agent_mask

        augmented_observations = local_embeddings
        for layer in self.layers:
            augmented_observations = layer(
                augmented_observations,
                src_key_padding_mask=src_key_padding_mask,
            )
        augmented_observations = self.norm(augmented_observations)
        return augmented_observations

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


def _skip_init(module: nn.Linear) -> nn.Linear:
    return module


def resolve_transformer_ff_config(config: MATEncoderConfig) -> FeedForwardConfig:
    if config.transformer_ff_config is None:
        return MLPConfig(hidden_dims=[config.dim_feedforward])
    return config.transformer_ff_config
