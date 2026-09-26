from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import (
    DEFAULT_ORTHOGONAL_GAIN,
    LinearInitialization,
    init_linear_orthogonal,
    make_init_linear_orthogonal,
)
from swarmbots.learn.nn_components.popart import PopArtLinear

PoolMode = Literal["mean", "sum", "max"]


def _pool_over_set(
    x: torch.Tensor,
    *,
    set_dim: int,
    mode: PoolMode,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is None:
        if mode == "mean":
            return x.mean(dim=set_dim)
        if mode == "sum":
            return x.sum(dim=set_dim)
        if mode == "max":
            return x.max(dim=set_dim).values
        raise ValueError(f"Unknown pool mode: {mode}")

    if mask.dtype != torch.bool:
        raise ValueError(f"Expected mask dtype bool, got {mask.dtype}")
    if mask.ndim != x.ndim - 1:
        raise ValueError(f"Expected mask ndim {x.ndim - 1}, got {mask.ndim}")
    if mask.shape != x.shape[:-1]:
        raise ValueError(f"Expected mask shape {x.shape[:-1]}, got {mask.shape}")

    mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
    if mode == "sum":
        return (x * mask_f).sum(dim=set_dim)
    if mode == "mean":
        denom = mask_f.sum(dim=set_dim).clamp_min(1.0)
        return (x * mask_f).sum(dim=set_dim) / denom
    if mode == "max":
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(~mask.unsqueeze(-1), neg_inf)
        pooled = x_masked.max(dim=set_dim).values
        any_valid = mask.any(dim=set_dim)
        return torch.where(any_valid.unsqueeze(-1), pooled, torch.zeros_like(pooled))
    raise ValueError(f"Unknown pool mode: {mode}")


class DeepSet(nn.Module):
    def __init__(
        self,
        *,
        element_encoder: nn.Module,
        set_decoder: nn.Module,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        context_features: int = 0,
        context_in_elements: bool = False,
    ) -> None:
        super().__init__()
        self.element_encoder = element_encoder
        self.set_decoder = set_decoder
        self.set_dim = int(set_dim)
        self.pool_mode: PoolMode = pool_mode
        self.context_features = int(context_features)
        self.context_in_elements = bool(context_in_elements)

    def forward(
        self,
        elements: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        element_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.context_in_elements:
            expanded_context = context.unsqueeze(self.set_dim).expand(*elements.shape[:-1], context.shape[-1])
            elements = torch.cat((elements, expanded_context), dim=-1)

        encoded = self.element_encoder(elements)
        pooled = _pool_over_set(encoded, set_dim=self.set_dim, mode=self.pool_mode, mask=element_mask)

        if self.context_features > 0 and not self.context_in_elements:
            pooled = torch.cat((pooled, context), dim=-1)

        return self.set_decoder(pooled)


class DeepSetCriticHiddenDims(TypedDict):
    local_projection_hidden_dims: list[int]
    value_regressor_hidden_dims: list[int]


@dataclass(frozen=True)
class DeepSetCriticConfig:
    local_projection_hidden_dims: list[int] = field(default_factory=list)
    value_regressor_hidden_dims: list[int] = field(default_factory=list)


class DeepSetCritic(nn.Module):
    """
    Convenience wrapper for a scalar-value DeepSet critic.
    """

    def __init__(
        self,
        *,
        num_local_features: int,
        local_projection_hidden_dims: list[int],
        value_regressor_hidden_dims: list[int],
        num_global_features: int = 0,
        set_dim: int = 1,
        pool_mode: PoolMode = "mean",
        linear_init: LinearInitialization = init_linear_orthogonal,
        local_projection_linear_init_gain: float = DEFAULT_ORTHOGONAL_GAIN,
        value_regressor_linear_init_gain: float = DEFAULT_ORTHOGONAL_GAIN,
        value_head_linear_init_gain: float = 0.01,
        act_fn_cls: ActivationFactory = nn.Tanh,
        context_in_elements: bool = False,
        use_popart: bool = False,
        popart_beta: float = 3e-4,
        popart_eps: float = 1e-5,
        popart_min_std: float = 1e-4,
        popart_init_sigma: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_local_features = int(num_local_features)
        self.num_global_features = int(num_global_features)
        self.context_in_elements = bool(context_in_elements)
        self.use_popart = bool(use_popart)

        if self.context_in_elements and self.num_global_features == 0:
            raise ValueError("context_in_elements requires num_global_features > 0")

        if linear_init is init_linear_orthogonal:
            local_projection_linear_init = make_init_linear_orthogonal(local_projection_linear_init_gain)
            value_regressor_linear_init = make_init_linear_orthogonal(value_regressor_linear_init_gain)
            value_head_linear_init = make_init_linear_orthogonal(value_head_linear_init_gain)
        else:
            local_projection_linear_init = linear_init
            value_regressor_linear_init = linear_init
            value_head_linear_init = linear_init

        element_input_features = self.num_local_features + (self.num_global_features if self.context_in_elements else 0)
        context_features_after_pool = 0 if self.context_in_elements else self.num_global_features

        if local_projection_hidden_dims:
            element_encoder: nn.Module = MLP(
                input_dim=element_input_features,
                hidden_dims=local_projection_hidden_dims,
                end_with_act_fn=True,
                linear_init=local_projection_linear_init,
                act_fn_cls=act_fn_cls,
            )
            pooled_dim = int(local_projection_hidden_dims[-1])
        else:
            element_encoder = nn.Identity()
            pooled_dim = element_input_features

        value_regressor_input_dim = pooled_dim + context_features_after_pool
        if value_regressor_hidden_dims:
            value_regressor_trunk: nn.Module = MLP(
                input_dim=value_regressor_input_dim,
                hidden_dims=value_regressor_hidden_dims,
                end_with_act_fn=True,
                linear_init=value_regressor_linear_init,
                act_fn_cls=act_fn_cls,
            )
            value_regressor_head_input_dim = int(value_regressor_hidden_dims[-1])
        else:
            value_regressor_trunk = nn.Identity()
            value_regressor_head_input_dim = value_regressor_input_dim

        if self.use_popart:
            self.popart_head: PopArtLinear | None = PopArtLinear(
                in_features=value_regressor_head_input_dim,
                out_features=1,
                beta=popart_beta,
                eps=popart_eps,
                min_std=popart_min_std,
                init_sigma=popart_init_sigma,
                init_gain=value_head_linear_init_gain,
            )
            value_regressor_head: nn.Module = self.popart_head
        else:
            self.popart_head = None
            value_regressor_head = nn.Linear(value_regressor_head_input_dim, 1)
            value_head_linear_init(value_regressor_head)

        value_regressor = nn.Sequential(value_regressor_trunk, value_regressor_head)

        self.deepset = DeepSet(
            element_encoder=element_encoder,
            set_decoder=value_regressor,
            set_dim=set_dim,
            pool_mode=pool_mode,
            context_features=context_features_after_pool,
            context_in_elements=self.context_in_elements,
        )

    def forward(
        self,
        local_features: torch.Tensor,
        global_features: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = global_features if self.num_global_features > 0 else None
        return self.deepset(local_features, context=context, element_mask=agent_mask).squeeze(dim=-1)

    @property
    def has_popart(self) -> bool:
        return self.popart_head is not None

    def update_popart(self, targets: torch.Tensor) -> None:
        if self.popart_head is None:
            raise RuntimeError("PopArt is not enabled for this DeepSetCritic")
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
