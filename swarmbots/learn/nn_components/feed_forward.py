from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


class MLP(nn.Sequential):

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            end_with_act_fn: bool,
            start_with_act_fn: bool = False,
            linear_init: LinearInitialization = init_linear_orthogonal,
            final_linear_init: LinearInitialization | None = None,
            act_fn_cls: ActivationFactory = nn.Tanh,
            bias: bool = True,
            dropout: float = 0.0,
    ) -> None:
        assert len(hidden_dims) > 0

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        dims = [input_dim, *hidden_dims]
        n_layers = len(dims) - 1

        modules: list[nn.Module] = []

        if start_with_act_fn:
            modules.append(make_activation(act_fn_cls, num_features=input_dim))
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))

        for i in range(n_layers):
            linear = nn.Linear(dims[i], dims[i + 1], bias=bias)
            is_final_layer = i == n_layers - 1
            if is_final_layer and final_linear_init is not None:
                final_linear_init(linear)
            else:
                linear_init(linear)
            modules.append(linear)

            if i < n_layers - 1 or end_with_act_fn:
                modules.append(make_activation(act_fn_cls, num_features=dims[i + 1]))
                if dropout > 0.0:
                    modules.append(nn.Dropout(dropout))

        super().__init__(*modules)


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=list)


class GLU(nn.Module):
    activation_factory: ActivationFactory = nn.Sigmoid

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            *,
            linear_init: LinearInitialization = init_linear_orthogonal,
            output_linear_init: LinearInitialization = init_linear_orthogonal,
            bias: bool = True,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.value_projection = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.output_projection = nn.Linear(hidden_dim, output_dim, bias=bias)
        linear_init(self.gate_projection)
        linear_init(self.value_projection)
        output_linear_init(self.output_projection)
        self.activation = make_activation(self.activation_factory, num_features=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gated_values = self.activation(self.gate_projection(inputs)) * self.value_projection(inputs)
        return self.output_projection(self.dropout(gated_values))


class BilinearGLU(GLU):
    activation_factory = nn.Identity


class ReGLU(GLU):
    activation_factory = nn.ReLU


class GEGLU(GLU):
    activation_factory = nn.GELU


class SwiGLU(GLU):
    activation_factory = nn.SiLU


NormalizationFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class GLUStackConfig:
    n_layers: int
    pre_norm: NormalizationFactory | None = None
    norm_first_layer: bool = True
    residual: bool = False
    residual_first_layer: bool = True

    def __post_init__(self) -> None:
        if self.n_layers <= 1:
            raise ValueError(f"Stacked GLUs require n_layers > 1, got {self.n_layers}")


class StackedGLU(nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            *,
            glu_cls: type[GLU],
            n_layers: int,
            pre_norm: NormalizationFactory | None = None,
            norm_first_layer: bool = True,
            residual: bool = False,
            residual_first_layer: bool = True,
            linear_init: LinearInitialization = init_linear_orthogonal,
            output_linear_init: LinearInitialization = init_linear_orthogonal,
            bias: bool = True,
            dropout: float = 0.0,
    ) -> None:
        if n_layers <= 1:
            raise ValueError(f"Stacked GLUs require n_layers > 1, got {n_layers}")
        super().__init__()
        self.layers = nn.ModuleList([
            glu_cls(
                input_dim=input_dim if layer_idx == 0 else output_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                linear_init=linear_init,
                output_linear_init=output_linear_init,
                bias=bias,
                dropout=dropout,
            )
            for layer_idx in range(n_layers)
        ])
        self.pre_norms = nn.ModuleList([
            (
                nn.Identity()
                if pre_norm is None or (layer_idx == 0 and not norm_first_layer)
                else pre_norm(input_dim if layer_idx == 0 else output_dim)
            )
            for layer_idx in range(n_layers)
        ])
        self.residual = residual
        self.residual_first_layer = residual_first_layer

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[GLU]:
        return iter(self.layers)

    def __getitem__(self, index: int) -> GLU:
        return self.layers[index]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for layer_idx, (pre_norm, layer) in enumerate(zip(self.pre_norms, self, strict=True)):
            layer_output = layer(pre_norm(hidden))
            use_residual = self.residual and (layer_idx > 0 or self.residual_first_layer)
            if use_residual and layer.gate_projection.in_features == layer.output_projection.out_features:
                layer_output = hidden + layer_output
            hidden = layer_output
        return hidden


@dataclass(frozen=True)
class GLUConfig:
    hidden_dim: int
    stacked: GLUStackConfig | None = None
    glu_cls: ClassVar[type[GLU]] = GLU


@dataclass(frozen=True)
class BilinearGLUConfig(GLUConfig):
    glu_cls: ClassVar[type[GLU]] = BilinearGLU


@dataclass(frozen=True)
class ReGLUConfig(GLUConfig):
    glu_cls: ClassVar[type[GLU]] = ReGLU


@dataclass(frozen=True)
class GEGLUConfig(GLUConfig):
    glu_cls: ClassVar[type[GLU]] = GEGLU


@dataclass(frozen=True)
class SwiGLUConfig(GLUConfig):
    glu_cls: ClassVar[type[GLU]] = SwiGLU


FeedForwardConfig = MLPConfig | GLUConfig


def make_feedforward(
        *,
        input_dim: int,
        output_dim: int,
        config: FeedForwardConfig,
        linear_init: LinearInitialization = init_linear_orthogonal,
        output_linear_init: LinearInitialization = init_linear_orthogonal,
        act_fn_cls: ActivationFactory = nn.Tanh,
        bias: bool = True,
        dropout: float = 0.0,
) -> nn.Module:
    if isinstance(config, MLPConfig):
        if not config.hidden_dims:
            output_projection = nn.Linear(input_dim, output_dim, bias=bias)
            output_linear_init(output_projection)
            return output_projection
        return MLP(
            input_dim=input_dim,
            hidden_dims=[*config.hidden_dims, output_dim],
            end_with_act_fn=False,
            linear_init=linear_init,
            final_linear_init=output_linear_init,
            act_fn_cls=act_fn_cls,
            bias=bias,
            dropout=dropout,
        )
    if isinstance(config, GLUConfig):
        glu_kwargs = dict(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            linear_init=linear_init,
            output_linear_init=output_linear_init,
            bias=bias,
            dropout=dropout,
        )
        if config.stacked is None:
            return config.glu_cls(**glu_kwargs)
        return StackedGLU(
            glu_cls=config.glu_cls,
            n_layers=config.stacked.n_layers,
            pre_norm=config.stacked.pre_norm,
            norm_first_layer=config.stacked.norm_first_layer,
            residual=config.stacked.residual,
            residual_first_layer=config.stacked.residual_first_layer,
            **glu_kwargs,
        )
    raise TypeError(f"Unsupported feed-forward config: {type(config).__name__}")


def feedforward_linear_layers(module: nn.Module) -> tuple[list[nn.Linear], list[nn.Linear]]:
    if isinstance(module, nn.Linear):
        return [], [module]
    if isinstance(module, MLP):
        linear_layers = [child for child in module if isinstance(child, nn.Linear)]
        return linear_layers[:-1], linear_layers[-1:]
    if isinstance(module, GLU):
        return [module.gate_projection, module.value_projection], [module.output_projection]
    if isinstance(module, StackedGLU):
        hidden_layers: list[nn.Linear] = []
        output_layers: list[nn.Linear] = []
        for layer in module:
            if not isinstance(layer, GLU):
                raise TypeError(f"Unexpected StackedGLU child module: {type(layer).__name__}")
            hidden_layers.extend((layer.gate_projection, layer.value_projection))
            output_layers.append(layer.output_projection)
        return hidden_layers, output_layers
    raise TypeError(f"Unsupported feed-forward module: {type(module).__name__}")
