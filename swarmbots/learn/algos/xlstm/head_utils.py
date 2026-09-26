import math

import torch
from torch import nn
from torch.nn import functional as F


def require_heads_divide_hidden_dim(*, hidden_dim: int, num_heads: int) -> None:
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")


def init_linspace_bias_(bias: torch.Tensor, *, start: float, end: float) -> torch.Tensor:
    values = torch.linspace(start, end, bias.numel(), device=bias.device, dtype=bias.dtype)
    with torch.no_grad():
        bias.copy_(values)
    return bias


class HeadwiseLinearProjection(nn.Module):

    def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            num_projections: int,
            bias: bool,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if num_projections < 1:
            raise ValueError(f"num_projections must be >= 1, got {num_projections}")
        require_heads_divide_hidden_dim(hidden_dim=hidden_dim, num_heads=num_heads)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_projections = num_projections
        self.head_dim = hidden_dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_projections, num_heads, self.head_dim, self.head_dim))
        self.bias = nn.Parameter(torch.empty(num_projections, num_heads, self.head_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * self.hidden_dim)))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        input_heads = inputs.reshape(*inputs.shape[:-1], self.num_heads, self.head_dim)
        projected = torch.einsum("...ni,pnoi->...pno", input_heads, self.weight)
        if self.bias is not None:
            projected = projected + self.bias
        return projected.reshape(*inputs.shape[:-1], self.num_projections * self.hidden_dim)


class MultiHeadLayerNorm(nn.Module):

    def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            eps: float = 1e-5,
            bias: bool = False,
            residual_weight: bool = False,
    ) -> None:
        super().__init__()
        require_heads_divide_hidden_dim(hidden_dim=hidden_dim, num_heads=num_heads)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.eps = eps
        self.residual_weight = residual_weight
        self.weight = nn.Parameter(torch.empty(hidden_dim))
        self.bias = nn.Parameter(torch.empty(hidden_dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.residual_weight:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def effective_weight(self) -> torch.Tensor:
        return 1.0 + self.weight if self.residual_weight else self.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs shape (B, NH, T, DH), got {tuple(inputs.shape)}")
        batch_size, num_heads, sequence_length, head_dim = inputs.shape
        normalized_inputs = inputs.transpose(1, 2).reshape(batch_size * sequence_length, num_heads * head_dim)
        outputs = F.group_norm(
            normalized_inputs,
            num_groups=num_heads,
            weight=self.effective_weight,
            bias=self.bias,
            eps=self.eps,
        )
        return outputs.reshape(batch_size, sequence_length, num_heads, head_dim).transpose(1, 2)
