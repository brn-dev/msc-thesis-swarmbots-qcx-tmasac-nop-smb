from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inverse_softplus(x: float) -> float:
    # softplus(y) = log(1 + exp(y)) => y = log(exp(x) - 1)
    return float(torch.log(torch.expm1(torch.tensor(x))).item())


class PELU(nn.Module):
    """
    Parametric Exponential Linear Unit (PELU).

    Paper: https://arxiv.org/pdf/1605.09332

    For input x:
    - x >= 0: (a / b) * x
    - x <  0: a * (exp(x / b) - 1)

    Parameters a, b are constrained to be positive via softplus.
    """

    def __init__(
        self,
        *,
        per_feature: bool = False,
        num_features: int | None = None,
        feature_dim: int = 1,
        init_a: float = 1.0,
        init_b: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if per_feature and num_features is None:
            raise ValueError("When per_feature=True, num_features must be set.")
        if init_a <= 0 or init_b <= 0:
            raise ValueError("init_a and init_b must be > 0.")
        if eps <= 0:
            raise ValueError("eps must be > 0.")

        self.per_feature: Final[bool] = per_feature
        self.num_features: Final[int | None] = num_features
        self.feature_dim: Final[int] = feature_dim
        self.eps: Final[float] = float(eps)

        shape = (num_features,) if per_feature else ()
        self._raw_a = nn.Parameter(torch.full(shape, _inverse_softplus(init_a)))
        self._raw_b = nn.Parameter(torch.full(shape, _inverse_softplus(init_b)))

    def extra_repr(self) -> str:
        return (
            f"per_feature={self.per_feature}, num_features={self.num_features}, "
            f"feature_dim={self.feature_dim}, eps={self.eps:g}"
        )

    def _broadcast_param(self, param: Tensor, x: Tensor) -> Tensor:
        if not self.per_feature:
            return param

        assert self.num_features is not None

        feature_dim = self.feature_dim
        if feature_dim < 0:
            feature_dim = x.dim() + feature_dim

        if x.dim() == 0:
            raise ValueError("PELU expects a tensor with at least 1 dimension.")
        if not (0 <= feature_dim < x.dim()):
            raise ValueError(f"Invalid feature_dim={self.feature_dim} for x.dim()={x.dim()}.")
        if x.size(feature_dim) != self.num_features:
            raise ValueError(
                f"x.size(feature_dim)={x.size(feature_dim)} does not match num_features={self.num_features}."
            )

        view_shape = [1] * x.dim()
        view_shape[feature_dim] = self.num_features
        return param.view(*view_shape)

    def forward(self, x: Tensor) -> Tensor:
        a = F.softplus(self._raw_a) + self.eps
        b = F.softplus(self._raw_b) + self.eps

        a = self._broadcast_param(a, x)
        b = self._broadcast_param(b, x)

        x_pos = x.clamp_min(0)
        x_neg = x.clamp_max(0)

        pos = (a / b) * x_pos
        neg = a * torch.expm1(x_neg / b)
        return pos + neg