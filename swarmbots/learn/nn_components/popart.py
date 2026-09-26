from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PopArtLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int = 1,
            *,
            bias: bool = True,
            beta: float = 3e-4,
            eps: float = 1e-5,
            min_std: float = 1e-4,
            init_mu: float = 0.0,
            init_sigma: float = 1.0,
            init_gain: float = 1.0,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.beta = float(beta)
        self.eps = float(eps)
        self.min_std = float(min_std)
        self.init_gain = float(init_gain)

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((self.out_features,), **factory_kwargs))
        else:
            self.bias = None

        self.register_buffer("mu", torch.full((self.out_features,), float(init_mu), **factory_kwargs))
        init_sigma_t = torch.full((self.out_features,), float(init_sigma), **factory_kwargs)
        init_nu = self.mu.square() + init_sigma_t.square()
        self.register_buffer("nu", init_nu)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight, gain=self.init_gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_linear(
            cls,
            linear: nn.Linear,
            *,
            beta: float = 3e-4,
            eps: float = 1e-5,
            min_std: float = 1e-4,
            init_mu: float = 0.0,
            init_sigma: float = 1.0,
            init_gain: float = 1.0,
    ) -> PopArtLinear:
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            beta=beta,
            eps=eps,
            min_std=min_std,
            init_mu=init_mu,
            init_sigma=init_sigma,
            init_gain=init_gain,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(linear.weight)
            if linear.bias is not None:
                assert layer.bias is not None
                layer.bias.copy_(linear.bias)
        return layer

    def _sigma_from(self, mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        var = nu - mu.square()
        sigma = torch.sqrt(torch.clamp(var, min=self.eps))
        return torch.clamp(sigma, min=self.min_std)

    def sigma(self) -> torch.Tensor:
        return self._sigma_from(self.mu, self.nu)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mu) / self.sigma()

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.sigma() + self.mu

    @torch.no_grad()
    def update(self, targets: torch.Tensor) -> None:
        if targets.numel() == 0:
            return

        y = targets.to(device=self.mu.device, dtype=self.mu.dtype)
        if y.ndim == 0:
            y = y.unsqueeze(0)
        if self.out_features == 1:
            y = y.reshape(-1)
        else:
            y = y.reshape(-1, self.out_features)

        batch_mean = y.mean(dim=0)
        batch_second_moment = y.square().mean(dim=0)

        old_mu = self.mu.clone()
        old_nu = self.nu.clone()
        old_sigma = self._sigma_from(old_mu, old_nu)

        new_mu = (1.0 - self.beta) * old_mu + self.beta * batch_mean
        new_nu = (1.0 - self.beta) * old_nu + self.beta * batch_second_moment
        new_sigma = self._sigma_from(new_mu, new_nu)

        scale = old_sigma / new_sigma
        self.weight.mul_(scale.unsqueeze(-1))
        if self.bias is not None:
            self.bias.copy_((old_sigma * self.bias + old_mu - new_mu) / new_sigma)

        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_norm = F.linear(x, self.weight, self.bias)
        return self.denormalize(y_norm)
