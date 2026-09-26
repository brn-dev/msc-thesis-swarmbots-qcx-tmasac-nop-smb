import math
from dataclasses import dataclass
from typing import Optional, Self, Any

import torch
from torch import distributions as torchdist, nn
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionDist,
    ActionNetInitialization,
    ActionMetricsSplitterInput,
    compute_action_metrics,
    validate_probability_clamp_epsilon,
)


@dataclass(frozen=True)
class BetaMixtureConfig:
    num_components: int
    alphas: tuple[float, ...]
    betas: tuple[float, ...]
    epsilon: float = 1e-6


class BetaMixtureActionDist(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            num_components: int,
            action_net_initialization: ActionNetInitialization | None,
            epsilon: float = 1e-6,
            alphas: tuple[float, ...] | None = None,
            betas: tuple[float, ...] | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        if num_components < 2:
            raise ValueError(f"num_components must be >= 2, got {num_components}.")
        validate_probability_clamp_epsilon(epsilon)

        self.num_components = num_components
        self.epsilon = epsilon

        if alphas is None:
            alphas = tuple(1.0 + math.log(2.0) for _ in range(num_components))
        if betas is None:
            betas = tuple(1.0 + math.log(2.0) for _ in range(num_components))

        self._validate_shape_and_values(parameter_name="alphas", values=alphas)
        self._validate_shape_and_values(parameter_name="betas", values=betas)

        self.output_net = nn.Linear(latent_dim, action_dim * num_components * 3)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, num_components, 3)
            bias[:, :, 1] = torch.as_tensor(
                [_inverse_softplus(alpha - 1.0) for alpha in alphas],
                dtype=bias.dtype,
                device=bias.device,
            )
            bias[:, :, 2] = torch.as_tensor(
                [_inverse_softplus(beta - 1.0) for beta in betas],
                dtype=bias.dtype,
                device=bias.device,
            )

        self.weight_logits: Optional[torch.Tensor] = None
        self.components: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self.num_components, 3)
        self.weight_logits = raw[..., 0]
        alpha = 1.0 + F.softplus(raw[..., 1])
        beta = 1.0 + F.softplus(raw[..., 2])
        self.components = torchdist.Beta(
            concentration1=alpha,
            concentration0=beta,
            validate_args=False,
        )
        return self

    def _assert_ready(self) -> None:
        if self.weight_logits is None or self.components is None:
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        component_indices = torch.multinomial(
            weights.reshape(-1, self.num_components),
            num_samples=1,
        ).reshape(weights.shape[:-1])
        sampled_components = self.components.sample()
        sampled_in_01 = torch.gather(sampled_components, dim=-1, index=component_indices.unsqueeze(-1)).squeeze(-1)
        return 2.0 * sampled_in_01 - 1.0

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        mean_in_01 = (weights * self.components.mean).sum(dim=-1)
        return 2.0 * mean_in_01 - 1.0

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._assert_ready()
        actions_in_01 = ((actions + 1.0) * 0.5).clamp(self.epsilon, 1.0 - self.epsilon)
        log_components = self.components.log_prob(actions_in_01.unsqueeze(-1))
        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        log_prob_in_01 = torch.logsumexp(log_weights + log_components, dim=-1)
        log_prob_in_m1_1 = log_prob_in_01 + math.log(0.5)
        return log_prob_in_m1_1.sum(dim=AGENT_ACTIONS_DIM)

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

    def _validate_shape_and_values(self, parameter_name: str, values: tuple[float, ...]) -> None:
        if len(values) != self.num_components:
            raise ValueError(
                f"Expected {self.num_components} {parameter_name} values, got {len(values)}: {values}."
            )
        for value in values:
            if value <= 1.0:
                raise ValueError(f"All {parameter_name} values must be > 1.0, got {values}.")

    @property
    def compile_friendly(self) -> bool:
        return True

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "num_components": self.num_components,
            "epsilon": self.epsilon,
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
