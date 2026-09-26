import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM, ActionNetInitialization
from swarmbots.learn.action_dists.bounded_quantile_action_dist import (
    BoundedQuantileActionDist,
    BoundedQuantileConfig,
)
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig


@dataclass(frozen=True)
class BernsteinQuantileConfig(BoundedQuantileConfig):
    degree: int = 6
    min_normalized_increment: float = 1e-3
    inverse_iterations: int = 48


class BernsteinQuantileActionDist(BoundedQuantileActionDist):
    MODE_NUM_CANDIDATES = 33

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            degree: int = 6,
            min_normalized_increment: float = 1e-3,
            inverse_iterations: int = 48,
            action_net_initialization: ActionNetInitialization | None = None,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        if not (0.0 < min_normalized_increment < 1.0 / degree):
            raise ValueError(
                "min_normalized_increment must be in (0, 1 / degree), "
                f"got {min_normalized_increment} for degree {degree}"
            )
        if inverse_iterations < 1:
            raise ValueError(f"inverse_iterations must be >= 1, got {inverse_iterations}")

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            parameters_per_action=degree,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )
        self.degree = degree
        self.min_normalized_increment = min_normalized_increment
        self.inverse_iterations = inverse_iterations

        self.register_buffer(
            "basis_indices",
            torch.arange(degree + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "basis_coefficients",
            torch.tensor([math.comb(degree, j) for j in range(degree + 1)], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "derivative_basis_indices",
            torch.arange(degree, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "derivative_basis_coefficients",
            torch.tensor([math.comb(degree - 1, j) for j in range(degree)], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mode_quantiles",
            torch.linspace(0.0, 1.0, self.MODE_NUM_CANDIDATES, dtype=torch.float32),
            persistent=False,
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        actions_float = actions.float()
        valid_actions = (actions_float >= -1.0) & (actions_float <= 1.0)
        inversion_targets = actions_float.clamp(-1.0, 1.0)
        lower = torch.zeros_like(inversion_targets)
        upper = torch.ones_like(inversion_targets)
        deltas, coefficients = self._deltas_and_coefficients()

        with torch.no_grad():
            for _ in range(self.inverse_iterations):
                midpoint = 0.5 * (lower + upper)
                midpoint_actions = self._action_from_u(midpoint, coefficients)
                lower = torch.where(midpoint_actions < inversion_targets, midpoint, lower)
                upper = torch.where(midpoint_actions >= inversion_targets, midpoint, upper)

        u_value = 0.5 * (lower + upper)
        reconstructed_actions = self._action_from_u(u_value, coefficients)
        derivatives = self._derivative_from_u(u_value, deltas)
        # Bisection supplies only the inverse value. A differentiable Newton step
        # adds the implicit inverse gradient without retaining the iteration graph.
        u = u_value - (reconstructed_actions - inversion_targets) / derivatives
        u = u + (u.clamp(0.0, 1.0) - u).detach()
        log_det = self._log_det_from_u(u, deltas)
        log_prob_per_action = torch.where(valid_actions, -log_det, -torch.inf)
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def _transform_forward_and_log_det(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        deltas, coefficients = self._deltas_and_coefficients()
        action_basis = self._bernstein_basis(
            u,
            degree=self.degree,
            indices=self.basis_indices.float(),
            binomial_coefficients=self.basis_coefficients.float(),
        )
        actions = (coefficients * action_basis).sum(dim=-1)
        return actions, self._log_det_from_u(u, deltas)

    def _mode_and_log_det(self) -> tuple[torch.Tensor, torch.Tensor]:
        deltas, coefficients = self._deltas_and_coefficients()
        quantiles = self.mode_quantiles.float()
        actions = self._action_from_u(quantiles, coefficients.unsqueeze(-2))
        log_det = self._log_det_from_u(quantiles, deltas.unsqueeze(-2))
        return self._select_min_log_det_candidate(
            actions=actions,
            log_det=log_det,
            quantiles=quantiles,
        )

    def _log_det_from_u(self, u: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        return self._derivative_from_u(u, deltas).log()

    def _derivative_from_u(self, u: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        derivative_basis = self._bernstein_basis(
            u,
            degree=self.degree - 1,
            indices=self.derivative_basis_indices.float(),
            binomial_coefficients=self.derivative_basis_coefficients.float(),
        )
        return self.degree * (deltas * derivative_basis).sum(dim=-1)

    def _action_from_u(self, u: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        action_basis = self._bernstein_basis(
            u,
            degree=self.degree,
            indices=self.basis_indices.float(),
            binomial_coefficients=self.basis_coefficients.float(),
        )
        return (coefficients * action_basis).sum(dim=-1)

    def _deltas_and_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        raw_parameters = self._get_raw_parameters()
        probabilities = F.softmax(raw_parameters, dim=-1)
        normalized_increments = (
            self.min_normalized_increment
            + (1.0 - self.degree * self.min_normalized_increment) * probabilities
        )
        parameterized_deltas = 2.0 * normalized_increments

        first = -torch.ones_like(parameterized_deltas[..., :1])
        last = torch.ones_like(parameterized_deltas[..., :1])
        interior = -1.0 + torch.cumsum(parameterized_deltas, dim=-1)[..., :-1]
        coefficients = torch.cat((first, interior, last), dim=-1)
        deltas = coefficients[..., 1:] - coefficients[..., :-1]
        return deltas, coefficients

    @staticmethod
    def _bernstein_basis(
            u: torch.Tensor,
            *,
            degree: int,
            indices: torch.Tensor,
            binomial_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        expanded_u = u.unsqueeze(-1)
        return (
            binomial_coefficients
            * expanded_u.pow(indices)
            * (1.0 - expanded_u).pow(degree - indices)
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "degree": self.degree,
            "min_normalized_increment": self.min_normalized_increment,
            "inverse_iterations": self.inverse_iterations,
            "mode_num_candidates": self.MODE_NUM_CANDIDATES,
        }
