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
class RationalQuadraticSplineQuantileConfig(BoundedQuantileConfig):
    num_bins: int = 6
    min_bin_width: float = 1e-3
    min_bin_height: float = 1e-3
    min_derivative: float = 1e-3
    fixed_boundary_derivatives: bool = False


class RationalQuadraticSplineQuantileActionDist(BoundedQuantileActionDist):
    MODE_CANDIDATES_PER_BIN = 9

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            num_bins: int = 6,
            min_bin_width: float = 1e-3,
            min_bin_height: float = 1e-3,
            min_derivative: float = 1e-3,
            fixed_boundary_derivatives: bool = False,
            action_net_initialization: ActionNetInitialization | None = None,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if not (0.0 < min_bin_width < 1.0 / num_bins):
            raise ValueError(f"min_bin_width must be in (0, 1 / num_bins), got {min_bin_width}")
        if not (0.0 < min_bin_height < 1.0 / num_bins):
            raise ValueError(f"min_bin_height must be in (0, 1 / num_bins), got {min_bin_height}")
        if not (0.0 < min_derivative < 2.0):
            raise ValueError(f"min_derivative must be in (0, 2), got {min_derivative}")

        num_derivative_parameters = num_bins - 1 if fixed_boundary_derivatives else num_bins + 1
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            parameters_per_action=2 * num_bins + num_derivative_parameters,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.fixed_boundary_derivatives = fixed_boundary_derivatives
        self.derivative_parameter_offset = math.log(math.expm1(2.0 - min_derivative))
        self.register_buffer(
            "mode_bin_coordinates",
            torch.linspace(0.0, 1.0, self.MODE_CANDIDATES_PER_BIN, dtype=torch.float32),
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
        widths, heights, derivatives, _x_knots, y_knots = self._spline_parameters()
        inversion_actions = actions_float.clamp(-1.0, 1.0)
        bin_indices = self._find_bins(inversion_actions, y_knots)

        y_left = self._gather(y_knots, bin_indices)
        y_right = self._gather(y_knots, bin_indices + 1)
        heights_in_bin = self._gather(heights, bin_indices)
        widths_in_bin = self._gather(widths, bin_indices)
        derivative_left = self._gather(derivatives, bin_indices)
        derivative_right = self._gather(derivatives, bin_indices + 1)
        secant_slopes = heights_in_bin / widths_in_bin

        normalized_distance_from_left = (inversion_actions - y_left) / heights_in_bin
        normalized_distance_from_right = (y_right - inversion_actions) / heights_in_bin
        mirrored = normalized_distance_from_right < normalized_distance_from_left
        normalized_distance = torch.where(
            mirrored,
            normalized_distance_from_right,
            normalized_distance_from_left,
        ).clamp(0.0, 1.0)
        derivative_at_start = torch.where(mirrored, derivative_right, derivative_left)
        derivative_at_end = torch.where(mirrored, derivative_left, derivative_right)

        with torch.no_grad():
            inverse_coordinate = self._solve_inverse_coordinate(
                normalized_distance=normalized_distance,
                secant_slopes=secant_slopes,
                derivative_at_start=derivative_at_start,
                derivative_at_end=derivative_at_end,
            ).clamp(0.0, 1.0)

        zero = torch.zeros_like(y_left)
        action_offset, initial_log_det = self._evaluate_bin(
            theta=inverse_coordinate,
            y_left=zero,
            widths=widths_in_bin,
            heights=heights_in_bin,
            derivative_left=derivative_at_start,
            derivative_right=derivative_at_end,
        )
        reconstructed_actions = torch.where(mirrored, y_right - action_offset, y_left + action_offset)
        derivative_sign = torch.where(mirrored, -torch.ones_like(y_left), torch.ones_like(y_left))
        derivative_wrt_coordinate = derivative_sign * initial_log_det.exp() * widths_in_bin

        # The detached analytic root supplies the value; one differentiable Newton
        # step recovers the implicit inverse gradient with no retained solver graph.
        inverse_coordinate = inverse_coordinate - (
            reconstructed_actions - inversion_actions
        ) / derivative_wrt_coordinate
        inverse_coordinate = inverse_coordinate + (
            inverse_coordinate.clamp(0.0, 1.0) - inverse_coordinate
        ).detach()
        _reconstructed_actions, log_det = self._evaluate_bin(
            theta=inverse_coordinate,
            y_left=zero,
            widths=widths_in_bin,
            heights=heights_in_bin,
            derivative_left=derivative_at_start,
            derivative_right=derivative_at_end,
        )
        log_prob_per_action = torch.where(valid_actions, -log_det, -torch.inf)
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    @staticmethod
    def _solve_inverse_coordinate(
            *,
            normalized_distance: torch.Tensor,
            secant_slopes: torch.Tensor,
            derivative_at_start: torch.Tensor,
            derivative_at_end: torch.Tensor,
    ) -> torch.Tensor:
        slope_difference = derivative_at_start + derivative_at_end - 2.0 * secant_slopes
        quadratic_a = normalized_distance * slope_difference + secant_slopes - derivative_at_start
        quadratic_b = derivative_at_start - normalized_distance * slope_difference
        quadratic_c = -secant_slopes * normalized_distance
        discriminant = (quadratic_b.square() - 4.0 * quadratic_a * quadratic_c).clamp_min(0.0)
        sqrt_discriminant = discriminant.sqrt()
        coordinate = torch.where(
            quadratic_b >= 0.0,
            (2.0 * quadratic_c) / (-quadratic_b - sqrt_discriminant),
            (-quadratic_b + sqrt_discriminant) / (2.0 * quadratic_a),
        )
        return torch.where(normalized_distance <= 0.0, torch.zeros_like(coordinate), coordinate)

    def _transform_forward_and_log_det(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivatives, x_knots, y_knots = self._spline_parameters()
        bin_indices = self._find_bins(u, x_knots)
        x_left = self._gather(x_knots, bin_indices)
        y_left = self._gather(y_knots, bin_indices)
        widths_in_bin = self._gather(widths, bin_indices)
        heights_in_bin = self._gather(heights, bin_indices)
        derivative_left = self._gather(derivatives, bin_indices)
        derivative_right = self._gather(derivatives, bin_indices + 1)

        theta = (u - x_left) / widths_in_bin
        return self._evaluate_bin(
            theta=theta,
            y_left=y_left,
            widths=widths_in_bin,
            heights=heights_in_bin,
            derivative_left=derivative_left,
            derivative_right=derivative_right,
        )

    def _mode_and_log_det(self) -> tuple[torch.Tensor, torch.Tensor]:
        widths, heights, derivatives, x_knots, y_knots = self._spline_parameters()
        theta = self.mode_bin_coordinates.float()
        actions, log_det = self._evaluate_bin(
            theta=theta,
            y_left=y_knots[..., :-1].unsqueeze(-1),
            widths=widths.unsqueeze(-1),
            heights=heights.unsqueeze(-1),
            derivative_left=derivatives[..., :-1].unsqueeze(-1),
            derivative_right=derivatives[..., 1:].unsqueeze(-1),
        )
        quantiles = x_knots[..., :-1].unsqueeze(-1) + widths.unsqueeze(-1) * theta
        return self._select_min_log_det_candidate(
            actions=actions.flatten(start_dim=-2),
            log_det=log_det.flatten(start_dim=-2),
            quantiles=quantiles.flatten(start_dim=-2),
        )

    @staticmethod
    def _evaluate_bin(
            *,
            theta: torch.Tensor,
            y_left: torch.Tensor,
            widths: torch.Tensor,
            heights: torch.Tensor,
            derivative_left: torch.Tensor,
            derivative_right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        one_minus_theta = 1.0 - theta
        theta_complement_product = theta * one_minus_theta
        secant_slopes = heights / widths
        denominator = secant_slopes + (
            derivative_left + derivative_right - 2.0 * secant_slopes
        ) * theta_complement_product
        numerator = (
            secant_slopes * theta.square()
            + derivative_left * theta_complement_product
        )
        actions = y_left + heights * numerator / denominator

        derivative_numerator = (
            derivative_right * theta.square()
            + 2.0 * secant_slopes * theta_complement_product
            + derivative_left * one_minus_theta.square()
        )
        log_det = (
            2.0 * secant_slopes.log()
            + derivative_numerator.log()
            - 2.0 * denominator.log()
        )
        return actions, log_det

    def _spline_parameters(
            self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_parameters = self._get_raw_parameters()
        raw_widths = raw_parameters[..., :self.num_bins]
        raw_heights = raw_parameters[..., self.num_bins:2 * self.num_bins]
        raw_derivatives = raw_parameters[..., 2 * self.num_bins:]

        parameterized_widths = self.min_bin_width + (
            1.0 - self.num_bins * self.min_bin_width
        ) * F.softmax(raw_widths, dim=-1)
        normalized_heights = self.min_bin_height + (
            1.0 - self.num_bins * self.min_bin_height
        ) * F.softmax(raw_heights, dim=-1)
        parameterized_heights = 2.0 * normalized_heights
        learned_derivatives = self.min_derivative + F.softplus(
            raw_derivatives + self.derivative_parameter_offset
        )
        if self.fixed_boundary_derivatives:
            boundary_derivative = torch.full_like(parameterized_widths[..., :1], 2.0)
            derivatives = torch.cat((boundary_derivative, learned_derivatives, boundary_derivative), dim=-1)
        else:
            derivatives = learned_derivatives

        zero = torch.zeros_like(parameterized_widths[..., :1])
        one = torch.ones_like(parameterized_widths[..., :1])
        x_knots = torch.cat((zero, torch.cumsum(parameterized_widths, dim=-1)[..., :-1], one), dim=-1)
        widths = x_knots[..., 1:] - x_knots[..., :-1]
        minus_one = -torch.ones_like(parameterized_heights[..., :1])
        y_knots = torch.cat(
            (minus_one, -1.0 + torch.cumsum(parameterized_heights, dim=-1)[..., :-1], one),
            dim=-1,
        )
        heights = y_knots[..., 1:] - y_knots[..., :-1]
        return widths, heights, derivatives, x_knots, y_knots

    @staticmethod
    def _find_bins(values: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        interior_knots = knots[..., 1:-1]
        return (values.unsqueeze(-1) >= interior_knots).sum(dim=-1)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values.gather(dim=-1, index=indices.unsqueeze(-1)).squeeze(-1)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "num_bins": self.num_bins,
            "min_bin_width": self.min_bin_width,
            "min_bin_height": self.min_bin_height,
            "min_derivative": self.min_derivative,
            "fixed_boundary_derivatives": self.fixed_boundary_derivatives,
            "mode_candidates_per_bin": self.MODE_CANDIDATES_PER_BIN,
        }
