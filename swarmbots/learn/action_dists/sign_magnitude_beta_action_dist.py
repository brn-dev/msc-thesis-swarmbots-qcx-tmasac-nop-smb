import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import distributions as torchdist

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.sign_magnitude_action_dist import (
    SignMagnitudeActionDist,
    default_sign_magnitude_entropy_loss_config,
    shape_parameter_greater_than_one,
)
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class SignMagnitudeBetaConfig:
    initial_positive_prob: float | None = None
    epsilon: float = 1e-6
    negative_alpha: float = 1.0 + math.log(2.0)
    negative_beta: float = 1.0 + math.log(2.0)
    positive_alpha: float = 1.0 + math.log(2.0)
    positive_beta: float = 1.0 + math.log(2.0)
    ent_loss_coef: float = 0.0
    beta_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(
        default_factory=default_sign_magnitude_entropy_loss_config
    )
    beta_ent_loss_config: EntropyLossConfig = field(
        default_factory=default_sign_magnitude_entropy_loss_config
    )


class SignMagnitudeBetaActionDist(SignMagnitudeActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            initial_positive_prob: float | None = None,
            epsilon: float = 1e-6,
            negative_alpha: float = 1.0 + math.log(2.0),
            negative_beta: float = 1.0 + math.log(2.0),
            positive_alpha: float = 1.0 + math.log(2.0),
            positive_beta: float = 1.0 + math.log(2.0),
            ent_loss_coef: float = 0.0,
            beta_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            beta_ent_loss_config: EntropyLossConfig | None = None,
            initial_zero_prob: float | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            initial_positive_prob=initial_positive_prob,
            epsilon=epsilon,
            initial_magnitude_parameters=(negative_alpha, negative_beta, positive_alpha, positive_beta),
            magnitude_parameter_names=(
                "negative_alpha",
                "negative_beta",
                "positive_alpha",
                "positive_beta",
            ),
            ent_loss_coef=ent_loss_coef,
            magnitude_ent_scale=beta_ent_scale,
            categorical_ent_loss_config=categorical_ent_loss_config,
            magnitude_ent_loss_config=beta_ent_loss_config,
            magnitude_entropy_metric_name="ent_beta",
            initial_zero_prob=initial_zero_prob,
        )

    @property
    def beta_ent_scale(self) -> float:
        return self.magnitude_ent_scale

    @beta_ent_scale.setter
    def beta_ent_scale(self, value: float) -> None:
        self.set_magnitude_ent_scale(value)

    @property
    def beta_ent_loss_config(self) -> EntropyLossConfig:
        return self.magnitude_ent_loss_config

    @beta_ent_loss_config.setter
    def beta_ent_loss_config(self, value: EntropyLossConfig) -> None:
        self.magnitude_ent_loss_config = value

    @property
    def negative_beta_dist(self) -> torchdist.Beta | None:
        if self.negative_magnitude_dist is None:
            return None
        assert isinstance(self.negative_magnitude_dist, torchdist.Beta)
        return self.negative_magnitude_dist

    @property
    def positive_beta_dist(self) -> torchdist.Beta | None:
        if self.positive_magnitude_dist is None:
            return None
        assert isinstance(self.positive_magnitude_dist, torchdist.Beta)
        return self.positive_magnitude_dist

    def _make_magnitude_distributions(
            self,
            raw_magnitude_parameters: torch.Tensor,
    ) -> tuple[torchdist.Distribution, torchdist.Distribution]:
        negative_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 0])
        negative_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 1])
        positive_alpha = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 2])
        positive_beta = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 3])
        return (
            torchdist.Beta(
                concentration1=negative_alpha,
                concentration0=negative_beta,
                validate_args=False,
            ),
            torchdist.Beta(
                concentration1=positive_alpha,
                concentration0=positive_beta,
                validate_args=False,
            ),
        )

    def _magnitude_modes(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.negative_beta_dist is not None
        assert self.positive_beta_dist is not None

        def beta_mode(distribution: torchdist.Beta) -> torch.Tensor:
            return (
                (distribution.concentration1 - 1.0)
                / (distribution.concentration1 + distribution.concentration0 - 2.0)
            )

        return beta_mode(self.negative_beta_dist), beta_mode(self.positive_beta_dist)

    def set_beta_ent_scale(self, value: float) -> None:
        self.set_magnitude_ent_scale(value)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "beta_ent_scale": self.beta_ent_scale,
            "beta_ent_loss_config": serialize_dataclass(self.beta_ent_loss_config),
        }
