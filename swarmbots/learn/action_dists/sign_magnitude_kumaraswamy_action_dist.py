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
class SignMagnitudeKumaraswamyConfig:
    initial_positive_prob: float | None = None
    epsilon: float = 1e-6
    negative_a: float = 2.0
    negative_b: float = 2.5
    positive_a: float = 2.0
    positive_b: float = 2.5
    ent_loss_coef: float = 0.0
    kumaraswamy_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(
        default_factory=default_sign_magnitude_entropy_loss_config
    )
    kumaraswamy_ent_loss_config: EntropyLossConfig = field(
        default_factory=default_sign_magnitude_entropy_loss_config
    )


class SignMagnitudeKumaraswamyActionDist(SignMagnitudeActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            initial_positive_prob: float | None = None,
            epsilon: float = 1e-6,
            negative_a: float = 2.0,
            negative_b: float = 2.5,
            positive_a: float = 2.0,
            positive_b: float = 2.5,
            ent_loss_coef: float = 0.0,
            kumaraswamy_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            kumaraswamy_ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        self.initial_negative_a = negative_a
        self.initial_negative_b = negative_b
        self.initial_positive_a = positive_a
        self.initial_positive_b = positive_b
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            initial_positive_prob=initial_positive_prob,
            epsilon=epsilon,
            initial_magnitude_parameters=(negative_a, negative_b, positive_a, positive_b),
            magnitude_parameter_names=("negative_a", "negative_b", "positive_a", "positive_b"),
            ent_loss_coef=ent_loss_coef,
            magnitude_ent_scale=kumaraswamy_ent_scale,
            categorical_ent_loss_config=categorical_ent_loss_config,
            magnitude_ent_loss_config=kumaraswamy_ent_loss_config,
            magnitude_entropy_metric_name="ent_kumaraswamy",
        )
        self.negative_a: torch.Tensor | None = None
        self.negative_b: torch.Tensor | None = None
        self.positive_a: torch.Tensor | None = None
        self.positive_b: torch.Tensor | None = None

    @property
    def kumaraswamy_ent_scale(self) -> float:
        return self.magnitude_ent_scale

    @kumaraswamy_ent_scale.setter
    def kumaraswamy_ent_scale(self, value: float) -> None:
        self.set_magnitude_ent_scale(value)

    @property
    def kumaraswamy_ent_loss_config(self) -> EntropyLossConfig:
        return self.magnitude_ent_loss_config

    @kumaraswamy_ent_loss_config.setter
    def kumaraswamy_ent_loss_config(self, value: EntropyLossConfig) -> None:
        self.magnitude_ent_loss_config = value

    @property
    def negative_dist(self) -> torchdist.Kumaraswamy | None:
        if self.negative_magnitude_dist is None:
            return None
        assert isinstance(self.negative_magnitude_dist, torchdist.Kumaraswamy)
        return self.negative_magnitude_dist

    @property
    def positive_dist(self) -> torchdist.Kumaraswamy | None:
        if self.positive_magnitude_dist is None:
            return None
        assert isinstance(self.positive_magnitude_dist, torchdist.Kumaraswamy)
        return self.positive_magnitude_dist

    def _make_magnitude_distributions(
            self,
            raw_magnitude_parameters: torch.Tensor,
    ) -> tuple[torchdist.Distribution, torchdist.Distribution]:
        self.negative_a = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 0])
        self.negative_b = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 1])
        self.positive_a = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 2])
        self.positive_b = shape_parameter_greater_than_one(raw_magnitude_parameters[..., 3])
        return (
            torchdist.Kumaraswamy(
                self.negative_a,
                self.negative_b,
                validate_args=False,
            ),
            torchdist.Kumaraswamy(
                self.positive_a,
                self.positive_b,
                validate_args=False,
            ),
        )

    def _magnitude_modes(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.negative_a is not None
        assert self.negative_b is not None
        assert self.positive_a is not None
        assert self.positive_b is not None

        def kumaraswamy_mode(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return ((a - 1.0) / (a * b - 1.0)).pow(1.0 / a)

        return (
            kumaraswamy_mode(self.negative_a, self.negative_b),
            kumaraswamy_mode(self.positive_a, self.positive_b),
        )

    def _rsample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_a is not None
        assert self.negative_b is not None
        assert self.positive_a is not None
        assert self.positive_b is not None
        return (
            kumaraswamy_icdf(
                torch.rand_like(self.negative_a),
                self.negative_a,
                self.negative_b,
                epsilon=self.epsilon,
            ),
            kumaraswamy_icdf(
                torch.rand_like(self.positive_a),
                self.positive_a,
                self.positive_b,
                epsilon=self.epsilon,
            ),
        )

    def _sample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self._rsample_independent_magnitudes()

    def set_kumaraswamy_ent_scale(self, value: float) -> None:
        self.set_magnitude_ent_scale(value)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "kumaraswamy_ent_scale": self.kumaraswamy_ent_scale,
            "negative_a": self.initial_negative_a,
            "negative_b": self.initial_negative_b,
            "positive_a": self.initial_positive_a,
            "positive_b": self.initial_positive_b,
            "kumaraswamy_ent_loss_config": serialize_dataclass(self.kumaraswamy_ent_loss_config),
        }


def kumaraswamy_icdf(
        u: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        epsilon: float,
) -> torch.Tensor:
    u = u.clamp(epsilon, 1.0 - epsilon)
    inner = -torch.expm1(torch.log1p(-u) / b)
    return torch.exp(torch.log(inner.clamp_min(torch.finfo(inner.dtype).tiny)) / a).clamp(
        epsilon,
        1.0 - epsilon,
    )
