import math
from dataclasses import dataclass, field
from typing import Any, Optional, Self

import torch
from torch import distributions as torchdist, nn
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionDist,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_action_metrics,
    resolve_action_metrics_splitter,
    validate_probability_clamp_epsilon,
)
from swarmbots.learn.action_dists.entropy_utils import (
    AgentActionsReduction,
    EntropyLossConfig,
    compute_ent_loss,
    compute_ent_metrics,
)
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.serialization_utils import serialize_dataclass


def _default_entropy_loss_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


@dataclass(frozen=True)
class BetaConfig:
    alpha: float = 1.0 + math.log(2.0)
    beta: float = 1.0 + math.log(2.0)
    epsilon: float = 1e-6
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)


class BetaActionDist(ActionDist):
    _OUTPUTS_PER_ACTION = 2

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            alpha: float = 1.0 + math.log(2.0),
            beta: float = 1.0 + math.log(2.0),
            epsilon: float = 1e-6,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        if alpha <= 1.0:
            raise ValueError(f"alpha must be > 1.0, got {alpha}")
        if beta <= 1.0:
            raise ValueError(f"beta must be > 1.0, got {beta}")
        validate_probability_clamp_epsilon(epsilon)
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.ent_loss_coef = ent_loss_coef
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else _default_entropy_loss_config()

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            bias[:, 0] = _inverse_softplus(alpha - 1.0)
            bias[:, 1] = _inverse_softplus(beta - 1.0)

        self.distribution: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        alpha = 1.0 + F.softplus(raw[..., 0])
        beta = 1.0 + F.softplus(raw[..., 1])
        self.distribution = torchdist.Beta(
            concentration1=alpha,
            concentration0=beta,
            validate_args=False,
        )
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = self.distribution.sample()
        return 2.0 * actions_01 - 1.0

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = self.distribution.rsample()
        return 2.0 * actions_01 - 1.0

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return 2.0 * self.distribution.mean - 1.0

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        actions_01 = ((actions + 1.0) * 0.5).clamp(self.epsilon, 1.0 - self.epsilon)
        log_prob_01 = self.distribution.log_prob(actions_01)
        return (log_prob_01 + math.log(0.5)).sum(dim=AGENT_ACTIONS_DIM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0.0:
            return {}, {}

        entropy_per_action = self.distribution.entropy() + math.log(2.0)
        entropy_loss = self.ent_loss_coef * compute_ent_loss(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
        )
        entropy_metrics = compute_ent_metrics(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=resolve_action_metrics_splitter(action_splitter),
            name="ent",
        )
        return {"entropy": entropy_loss}, entropy_metrics

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

    @property
    def compile_friendly(self) -> bool:
        return True

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "alpha": self.alpha,
            "beta": self.beta,
            "epsilon": self.epsilon,
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
