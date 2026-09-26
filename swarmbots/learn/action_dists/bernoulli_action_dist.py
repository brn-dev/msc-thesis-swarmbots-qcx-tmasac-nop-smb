import math
from dataclasses import dataclass, field
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.action_dists.discrete_action_dist import DiscreteActionDist
from swarmbots.learn.action_dists.entropy_utils import (
    EntropyLossConfig,
    compute_ent_loss,
    compute_ent_metrics,
)
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.summary_statistics import HistogramConfig


BERNOULLI_ACTION_HISTOGRAM = HistogramConfig(bins=2, low=0.0, high=1.0)


@dataclass(frozen=True)
class BernoulliConfig:
    initial_prob: float | None = None
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=EntropyLossConfig)


class BernoulliActionDist(DiscreteActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            initial_prob: float | None = None,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ):
        if ent_loss_coef < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.initial_prob = initial_prob
        self.ent_loss_coef = ent_loss_coef
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else EntropyLossConfig()

        self.distribution: Optional[torchdist.Bernoulli] = None
        if initial_prob is not None:
            if not 0.0 < initial_prob < 1.0:
                raise ValueError(
                    f"initial_prob must be strictly between 0 and 1, got {initial_prob}."
                )

            initial_logit = math.log(initial_prob / (1.0 - initial_prob))
            with torch.no_grad():
                self.action_net.bias.fill_(initial_logit)

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        self.distribution = torchdist.Bernoulli(logits=action_logits, validate_args=False)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return torch.round(self.distribution.probs)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=AGENT_ACTIONS_DIM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0:
            return {}, {}
        entropy_per_action = self.distribution.entropy()
        entropy_loss = self.ent_loss_coef * compute_ent_loss(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
        )
        metrics = compute_ent_metrics(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=resolve_action_metrics_splitter(action_splitter),
            name="ent",
        )
        return {"entropy": entropy_loss}, metrics

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    @property
    def compile_friendly(self) -> bool:
        return True

    def _get_metrics_histogram(self) -> HistogramConfig:
        return BERNOULLI_ACTION_HISTOGRAM

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "initial_prob": self.initial_prob,
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
        }
