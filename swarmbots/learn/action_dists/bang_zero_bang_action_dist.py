from dataclasses import dataclass, field
from typing import Optional, Self, Any

import torch
from torch import distributions as torchdist

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


@dataclass(frozen=True)
class BangZeroBangConfig:
    bang: float = 1.0
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=EntropyLossConfig)


class BangZeroBangActionDist(DiscreteActionDist):
    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            bang: float = 1.0,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        if bang <= 0.0:
            raise ValueError(f"bang must be > 0, got {bang}.")
        if bang > 1.0:
            raise ValueError(f"bang must be <= 1.0 for Box(-1, 1) actions, got {bang}.")
        if ent_loss_coef < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim * 3,
            action_net_initialization=action_net_initialization,
        )
        self.agent_action_dim = action_dim
        self.bang = bang
        self.ent_loss_coef = ent_loss_coef
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else EntropyLossConfig()
        self.distribution: Optional[torchdist.Categorical] = None

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        reshaped_logits = action_logits.view(*action_logits.shape[:-1], self.agent_action_dim, 3)
        self.distribution = torchdist.Categorical(logits=reshaped_logits, validate_args=False)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampled_indices = self.distribution.sample()
        return self._indices_to_actions(sampled_indices)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        greedy_indices = self.distribution.probs.argmax(dim=-1)
        return self._indices_to_actions(greedy_indices)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        action_indices = self._actions_to_indices(actions)
        return self.distribution.log_prob(action_indices).sum(dim=AGENT_ACTIONS_DIM)

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

    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions / self.bang).round().to(dtype=torch.long).add(1).clamp_(0, 2)

    def _indices_to_actions(self, indices: torch.Tensor) -> torch.Tensor:
        return (indices.to(dtype=torch.float32) - 1.0) * self.bang

    def _get_metrics_histogram(self) -> HistogramConfig:
        return HistogramConfig(bins=3, low=-self.bang, high=self.bang)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "agent_action_dim": self.agent_action_dim,
            "bang": self.bang,
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
        }
