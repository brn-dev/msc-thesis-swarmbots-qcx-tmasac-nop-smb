import abc
from typing import Self, Any

import torch

from swarmbots.learn.action_dists.action_dist import (
    ActionNetInitialization,
    ActionDist,
    ActionMetricsSplitterInput,
    compute_action_metrics,
)
from swarmbots.learn.summary_statistics import HistogramConfig


class DiscreteActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_logits = self.action_net(latent_pi)
        return self.update_distribution_params(action_logits)

    @abc.abstractmethod
    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        raise NotImplementedError

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(
            actions,
            action_splitter,
            histogram=self._get_metrics_histogram(),
        )

    def _get_metrics_histogram(self) -> HistogramConfig | int:
        return 11
