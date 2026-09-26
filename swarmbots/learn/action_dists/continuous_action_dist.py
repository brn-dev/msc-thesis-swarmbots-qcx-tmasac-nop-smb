import abc
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist

from swarmbots.learn.action_dists.action_dist import (
    BOUNDED_ACTION_HISTOGRAM,
    ActionNetInitialization,
    ActionDist,
    ActionMetricsSplitterInput,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.action_dists.entropy_utils import (
    EntropyLossConfig,
    compute_ent_loss,
    compute_ent_metrics,
)
from swarmbots.learn.losses import LossMetrics
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.summary_statistics import compute_summary_statistics


STD_HISTOGRAM_BINS = 11


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        if ent_loss_coef < 0:
            raise ValueError(f"Expected ent_loss_coef >= 0, got {ent_loss_coef}")
        self.ent_loss_coef = ent_loss_coef
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else EntropyLossConfig()

        self.distribution: Optional[torchdist.Distribution] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    @abc.abstractmethod
    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        raise NotImplementedError

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def compute_entropy_loss(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[Optional[torch.Tensor], LossMetrics]:
        if self.ent_loss_coef <= 0:
            return None, {}
        entropy_per_action = self.distribution.entropy()
        entropy_loss_per_agent = self.ent_loss_coef * compute_ent_loss(
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
        return entropy_loss_per_agent, metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
        }


    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        splitter = resolve_action_metrics_splitter(action_splitter)
        metrics = compute_action_metrics(actions, splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

        if not hasattr(self, "log_stds"):
            return metrics
        log_stds = getattr(self, "log_stds")
        if not isinstance(log_stds, torch.Tensor):
            return metrics

        std_values = torch.exp(log_stds)

        can_split_stds = not (std_values.ndim >= 1 and std_values.shape[-1] == 1 and self.action_dim > 1)
        if splitter is not None and can_split_stds:
            for key, split_stds in splitter(std_values).items():
                metrics[f"std_{key}"] = compute_summary_statistics(
                    split_stds, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
                )
        else:
            metrics["std"] = compute_summary_statistics(
                std_values, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
            )

        return metrics

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"Expected ent_loss_coef >= 0, got {value}")
        self.ent_loss_coef = value
