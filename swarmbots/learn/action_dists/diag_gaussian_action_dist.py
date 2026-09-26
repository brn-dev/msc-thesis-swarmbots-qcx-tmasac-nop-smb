import math
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput, ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class DiagGaussianActionDist(ContinuousActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )
        self.std_learnable = std_learnable

        self.log_stds = nn.Parameter(
            torch.ones((self.action_dim,)) * math.log(std),
            requires_grad=std_learnable
        )

        self.distribution: Optional[torchdist.Normal] = None

    def set_std(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        with torch.no_grad():
            self.log_stds[:] = math.log(std)

    def scale_std(self, multiplier: float) -> None:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        with torch.no_grad():
            self.log_stds += math.log(multiplier)

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        self.distribution = torchdist.Normal(
            loc=means,
            scale=torch.exp(log_stds),
            validate_args=False,
        )
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.sample()

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return self.distribution.mean

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        ent_loss, ent_loss_metrics = self.compute_entropy_loss(
            agent_mask=agent_mask,
            action_splitter=action_splitter,
        )
        losses: LossDict = {}
        if ent_loss is not None:
            losses["entropy"] = ent_loss
        return losses, ent_loss_metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        std_tensor = torch.exp(self.log_stds.detach())
        if std_tensor.numel() == 1:
            std_value: float | list[float] = float(std_tensor.item())
        else:
            std_value = std_tensor.cpu().tolist()
        return {
            **super().get_hyper_parameters(),
            "std_learnable": self.std_learnable,
            "std": std_value,
        }

    @property
    def compile_friendly(self) -> bool:
        return True
