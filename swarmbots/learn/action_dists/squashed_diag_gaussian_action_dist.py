from dataclasses import dataclass, field
from typing import Optional, Self, Any

import torch

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput, ActionNetInitialization
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@dataclass(frozen=True)
class SquashedDiagGaussianConfig:
    std: float
    std_learnable: bool
    epsilon: float = 1e-6
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=EntropyLossConfig)


# Inspired by
class SquashedDiagGaussianActionDist(DiagGaussianActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )

        self.epsilon = epsilon
        self._last_gaussian_actions: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        super().update_latent_features(latent_pi)
        return self

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
            gaussian_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))

        return log_prob

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

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_gaussian_actions = super().sample()
        return torch.tanh(self._last_gaussian_actions)

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_gaussian_actions = super().rsample()
        return torch.tanh(self._last_gaussian_actions)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        self._last_gaussian_actions = super().mode()
        return torch.tanh(self._last_gaussian_actions)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ):
        # Avoids inverse tanh in the common sample/log-prob path, which is less numerically stable.
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
        )
        log_probs = self.log_prob(actions, gaussian_actions=self._last_gaussian_actions)
        return actions, log_probs

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "epsilon": self.epsilon,
            "squash_output": True,
        }

    @property
    def compile_friendly(self) -> bool:
        return True
