from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist


@dataclass(frozen=True)
class StickySignMagnitudeBetaConfig(SignMagnitudeBetaConfig):
    stickiness: float = 0.1


class StickySignMagnitudeBetaActionDist(SignMagnitudeBetaActionDist, StickyActionDist):
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
            stickiness: float = 0.0,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            initial_positive_prob=initial_positive_prob,
            epsilon=epsilon,
            negative_alpha=negative_alpha,
            negative_beta=negative_beta,
            positive_alpha=positive_alpha,
            positive_beta=positive_beta,
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=beta_ent_scale,
            categorical_ent_loss_config=categorical_ent_loss_config,
            beta_ent_loss_config=beta_ent_loss_config,
        )
        self._init_stickiness_buffer()
        self.set_stickiness(stickiness)

    def requires_previous_actions(self) -> bool:
        return True

    def _component_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        return self._effective_probs(previous_actions)

    def _component_log_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        return self._effective_probs(previous_actions).log()

    def _effective_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        assert self.weight_logits is not None
        base_probs = F.softmax(self.weight_logits, dim=-1)
        if previous_actions is None:
            raise ValueError("StickySignMagnitudeBetaActionDist requires previous_actions.")

        previous_indices = self._actions_to_indices(previous_actions)
        previous_one_hot = F.one_hot(
            previous_indices,
            num_classes=self._N_MIXTURE_COMPONENTS,
        ).to(dtype=base_probs.dtype)
        stickiness = self._stickiness_tensor(dtype=base_probs.dtype, device=base_probs.device)
        return torch.lerp(base_probs, previous_one_hot, stickiness)

    def _actions_to_indices(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.where(actions < 0.0, self._NEGATIVE_INDEX, self._POSITIVE_INDEX)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "stickiness": self.get_stickiness(),
        }
