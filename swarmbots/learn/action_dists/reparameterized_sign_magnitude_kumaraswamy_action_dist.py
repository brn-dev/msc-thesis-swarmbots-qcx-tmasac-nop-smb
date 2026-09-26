from dataclasses import dataclass

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.sign_magnitude_kumaraswamy_action_dist import (
    SignMagnitudeKumaraswamyActionDist,
    SignMagnitudeKumaraswamyConfig,
    kumaraswamy_icdf,
)

_kumaraswamy_icdf = kumaraswamy_icdf


@dataclass(frozen=True)
class ReparameterizedSignMagnitudeKumaraswamyConfig(SignMagnitudeKumaraswamyConfig):
    pass


class ReparameterizedSignMagnitudeKumaraswamyActionDist(SignMagnitudeKumaraswamyActionDist):
    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (agent, previous_actions)
        self._assert_ready()
        assert self.weight_logits is not None
        assert self.negative_a is not None
        assert self.negative_b is not None
        assert self.positive_a is not None
        assert self.positive_b is not None

        weights = F.softmax(self.weight_logits, dim=-1)
        negative_prob = weights[..., self._NEGATIVE_INDEX]
        positive_prob = weights[..., self._POSITIVE_INDEX]
        u = torch.rand_like(negative_prob).clamp(self.epsilon, 1.0 - self.epsilon)

        negative_mask = u < negative_prob
        negative_u = (u / negative_prob.clamp_min(self.epsilon)).clamp(self.epsilon, 1.0 - self.epsilon)
        positive_u = ((u - negative_prob) / positive_prob.clamp_min(self.epsilon)).clamp(
            self.epsilon,
            1.0 - self.epsilon,
        )

        negative_actions = -1.0 + kumaraswamy_icdf(
            negative_u,
            self.negative_a,
            self.negative_b,
            epsilon=self.epsilon,
        )
        positive_actions = kumaraswamy_icdf(
            positive_u,
            self.positive_a,
            self.positive_b,
            epsilon=self.epsilon,
        )
        return torch.where(negative_mask, negative_actions, positive_actions)
