from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sign_magnitude_kumaraswamy_action_dist import (
    SignMagnitudeKumaraswamyActionDist,
    SignMagnitudeKumaraswamyConfig,
)


@dataclass(frozen=True)
class GumbelSoftmaxSignMagnitudeBetaConfig(SignMagnitudeBetaConfig):
    gumbel_temperature: float = 1.0


@dataclass(frozen=True)
class GumbelSoftmaxSignMagnitudeKumaraswamyConfig(SignMagnitudeKumaraswamyConfig):
    gumbel_temperature: float = 1.0


def _straight_through_gumbel_softmax(
        logits: torch.Tensor,
        temperature: torch.Tensor,
) -> torch.Tensor:
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    relaxed_selection = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    hard_selection = F.one_hot(
        relaxed_selection.argmax(dim=-1),
        num_classes=logits.shape[-1],
    ).to(dtype=logits.dtype)
    return hard_selection - relaxed_selection.detach() + relaxed_selection


class _GumbelSoftmaxSignMagnitudeMixin:
    """Uses an exact hard component in forward and a biased straight-through gradient."""

    def __init__(self, *args: Any, gumbel_temperature: float = 1.0, **kwargs: Any) -> None:
        if gumbel_temperature <= 0.0:
            raise ValueError(f"gumbel_temperature must be > 0, got {gumbel_temperature}")
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_gumbel_temperature",
            torch.tensor(float(gumbel_temperature), dtype=torch.float32),
        )

    @property
    def gumbel_temperature(self) -> float:
        return float(self._gumbel_temperature.item())

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (agent, previous_actions)
        actions, _selection, _negative_magnitudes, _positive_magnitudes = self._rsample_with_selection()
        return actions

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic or not use_rsample:
            return super().get_actions_with_log_probs(
                latent_pi=latent_pi,
                deterministic=deterministic,
                agent=agent,
                previous_actions=previous_actions,
                use_rsample=False,
            )

        _ = (agent, previous_actions)
        self.update_latent_features(latent_pi)
        actions, selection, negative_magnitudes, positive_magnitudes = self._rsample_with_selection()
        component_log_probs = self._sampled_component_log_probs(
            negative_magnitudes,
            positive_magnitudes,
        )
        log_probs = (selection * component_log_probs).sum(dim=-1).sum(dim=AGENT_ACTIONS_DIM)
        return actions, log_probs

    def _rsample_with_selection(
            self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.weight_logits is not None
        selection = _straight_through_gumbel_softmax(
            self.weight_logits,
            temperature=self._gumbel_temperature.to(dtype=self.weight_logits.dtype),
        )
        negative_magnitudes, positive_magnitudes = self._rsample_independent_magnitudes()
        negative_actions = -1.0 + negative_magnitudes
        actions = (
                selection[..., self._NEGATIVE_INDEX] * negative_actions
                + selection[..., self._POSITIVE_INDEX] * positive_magnitudes
        )
        return actions, selection, negative_magnitudes, positive_magnitudes

    def set_gumbel_temperature(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"gumbel_temperature must be > 0, got {value}")
        self._gumbel_temperature.fill_(float(value))

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "gumbel_temperature": self.gumbel_temperature,
        }


class GumbelSoftmaxSignMagnitudeBetaActionDist(
        _GumbelSoftmaxSignMagnitudeMixin,
        SignMagnitudeBetaActionDist,
):
    pass


class GumbelSoftmaxSignMagnitudeKumaraswamyActionDist(
        _GumbelSoftmaxSignMagnitudeMixin,
        SignMagnitudeKumaraswamyActionDist,
):
    pass
