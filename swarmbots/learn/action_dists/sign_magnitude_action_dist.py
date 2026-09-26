import abc
import math
from typing import Any, Self

import torch
from torch import distributions as torchdist
from torch import nn
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


def default_sign_magnitude_entropy_loss_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def shape_parameter_greater_than_one(raw_parameter: torch.Tensor) -> torch.Tensor:
    minimum_offset = torch.finfo(raw_parameter.dtype).eps
    return 1.0 + F.softplus(raw_parameter).clamp_min(minimum_offset)


class SignMagnitudeActionDist(ActionDist):
    _N_MIXTURE_COMPONENTS = 2
    _OUTPUTS_PER_ACTION = 6

    _NEGATIVE_INDEX = 0
    _ZERO_INDEX: int | None = None
    _POSITIVE_INDEX = 1

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            *,
            initial_positive_prob: float | None,
            epsilon: float,
            initial_magnitude_parameters: tuple[float, float, float, float],
            magnitude_parameter_names: tuple[str, str, str, str],
            ent_loss_coef: float,
            magnitude_ent_scale: float,
            categorical_ent_loss_config: EntropyLossConfig | None,
            magnitude_ent_loss_config: EntropyLossConfig | None,
            magnitude_entropy_metric_name: str,
            initial_zero_prob: float | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        validate_probability_clamp_epsilon(epsilon)
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        if magnitude_ent_scale < 0.0:
            raise ValueError(f"magnitude_ent_scale must be >= 0, got {magnitude_ent_scale}")
        if initial_positive_prob is not None and not (0.0 < initial_positive_prob < 1.0):
            raise ValueError(
                f"initial_positive_prob must be strictly between 0 and 1, got {initial_positive_prob}"
            )
        if initial_zero_prob is not None and not (0.0 < initial_zero_prob < 1.0):
            raise ValueError(
                f"initial_zero_prob must be strictly between 0 and 1, got {initial_zero_prob}"
            )
        if self._ZERO_INDEX is None and initial_zero_prob is not None:
            raise ValueError("initial_zero_prob is only valid for a distribution with a zero component")
        for name, value in zip(magnitude_parameter_names, initial_magnitude_parameters, strict=True):
            if value <= 1.0:
                raise ValueError(f"{name} must be > 1.0, got {value}")

        self.initial_positive_prob = initial_positive_prob
        self.initial_zero_prob = initial_zero_prob
        self.epsilon = epsilon
        self.ent_loss_coef = ent_loss_coef
        self.magnitude_ent_scale = magnitude_ent_scale
        self.categorical_ent_loss_config = (
            categorical_ent_loss_config
            if categorical_ent_loss_config is not None
            else default_sign_magnitude_entropy_loss_config()
        )
        self.magnitude_ent_loss_config = (
            magnitude_ent_loss_config
            if magnitude_ent_loss_config is not None
            else default_sign_magnitude_entropy_loss_config()
        )
        self.magnitude_entropy_metric_name = magnitude_entropy_metric_name

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            if self._ZERO_INDEX is not None and (
                    initial_zero_prob is not None or initial_positive_prob is not None
            ):
                zero_prob = (
                    initial_zero_prob
                    if initial_zero_prob is not None
                    else (1.0 - initial_positive_prob) / 2.0
                )
                positive_prob = (
                    initial_positive_prob
                    if initial_positive_prob is not None
                    else (1.0 - initial_zero_prob) / 2.0
                )
                negative_prob = 1.0 - zero_prob - positive_prob
                if negative_prob <= 0.0:
                    raise ValueError(
                        "initial_zero_prob + initial_positive_prob must be < 1, "
                        f"got {zero_prob + positive_prob}"
                    )
                bias[:, self._NEGATIVE_INDEX] = math.log(negative_prob)
                bias[:, self._ZERO_INDEX] = math.log(zero_prob)
                bias[:, self._POSITIVE_INDEX] = math.log(positive_prob)
            elif initial_positive_prob is not None:
                bias[:, self._NEGATIVE_INDEX] = math.log(1.0 - initial_positive_prob)
                bias[:, self._POSITIVE_INDEX] = math.log(initial_positive_prob)
            for output_index, parameter in enumerate(
                    initial_magnitude_parameters,
                    start=self._N_MIXTURE_COMPONENTS,
            ):
                bias[:, output_index] = inverse_softplus(parameter - 1.0)

        self.weight_logits: torch.Tensor | None = None
        self.negative_magnitude_dist: torchdist.Distribution | None = None
        self.positive_magnitude_dist: torchdist.Distribution | None = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        self.weight_logits = raw[..., :self._N_MIXTURE_COMPONENTS]
        self.negative_magnitude_dist, self.positive_magnitude_dist = self._make_magnitude_distributions(
            raw[..., self._N_MIXTURE_COMPONENTS:]
        )
        return self

    @abc.abstractmethod
    def _make_magnitude_distributions(
            self,
            raw_magnitude_parameters: torch.Tensor,
    ) -> tuple[torchdist.Distribution, torchdist.Distribution]:
        raise NotImplementedError

    @abc.abstractmethod
    def _magnitude_modes(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent
        self._assert_ready()
        component_indices = torchdist.Categorical(
            probs=self._component_probs(previous_actions),
            validate_args=False,
        ).sample()
        negative_magnitudes, positive_magnitudes = self._sample_independent_magnitudes()
        return self._select_actions(component_indices, negative_magnitudes, positive_magnitudes)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        negative_magnitude_mode, positive_magnitude_mode = self._magnitude_modes()
        component_log_probs = self._component_log_probs(previous_actions)
        component_mode_scores = [
            component_log_probs[..., self._NEGATIVE_INDEX]
            + self.negative_magnitude_dist.log_prob(negative_magnitude_mode),
        ]
        if self._ZERO_INDEX is not None:
            component_mode_scores.append(component_log_probs[..., self._ZERO_INDEX])
        component_mode_scores.append(
            component_log_probs[..., self._POSITIVE_INDEX]
            + self.positive_magnitude_dist.log_prob(positive_magnitude_mode)
        )
        stacked_component_mode_scores = torch.stack(component_mode_scores, dim=-1)
        component_indices = stacked_component_mode_scores.argmax(dim=-1)
        return self._select_actions(
            component_indices,
            negative_magnitude_mode,
            positive_magnitude_mode,
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._assert_ready()
        log_weights = self._component_log_probs(previous_actions)
        negative_mask = actions < 0.0
        negative_log_probs, positive_log_probs = self._magnitude_log_probs(actions + 1.0, actions)
        log_prob_per_action = torch.where(
            negative_mask,
            log_weights[..., self._NEGATIVE_INDEX] + negative_log_probs,
            log_weights[..., self._POSITIVE_INDEX] + positive_log_probs,
        )
        if self._ZERO_INDEX is not None:
            log_prob_per_action = torch.where(
                actions == 0.0,
                log_weights[..., self._ZERO_INDEX],
                log_prob_per_action,
            )
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def _component_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        _ = previous_actions
        assert self.weight_logits is not None
        return F.softmax(self.weight_logits, dim=-1)

    def _component_log_probs(self, previous_actions: torch.Tensor | None) -> torch.Tensor:
        _ = previous_actions
        assert self.weight_logits is not None
        return F.log_softmax(self.weight_logits, dim=-1)

    def _sample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return self.negative_magnitude_dist.sample(), self.positive_magnitude_dist.sample()

    def _rsample_independent_magnitudes(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return (
            self.negative_magnitude_dist.rsample().clamp(self.epsilon, 1.0 - self.epsilon),
            self.positive_magnitude_dist.rsample().clamp(self.epsilon, 1.0 - self.epsilon),
        )

    def _magnitude_log_probs(
            self,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        return (
            self.negative_magnitude_dist.log_prob(
                negative_magnitudes.clamp(self.epsilon, 1.0 - self.epsilon)
            ),
            self.positive_magnitude_dist.log_prob(
                positive_magnitudes.clamp(self.epsilon, 1.0 - self.epsilon)
            ),
        )

    def _select_actions(
            self,
            component_indices: torch.Tensor,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        negative_actions = -1.0 + negative_magnitudes
        actions = torch.where(
            component_indices == self._NEGATIVE_INDEX,
            negative_actions,
            positive_magnitudes,
        )
        if self._ZERO_INDEX is not None:
            actions = torch.where(component_indices == self._ZERO_INDEX, 0.0, actions)
        return actions

    def _sampled_component_log_probs(
            self,
            negative_magnitudes: torch.Tensor,
            positive_magnitudes: torch.Tensor,
    ) -> torch.Tensor:
        negative_magnitude_log_probs, positive_magnitude_log_probs = self._magnitude_log_probs(
            negative_magnitudes,
            positive_magnitudes,
        )
        log_weights = self._component_log_probs(previous_actions=None)
        component_log_probs = [
            log_weights[..., self._NEGATIVE_INDEX] + negative_magnitude_log_probs,
        ]
        if self._ZERO_INDEX is not None:
            component_log_probs.append(log_weights[..., self._ZERO_INDEX])
        component_log_probs.append(
            log_weights[..., self._POSITIVE_INDEX] + positive_magnitude_log_probs
        )
        return torch.stack(component_log_probs, dim=-1)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        losses, categorical_entropy_per_action, magnitude_entropy_per_action = (
            self._compute_extra_losses_and_entropy_tensors(include_magnitude_entropy=True)
        )
        if not losses:
            return {}, {}
        assert categorical_entropy_per_action is not None
        assert magnitude_entropy_per_action is not None

        action_metrics_splitter = resolve_action_metrics_splitter(action_splitter)
        categorical_ent_metrics = compute_ent_metrics(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_categorical",
        )
        magnitude_ent_metrics = compute_ent_metrics(
            config=self.magnitude_ent_loss_config,
            entropy_per_action=magnitude_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name=self.magnitude_entropy_metric_name,
        )
        return losses, {**categorical_ent_metrics, **magnitude_ent_metrics}

    def compute_extra_losses_without_metrics(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> LossDict:
        _ = (agent_mask, action_splitter)
        losses, _categorical_entropy, _magnitude_entropy = self._compute_extra_losses_and_entropy_tensors(
            include_magnitude_entropy=self.magnitude_ent_scale > 0.0,
        )
        return losses

    def _compute_extra_losses_and_entropy_tensors(
            self,
            *,
            include_magnitude_entropy: bool,
    ) -> tuple[LossDict, torch.Tensor | None, torch.Tensor | None]:
        if self.ent_loss_coef <= 0.0:
            return {}, None, None

        self._assert_ready()
        assert self.weight_logits is not None
        assert self.negative_magnitude_dist is not None
        assert self.positive_magnitude_dist is not None
        weights = F.softmax(self.weight_logits, dim=-1)
        categorical_entropy_per_action = -(weights * F.log_softmax(self.weight_logits, dim=-1)).sum(dim=-1)
        categorical_ent_loss = compute_ent_loss(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
        )
        magnitude_entropy_per_action = None
        if include_magnitude_entropy:
            magnitude_entropy_per_action = (
                    weights[..., self._NEGATIVE_INDEX] * self.negative_magnitude_dist.entropy()
                    + weights[..., self._POSITIVE_INDEX] * self.positive_magnitude_dist.entropy()
            )
            magnitude_ent_loss = compute_ent_loss(
                config=self.magnitude_ent_loss_config,
                entropy_per_action=magnitude_entropy_per_action,
            )
            entropy_loss = self.ent_loss_coef * (
                    categorical_ent_loss + self.magnitude_ent_scale * magnitude_ent_loss
            )
        else:
            entropy_loss = self.ent_loss_coef * categorical_ent_loss
        return {"entropy": entropy_loss}, categorical_entropy_per_action, magnitude_entropy_per_action

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def set_magnitude_ent_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"magnitude_ent_scale must be >= 0, got {value}")
        self.magnitude_ent_scale = value

    @property
    def compile_friendly(self) -> bool:
        return True

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = {
            **super().get_hyper_parameters(),
            "epsilon": self.epsilon,
            "ent_loss_coef": self.ent_loss_coef,
            "initial_positive_prob": self.initial_positive_prob,
            "categorical_ent_loss_config": serialize_dataclass(self.categorical_ent_loss_config),
        }
        if self._ZERO_INDEX is not None:
            hyper_parameters["initial_zero_prob"] = self.initial_zero_prob
        return hyper_parameters

    def _assert_ready(self) -> None:
        if (
                self.weight_logits is None
                or self.negative_magnitude_dist is None
                or self.positive_magnitude_dist is None
        ):
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")


def inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
