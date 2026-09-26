import math
from dataclasses import dataclass, field
from typing import Any, Optional, Self

import torch
from torch import distributions as torchdist, nn
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


def _default_entropy_loss_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


@dataclass(frozen=True)
class LeftMiddleRightBetaConfig:
    eps_c: float
    initial_middle_prob: float | None = None
    epsilon: float = 1e-6
    left_alpha: float = 1.0 + math.log(2.0)
    left_beta: float = 1.0 + math.log(2.0)
    right_alpha: float = 1.0 + math.log(2.0)
    right_beta: float = 1.0 + math.log(2.0)
    ent_loss_coef: float = 0.0
    beta_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)
    beta_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)


class LeftMiddleRightBetaActionDist(ActionDist):
    _N_MIXTURE_COMPONENTS = 3
    _OUTPUTS_PER_ACTION = 7

    _LEFT_INDEX = 0
    _MIDDLE_INDEX = 1
    _RIGHT_INDEX = 2

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            eps_c: float,
            action_net_initialization: ActionNetInitialization | None,
            initial_middle_prob: float | None = None,
            epsilon: float = 1e-6,
            left_alpha: float = 1.0 + math.log(2.0),
            left_beta: float = 1.0 + math.log(2.0),
            right_alpha: float = 1.0 + math.log(2.0),
            right_beta: float = 1.0 + math.log(2.0),
            ent_loss_coef: float = 0.0,
            beta_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            beta_ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        if not (0.0 < eps_c < 1.0):
            raise ValueError(f"eps_c must be in (0, 1), got {eps_c}")
        validate_probability_clamp_epsilon(epsilon)
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        if beta_ent_scale < 0.0:
            raise ValueError(f"beta_ent_scale must be >= 0, got {beta_ent_scale}")
        if initial_middle_prob is not None and not (0.0 < initial_middle_prob < 1.0):
            raise ValueError(
                f"initial_middle_prob must be strictly between 0 and 1, got {initial_middle_prob}"
            )

        for name, value in (
                ("left_alpha", left_alpha),
                ("left_beta", left_beta),
                ("right_alpha", right_alpha),
                ("right_beta", right_beta),
        ):
            if value <= 1.0:
                raise ValueError(f"{name} must be > 1.0, got {value}")

        self.eps_c = eps_c
        self.epsilon = epsilon
        self.ent_loss_coef = ent_loss_coef
        self.beta_ent_scale = beta_ent_scale
        self.categorical_ent_loss_config = (
            categorical_ent_loss_config
            if categorical_ent_loss_config is not None
            else _default_entropy_loss_config()
        )
        self.beta_ent_loss_config = (
            beta_ent_loss_config
            if beta_ent_loss_config is not None
            else _default_entropy_loss_config()
        )
        self.initial_middle_prob = initial_middle_prob
        self.interval_width = 1.0 - eps_c
        self.log_interval_jacobian = math.log(1.0 / self.interval_width)
        self.middle_log_density = -math.log(2.0 * eps_c)

        self.output_net = nn.Linear(latent_dim, action_dim * self._OUTPUTS_PER_ACTION)
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self._OUTPUTS_PER_ACTION)
            if initial_middle_prob is not None:
                outer_prob = (1.0 - initial_middle_prob) / 2.0
                bias[:, self._LEFT_INDEX] = math.log(outer_prob)
                bias[:, self._MIDDLE_INDEX] = math.log(initial_middle_prob)
                bias[:, self._RIGHT_INDEX] = math.log(outer_prob)
            bias[:, 3] = _inverse_softplus(left_alpha - 1.0)
            bias[:, 4] = _inverse_softplus(left_beta - 1.0)
            bias[:, 5] = _inverse_softplus(right_alpha - 1.0)
            bias[:, 6] = _inverse_softplus(right_beta - 1.0)

        self.weight_logits: Optional[torch.Tensor] = None
        self.categorical_dist: Optional[torchdist.Categorical] = None
        self.left_beta_dist: Optional[torchdist.Beta] = None
        self.right_beta_dist: Optional[torchdist.Beta] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(*latent_pi.shape[:-1], self.action_dim, self._OUTPUTS_PER_ACTION)
        self.weight_logits = raw[..., :self._N_MIXTURE_COMPONENTS]
        self.categorical_dist = torchdist.Categorical(
            logits=self.weight_logits,
            validate_args=False,
        )

        left_alpha = 1.0 + F.softplus(raw[..., 3])
        left_beta = 1.0 + F.softplus(raw[..., 4])
        right_alpha = 1.0 + F.softplus(raw[..., 5])
        right_beta = 1.0 + F.softplus(raw[..., 6])
        self.left_beta_dist = torchdist.Beta(
            concentration1=left_alpha,
            concentration0=left_beta,
            validate_args=False,
        )
        self.right_beta_dist = torchdist.Beta(
            concentration1=right_alpha,
            concentration0=right_beta,
            validate_args=False,
        )
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        component_indices = self.categorical_dist.sample()
        left_01 = self.left_beta_dist.sample()
        right_01 = self.right_beta_dist.sample()
        middle_actions = torch.empty_like(left_01).uniform_(-self.eps_c, self.eps_c)

        left_actions = -1.0 + self.interval_width * left_01
        right_actions = self.eps_c + self.interval_width * right_01

        sampled_actions = middle_actions
        sampled_actions = torch.where(component_indices == self._LEFT_INDEX, left_actions, sampled_actions)
        sampled_actions = torch.where(component_indices == self._RIGHT_INDEX, right_actions, sampled_actions)
        return sampled_actions

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        weights = F.softmax(self.weight_logits, dim=-1)

        left_mean = -1.0 + self.interval_width * self.left_beta_dist.mean
        right_mean = self.eps_c + self.interval_width * self.right_beta_dist.mean
        return (
                weights[..., self._LEFT_INDEX] * left_mean
                + weights[..., self._RIGHT_INDEX] * right_mean
        )

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        left_mask = actions < -self.eps_c
        right_mask = actions > self.eps_c

        left_01 = ((actions + 1.0) / self.interval_width).clamp(self.epsilon, 1.0 - self.epsilon)
        right_01 = ((actions - self.eps_c) / self.interval_width).clamp(self.epsilon, 1.0 - self.epsilon)

        left_log_prob = (
                log_weights[..., self._LEFT_INDEX]
                + self.left_beta_dist.log_prob(left_01)
                + self.log_interval_jacobian
        )
        middle_log_prob = log_weights[..., self._MIDDLE_INDEX] + self.middle_log_density
        right_log_prob = (
                log_weights[..., self._RIGHT_INDEX]
                + self.right_beta_dist.log_prob(right_01)
                + self.log_interval_jacobian
        )

        log_prob_per_action = torch.where(
            left_mask,
            left_log_prob,
            torch.where(right_mask, right_log_prob, middle_log_prob),
        )
        return log_prob_per_action.sum(dim=AGENT_ACTIONS_DIM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0.0:
            return {}, {}

        weights = F.softmax(self.weight_logits, dim=-1)

        categorical_entropy_per_action = self.categorical_dist.entropy()
        weighted_beta_entropy_per_action = (
                weights[..., self._LEFT_INDEX] * self.left_beta_dist.entropy()
                + weights[..., self._RIGHT_INDEX] * self.right_beta_dist.entropy()
        )
        categorical_ent_loss = compute_ent_loss(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
        )
        beta_ent_loss = compute_ent_loss(
            config=self.beta_ent_loss_config,
            entropy_per_action=weighted_beta_entropy_per_action,
        )
        entropy_loss = self.ent_loss_coef * (categorical_ent_loss + self.beta_ent_scale * beta_ent_loss)

        action_metrics_splitter = resolve_action_metrics_splitter(action_splitter)

        categorical_ent_metrics = compute_ent_metrics(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_categorical",
        )
        beta_ent_metrics = compute_ent_metrics(
            config=self.beta_ent_loss_config,
            entropy_per_action=weighted_beta_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_beta",
        )
        metrics: LossMetrics = {
            **categorical_ent_metrics,
            **beta_ent_metrics,
        }
        return {"entropy": entropy_loss}, metrics

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def set_beta_ent_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"beta_ent_scale must be >= 0, got {value}")
        self.beta_ent_scale = value

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
        return {
            **super().get_hyper_parameters(),
            "eps_c": self.eps_c,
            "epsilon": self.epsilon,
            "ent_loss_coef": self.ent_loss_coef,
            "beta_ent_scale": self.beta_ent_scale,
            "initial_middle_prob": self.initial_middle_prob,
            "interval_width": self.interval_width,
            "categorical_ent_loss_config": serialize_dataclass(self.categorical_ent_loss_config),
            "beta_ent_loss_config": serialize_dataclass(self.beta_ent_loss_config),
        }


def _inverse_softplus(value: float) -> float:
    return value + math.log(-math.expm1(-value))
