import math
from dataclasses import dataclass, field
from typing import Any, Optional, Self

import torch
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


def _default_entropy_loss_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


@dataclass(frozen=True)
class ReparameterizedSquashedGaussianMixtureConfig:
    initial_action_modes: tuple[float, ...] = (-0.5, 0.5)
    initial_stds: tuple[float, ...] = (0.25, 0.25)
    initial_weights: tuple[float, ...] | None = None
    epsilon: float = 1e-6
    inverse_cdf_iterations: int = 48
    log_std_clamp_range: tuple[float, float] = (-8.0, 2.0)
    ent_loss_coef: float = 0.0
    gaussian_ent_scale: float = 1.0
    categorical_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)
    gaussian_ent_loss_config: EntropyLossConfig = field(default_factory=_default_entropy_loss_config)


class ReparameterizedSquashedGaussianMixtureActionDist(ActionDist):
    _OUTPUTS_PER_COMPONENT = 3

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            initial_action_modes: tuple[float, ...] = (-0.5, 0.5),
            initial_stds: tuple[float, ...] = (0.25, 0.25),
            initial_weights: tuple[float, ...] | None = None,
            epsilon: float = 1e-6,
            inverse_cdf_iterations: int = 48,
            log_std_clamp_range: tuple[float, float] = (-8.0, 2.0),
            ent_loss_coef: float = 0.0,
            gaussian_ent_scale: float = 1.0,
            categorical_ent_loss_config: EntropyLossConfig | None = None,
            gaussian_ent_loss_config: EntropyLossConfig | None = None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            init_action_net=False,
        )

        self.num_components = len(initial_action_modes)
        self._validate_init_params(
            initial_action_modes=initial_action_modes,
            initial_stds=initial_stds,
            initial_weights=initial_weights,
            epsilon=epsilon,
            inverse_cdf_iterations=inverse_cdf_iterations,
            log_std_clamp_range=log_std_clamp_range,
            ent_loss_coef=ent_loss_coef,
            gaussian_ent_scale=gaussian_ent_scale,
        )

        self.initial_action_modes = initial_action_modes
        self.initial_stds = initial_stds
        self.initial_weights = initial_weights
        self.epsilon = epsilon
        self.inverse_cdf_iterations = inverse_cdf_iterations
        self.log_std_clamp_range = log_std_clamp_range
        self.ent_loss_coef = ent_loss_coef
        self.gaussian_ent_scale = gaussian_ent_scale
        self.categorical_ent_loss_config = (
            categorical_ent_loss_config if categorical_ent_loss_config is not None else _default_entropy_loss_config()
        )
        self.gaussian_ent_loss_config = (
            gaussian_ent_loss_config if gaussian_ent_loss_config is not None else _default_entropy_loss_config()
        )

        self.output_net = nn.Linear(
            latent_dim,
            action_dim * self.num_components * self._OUTPUTS_PER_COMPONENT,
        )
        if action_net_initialization is not None:
            action_net_initialization(self.output_net)

        with torch.no_grad():
            bias = self.output_net.bias.view(action_dim, self.num_components, self._OUTPUTS_PER_COMPONENT)
            if initial_weights is not None:
                weights = torch.as_tensor(initial_weights, dtype=bias.dtype, device=bias.device)
                bias[:, :, 0] = weights.log()
            initial_means = torch.as_tensor(
                [_atanh_scalar(action_mode) for action_mode in initial_action_modes],
                dtype=bias.dtype,
                device=bias.device,
            )
            initial_log_stds = torch.as_tensor(
                [math.log(std) for std in initial_stds],
                dtype=bias.dtype,
                device=bias.device,
            )
            bias[:, :, 1] = initial_means
            bias[:, :, 2] = initial_log_stds

        self.weight_logits: Optional[torch.Tensor] = None
        self.means: Optional[torch.Tensor] = None
        self.log_stds: Optional[torch.Tensor] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw = self.output_net(latent_pi).view(
            *latent_pi.shape[:-1],
            self.action_dim,
            self.num_components,
            self._OUTPUTS_PER_COMPONENT,
        )
        self.weight_logits = raw[..., 0]
        self.means = raw[..., 1]
        self.log_stds = raw[..., 2].clamp(*self.log_std_clamp_range)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(agent=agent, previous_actions=previous_actions)

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (agent, previous_actions)
        self._assert_ready()

        u = torch.rand_like(self.weight_logits[..., 0]).clamp(self.epsilon, 1.0 - self.epsilon)
        return _SquashedGaussianMixtureICDF.apply(
            u,
            self.means,
            self.log_stds,
            self.weight_logits,
            self.epsilon,
            self.inverse_cdf_iterations,
        )

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        _ = previous_actions
        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)
        return (weights * torch.tanh(self.means)).sum(dim=-1)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        self._assert_ready()

        actions = actions.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        z = _atanh(actions).unsqueeze(-1)
        component_log_probs = _normal_log_prob(z, self.means, self.log_stds)
        log_weights = F.log_softmax(self.weight_logits, dim=-1)
        log_prob_z = torch.logsumexp(log_weights + component_log_probs, dim=-1)
        log_det_inverse = -torch.log1p(-actions.square() + self.epsilon)
        return (log_prob_z + log_det_inverse).sum(dim=AGENT_ACTIONS_DIM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0.0:
            return {}, {}

        self._assert_ready()
        weights = F.softmax(self.weight_logits, dim=-1)

        categorical_entropy_per_action = -(weights * F.log_softmax(self.weight_logits, dim=-1)).sum(dim=-1)
        gaussian_entropy_per_component = 0.5 * (1.0 + math.log(2.0 * math.pi)) + self.log_stds
        weighted_gaussian_entropy_per_action = (weights * gaussian_entropy_per_component).sum(dim=-1)
        categorical_ent_loss = compute_ent_loss(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
        )
        gaussian_ent_loss = compute_ent_loss(
            config=self.gaussian_ent_loss_config,
            entropy_per_action=weighted_gaussian_entropy_per_action,
        )
        entropy_loss = self.ent_loss_coef * (categorical_ent_loss + self.gaussian_ent_scale * gaussian_ent_loss)

        action_metrics_splitter = resolve_action_metrics_splitter(action_splitter)
        categorical_ent_metrics = compute_ent_metrics(
            config=self.categorical_ent_loss_config,
            entropy_per_action=categorical_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_categorical",
        )
        gaussian_ent_metrics = compute_ent_metrics(
            config=self.gaussian_ent_loss_config,
            entropy_per_action=weighted_gaussian_entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=action_metrics_splitter,
            name="ent_gaussian",
        )
        return {"entropy": entropy_loss}, {**categorical_ent_metrics, **gaussian_ent_metrics}

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def set_gaussian_ent_scale(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"gaussian_ent_scale must be >= 0, got {value}")
        self.gaussian_ent_scale = value

    @property
    def compile_friendly(self) -> bool:
        return False

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "num_components": self.num_components,
            "initial_action_modes": list(self.initial_action_modes),
            "initial_stds": list(self.initial_stds),
            "initial_weights": list(self.initial_weights) if self.initial_weights is not None else None,
            "epsilon": self.epsilon,
            "inverse_cdf_iterations": self.inverse_cdf_iterations,
            "log_std_clamp_range": list(self.log_std_clamp_range),
            "ent_loss_coef": self.ent_loss_coef,
            "gaussian_ent_scale": self.gaussian_ent_scale,
            "categorical_ent_loss_config": serialize_dataclass(self.categorical_ent_loss_config),
            "gaussian_ent_loss_config": serialize_dataclass(self.gaussian_ent_loss_config),
        }

    def _assert_ready(self) -> None:
        if self.weight_logits is None or self.means is None or self.log_stds is None:
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")

    def _validate_init_params(
            self,
            *,
            initial_action_modes: tuple[float, ...],
            initial_stds: tuple[float, ...],
            initial_weights: tuple[float, ...] | None,
            epsilon: float,
            inverse_cdf_iterations: int,
            log_std_clamp_range: tuple[float, float],
            ent_loss_coef: float,
            gaussian_ent_scale: float,
    ) -> None:
        if self.num_components < 2:
            raise ValueError(f"Expected at least 2 components, got {self.num_components}")
        if len(initial_stds) != self.num_components:
            raise ValueError(
                f"Expected {self.num_components} initial_stds values, got {len(initial_stds)}: {initial_stds}"
            )
        if initial_weights is not None and len(initial_weights) != self.num_components:
            raise ValueError(
                f"Expected {self.num_components} initial_weights values, got {len(initial_weights)}: {initial_weights}"
            )
        for action_mode in initial_action_modes:
            if not (-1.0 < action_mode < 1.0):
                raise ValueError(f"initial_action_modes must be in (-1, 1), got {initial_action_modes}")
        for std in initial_stds:
            if std <= 0.0:
                raise ValueError(f"initial_stds must be > 0, got {initial_stds}")
        if initial_weights is not None:
            for weight in initial_weights:
                if weight <= 0.0:
                    raise ValueError(f"initial_weights must be > 0, got {initial_weights}")
        validate_probability_clamp_epsilon(epsilon)
        if inverse_cdf_iterations < 1:
            raise ValueError(f"inverse_cdf_iterations must be >= 1, got {inverse_cdf_iterations}")
        if len(log_std_clamp_range) != 2 or log_std_clamp_range[0] >= log_std_clamp_range[1]:
            raise ValueError(f"log_std_clamp_range must be (min, max), got {log_std_clamp_range}")
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        if gaussian_ent_scale < 0.0:
            raise ValueError(f"gaussian_ent_scale must be >= 0, got {gaussian_ent_scale}")


class _SquashedGaussianMixtureICDF(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx: Any,
            u: torch.Tensor,
            means: torch.Tensor,
            log_stds: torch.Tensor,
            logits: torch.Tensor,
            epsilon: float,
            iterations: int,
    ) -> torch.Tensor:
        with torch.no_grad():
            stds = log_stds.exp()
            base_bound = _atanh_scalar(1.0 - epsilon)
            lo = torch.minimum(
                means.amin(dim=-1) - 8.0 * stds.amax(dim=-1),
                torch.full_like(u, -base_bound),
            )
            hi = torch.maximum(
                means.amax(dim=-1) + 8.0 * stds.amax(dim=-1),
                torch.full_like(u, base_bound),
            )

            for _ in range(iterations):
                mid = 0.5 * (lo + hi)
                cdf_mid = _gaussian_mixture_cdf_z(mid, means, log_stds, logits)
                lo = torch.where(cdf_mid < u, mid, lo)
                hi = torch.where(cdf_mid >= u, mid, hi)

            z = 0.5 * (lo + hi)
            actions = torch.tanh(z).clamp(-1.0 + epsilon, 1.0 - epsilon)

        ctx.epsilon = epsilon
        ctx.save_for_backward(actions, means, log_stds, logits)
        return actions

    @staticmethod
    def backward(
            ctx: Any,
            grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, None, None]:
        actions, means, log_stds, logits = ctx.saved_tensors
        epsilon = ctx.epsilon

        z = _atanh(actions).unsqueeze(-1)
        stds = log_stds.exp()
        weights = F.softmax(logits, dim=-1)
        t = (z - means) / stds
        component_cdfs = _normal_cdf(t)
        component_pdfs = _standard_normal_pdf(t)
        mixture_cdf = (weights * component_cdfs).sum(dim=-1)
        density_z = (weights * component_pdfs / stds).sum(dim=-1).clamp_min(epsilon)
        dx_dz = (1.0 - actions.square()).clamp_min(epsilon)
        common = grad_output * dx_dz / density_z

        grad_u = common
        grad_means = common.unsqueeze(-1) * weights * component_pdfs / stds
        grad_log_stds = common.unsqueeze(-1) * weights * component_pdfs * t
        grad_logits = -common.unsqueeze(-1) * weights * (component_cdfs - mixture_cdf.unsqueeze(-1))
        return grad_u, grad_means, grad_log_stds, grad_logits, None, None


def _gaussian_mixture_cdf_z(
        z: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        logits: torch.Tensor,
) -> torch.Tensor:
    t = (z.unsqueeze(-1) - means) / log_stds.exp()
    weights = F.softmax(logits, dim=-1)
    return (weights * _normal_cdf(t)).sum(dim=-1)


def _normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value / math.sqrt(2.0)))


def _standard_normal_pdf(value: torch.Tensor) -> torch.Tensor:
    return torch.exp(-0.5 * value.square()) / math.sqrt(2.0 * math.pi)


def _normal_log_prob(value: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    normalized = (value - mean) / log_std.exp()
    return -0.5 * normalized.square() - log_std - 0.5 * math.log(2.0 * math.pi)


def _atanh(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def _atanh_scalar(value: float) -> float:
    return 0.5 * (math.log1p(value) - math.log1p(-value))
