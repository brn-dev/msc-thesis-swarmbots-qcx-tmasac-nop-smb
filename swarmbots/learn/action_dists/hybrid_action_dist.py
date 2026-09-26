from dataclasses import dataclass, field, fields, replace
from typing import Any, Mapping, Self, TypeAlias

import torch
from torch import nn
import numpy as np
from gymnasium import spaces

import swarmbots
from swarmbots.learn.action_dists.action_dist import (
    ActionDist,
    AGENT_ACTIONS_DIM,
    ActionNetInitialization,
    ActionMetricsSplitter,
    ActionMetricsSplitterInput,
    ActionGradientEstimator,
)
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.beta_action_dist import BetaActionDist, BetaConfig
from swarmbots.learn.action_dists.bounded_quantile_action_dist import BoundedQuantileActionDist
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist, BangZeroBangConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist, GSDEConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.left_middle_right_beta_action_dist import (
    LeftMiddleRightBetaActionDist,
    LeftMiddleRightBetaConfig,
)
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import (
    StickySignMagnitudeBetaActionDist,
    StickySignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sticky_left_middle_right_beta_action_dist import (
    StickyLeftMiddleRightBetaActionDist,
    StickyLeftMiddleRightBetaConfig,
)
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureActionDist, BetaMixtureConfig
from swarmbots.learn.action_dists.bernstein_quantile_action_dist import (
    BernsteinQuantileActionDist,
    BernsteinQuantileConfig,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist, PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureActionDist,
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.rational_quadratic_spline_quantile_action_dist import (
    RationalQuadraticSplineQuantileActionDist,
    RationalQuadraticSplineQuantileConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist,
    SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.sticky_bang_zero_bang_action_dist import (
    StickyBangZeroBangActionDist,
    StickyBangZeroBangConfig,
)
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist
from swarmbots.learn.action_dists.ternary_sign_magnitude_beta_action_dist import (
    TernarySignMagnitudeBetaActionDist,
    TernarySignMagnitudeBetaConfig,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass


ContinuousActionDistConfig: TypeAlias = (
    SquashedDiagGaussianConfig
    | PredictedStdConfig
    | GSDEConfig
    | BetaConfig
    | BetaMixtureConfig
    | GumbelSoftmaxSignMagnitudeBetaConfig
    | GumbelSoftmaxSignMagnitudeKumaraswamyConfig
    | TernarySignMagnitudeBetaConfig
    | ReparameterizedSignMagnitudeKumaraswamyConfig
    | ReparameterizedSquashedGaussianMixtureConfig
    | BernsteinQuantileConfig
    | RationalQuadraticSplineQuantileConfig
    | StickySignMagnitudeBetaConfig
    | StickyLeftMiddleRightBetaConfig
    | SignMagnitudeBetaConfig
    | LeftMiddleRightBetaConfig
    | BangZeroBangConfig
    | StickyBangZeroBangConfig
)

ContinuousActionDistConfigInput: TypeAlias = (
    ContinuousActionDistConfig
    | list[ContinuousActionDistConfig | None]
    | None
)


@dataclass(frozen=True)
class _ContinuousActionDistSpec:
    distribution_type: type[ActionDist]
    gradient_estimator: ActionGradientEstimator = ActionGradientEstimator.NONE
    constructor_overrides: Mapping[str, Any] = field(default_factory=dict)


_CONTINUOUS_ACTION_DIST_SPECS: dict[type, _ContinuousActionDistSpec] = {
    SquashedDiagGaussianConfig: _ContinuousActionDistSpec(
        SquashedDiagGaussianActionDist,
        ActionGradientEstimator.PATHWISE,
    ),
    PredictedStdConfig: _ContinuousActionDistSpec(
        PredictedStdActionDist,
        ActionGradientEstimator.PATHWISE,
        {"squash_output": True},
    ),
    GSDEConfig: _ContinuousActionDistSpec(
        GSDEActionDist,
        ActionGradientEstimator.PATHWISE,
        {"squash_output": True},
    ),
    BetaConfig: _ContinuousActionDistSpec(BetaActionDist, ActionGradientEstimator.PATHWISE),
    BetaMixtureConfig: _ContinuousActionDistSpec(BetaMixtureActionDist),
    GumbelSoftmaxSignMagnitudeBetaConfig: _ContinuousActionDistSpec(
        GumbelSoftmaxSignMagnitudeBetaActionDist,
        ActionGradientEstimator.STRAIGHT_THROUGH,
    ),
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig: _ContinuousActionDistSpec(
        GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
        ActionGradientEstimator.STRAIGHT_THROUGH,
    ),
    TernarySignMagnitudeBetaConfig: _ContinuousActionDistSpec(
        TernarySignMagnitudeBetaActionDist,
        ActionGradientEstimator.STRAIGHT_THROUGH,
    ),
    ReparameterizedSignMagnitudeKumaraswamyConfig: _ContinuousActionDistSpec(
        ReparameterizedSignMagnitudeKumaraswamyActionDist,
        ActionGradientEstimator.PATHWISE,
    ),
    ReparameterizedSquashedGaussianMixtureConfig: _ContinuousActionDistSpec(
        ReparameterizedSquashedGaussianMixtureActionDist,
        ActionGradientEstimator.PATHWISE,
    ),
    BernsteinQuantileConfig: _ContinuousActionDistSpec(
        BernsteinQuantileActionDist,
        ActionGradientEstimator.PATHWISE,
    ),
    RationalQuadraticSplineQuantileConfig: _ContinuousActionDistSpec(
        RationalQuadraticSplineQuantileActionDist,
        ActionGradientEstimator.PATHWISE,
    ),
    StickySignMagnitudeBetaConfig: _ContinuousActionDistSpec(StickySignMagnitudeBetaActionDist),
    SignMagnitudeBetaConfig: _ContinuousActionDistSpec(SignMagnitudeBetaActionDist),
    StickyLeftMiddleRightBetaConfig: _ContinuousActionDistSpec(StickyLeftMiddleRightBetaActionDist),
    LeftMiddleRightBetaConfig: _ContinuousActionDistSpec(LeftMiddleRightBetaActionDist),
    StickyBangZeroBangConfig: _ContinuousActionDistSpec(StickyBangZeroBangActionDist),
    BangZeroBangConfig: _ContinuousActionDistSpec(BangZeroBangActionDist),
}


def continuous_action_gradient_estimator(
        config: ContinuousActionDistConfig,
) -> ActionGradientEstimator:
    return _continuous_action_dist_spec(config).gradient_estimator


def _continuous_action_dist_spec(config: ContinuousActionDistConfig) -> _ContinuousActionDistSpec:
    for config_type in type(config).__mro__:
        spec = _CONTINUOUS_ACTION_DIST_SPECS.get(config_type)
        if spec is not None:
            return spec
    raise TypeError(f"Unsupported continuous action config type: {type(config).__name__}")


def _replace_ent_loss_coef(
        config: ContinuousActionDistConfig | None,
        value: float,
) -> ContinuousActionDistConfig | None:
    if config is None or not any(
            config_field.name == "ent_loss_coef"
            for config_field in fields(config)
    ):
        return config
    return replace(config, ent_loss_coef=value)


def continuous_config_to_dicts(
        continuous_config: ContinuousActionDistConfigInput,
) -> dict[str, Any] | list[dict[str, Any] | None] | None:
    if continuous_config is None:
        return None
    if isinstance(continuous_config, list):
        return [
            serialize_dataclass(cc) if cc is not None else None
            for cc in continuous_config
        ]
    return serialize_dataclass(continuous_config)


def bernoulli_config_to_dict(
        bernoulli_config: BernoulliConfig | None,
) -> dict[str, Any] | None:
    if bernoulli_config is None:
        return None
    return serialize_dataclass(bernoulli_config)


class HybridActionDistribution(ActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_space: HybridActionSpace,
            continuous_config: ContinuousActionDistConfigInput,
            bernoulli_config: BernoulliConfig | None = None,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
    ):
        self.action_space = action_space
        self.action_dims = action_space.agent_action_dims
        if isinstance(continuous_config, list):
            if len(continuous_config) != action_space.n_spaces:
                raise ValueError(
                    f"Expected {action_space.n_spaces} continuous configs (one per action sub-space), "
                    f"got {len(continuous_config)}."
                )
            self.continuous_configs = continuous_config
        else:
            self.continuous_configs = [continuous_config] * action_space.n_spaces
        self.bernoulli_config = bernoulli_config

        gsde_indices = [
            i for i, (sub_space, config) in enumerate(
                zip(action_space.sub_spaces, self.continuous_configs, strict=True)
            )
            if isinstance(sub_space, spaces.Box) and isinstance(config, GSDEConfig)
        ]
        self.has_gsde = len(gsde_indices) > 0

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_space.total_agent_action_dim,
            action_net_initialization=None,
            init_action_net=False,
        )

        # noinspection PyTypeChecker
        self.distributions: list[ActionDist] = nn.ModuleList([
            make_proba_distribution(
                latent_dim,
                sub_space,
                sub_space_dim,
                action_net_initialization,
                cont_conf,
                bernoulli_config=bernoulli_config,
            )
            for sub_space, sub_space_dim, cont_conf
            in zip(action_space.sub_spaces, action_space.agent_action_dims, self.continuous_configs, strict=True)
        ])

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        for dist in self.distributions:
            dist.update_latent_features(latent_pi)
        return self

    def requires_previous_actions(self) -> bool:
        return any(dist.requires_previous_actions() for dist in self.distributions)

    @property
    def sampling_depends_on_agent(self) -> bool:
        return any(dist.sampling_depends_on_agent for dist in self.distributions)

    @property
    def compile_friendly(self) -> bool:
        return all(dist.compile_friendly for dist in self.distributions)

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "action_dims": list(self.action_dims),
            "has_gsde": self.has_gsde,
            "continuous_config": continuous_config_to_dicts(self.continuous_configs),
            "continuous_gradient_estimators": [
                None if config is None else continuous_action_gradient_estimator(config).value
                for config in self.continuous_configs
            ],
            "bernoulli_config": bernoulli_config_to_dict(self.bernoulli_config),
            "sub_distributions": [dist.get_hyper_parameters() for dist in self.distributions],
        }

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._sample_sub_distributions(
            sampler_name="sample",
            agent=agent,
            previous_actions=previous_actions,
        )

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._sample_sub_distributions(
            sampler_name="rsample",
            agent=agent,
            previous_actions=previous_actions,
        )

    def _sample_sub_distributions(
            self,
            *,
            sampler_name: str,
            agent: int | None,
            previous_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        actions: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            previous_action = split_previous_actions[idx]
            sampler = getattr(dist, sampler_name)
            if dist.sampling_depends_on_agent:
                actions.append(sampler(agent=agent, previous_actions=previous_action))
            else:
                actions.append(sampler(previous_actions=previous_action))
        return torch.cat(actions, dim=AGENT_ACTIONS_DIM)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        actions: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            actions.append(dist.mode(previous_actions=split_previous_actions[idx]))
        return torch.cat(actions, dim=AGENT_ACTIONS_DIM)

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        log_prob_parts: list[torch.Tensor] = []
        for idx, dist in enumerate(self.distributions):
            log_prob_parts.append(dist.log_prob(split_actions[idx], split_previous_actions[idx]))
        return torch.stack(log_prob_parts, dim=-1).sum(dim=-1)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_actions_with_log_probs_impl(
            latent_pi=latent_pi,
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
            on_policy=False,
            stratified_sample_dim=stratified_sample_dim,
        )

    def get_on_policy_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_actions_with_log_probs_impl(
            latent_pi=latent_pi,
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=False,
            on_policy=True,
        )

    def _get_actions_with_log_probs_impl(
            self,
            *,
            latent_pi: torch.Tensor,
            deterministic: bool,
            agent: int | None,
            previous_actions: torch.Tensor | None,
            use_rsample: bool,
            on_policy: bool,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions_parts: list[torch.Tensor] = []
        log_prob_parts: list[torch.Tensor] = []
        split_previous_actions: tuple[torch.Tensor | None, ...]
        if previous_actions is None:
            split_previous_actions = (None,) * len(self.distributions)
        else:
            split_previous_actions = torch.split(previous_actions, self.action_dims, dim=AGENT_ACTIONS_DIM)

        for idx, dist in enumerate(self.distributions):
            previous_action = split_previous_actions[idx]
            if on_policy:
                action_part, log_prob_part = dist.get_on_policy_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    agent=agent if dist.sampling_depends_on_agent else None,
                    previous_actions=previous_action,
                )
            elif stratified_sample_dim is not None and isinstance(dist, BoundedQuantileActionDist):
                action_part, log_prob_part = dist.get_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    previous_actions=previous_action,
                    use_rsample=use_rsample,
                    stratified_sample_dim=stratified_sample_dim,
                )
            elif dist.sampling_depends_on_agent:
                action_part, log_prob_part = dist.get_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    agent=agent,
                    previous_actions=previous_action,
                    use_rsample=use_rsample,
                )
            else:
                action_part, log_prob_part = dist.get_actions_with_log_probs(
                    latent_pi=latent_pi,
                    deterministic=deterministic,
                    previous_actions=previous_action,
                    use_rsample=use_rsample,
                )
            actions_parts.append(action_part)
            log_prob_parts.append(log_prob_part)

        actions = torch.cat(actions_parts, dim=AGENT_ACTIONS_DIM)
        log_probs = torch.stack(log_prob_parts, dim=-1).sum(dim=-1)
        return actions, log_probs

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        losses: LossDict = {}
        metrics: LossMetrics = {}
        splitters = self._resolve_action_splitters(action_splitter)
        for i, dist in enumerate(self.distributions):
            dist_losses, dist_metrics = dist.compute_extra_losses(
                agent_mask=agent_mask,
                action_splitter=splitters[i],
            )
            prefix = f"act{i}_"
            losses.update(self._prefix_named_values(dist_losses, prefix=prefix))
            metrics.update(self._prefix_named_values(dist_metrics, prefix=prefix))
        return losses, metrics

    def compute_extra_losses_without_metrics(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> LossDict:
        losses: LossDict = {}
        splitters = self._resolve_action_splitters(action_splitter)
        for i, dist in enumerate(self.distributions):
            compute_losses_without_metrics = getattr(dist, "compute_extra_losses_without_metrics", None)
            if callable(compute_losses_without_metrics):
                dist_losses = compute_losses_without_metrics(
                    agent_mask=agent_mask,
                    action_splitter=splitters[i],
                )
            else:
                dist_losses, _dist_metrics = dist.compute_extra_losses(
                    agent_mask=agent_mask,
                    action_splitter=splitters[i],
                )
            losses.update(self._prefix_named_values(dist_losses, prefix=f"act{i}_"))
        return losses

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        splitters = self._resolve_action_splitters(action_splitter)
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)

        metrics: dict[str, Any] = {}
        for i, (dist, dist_actions, dist_splitter) in enumerate(
                zip(self.distributions, split_actions, splitters, strict=True)
        ):
            for metric_name, metric_value in dist.get_metrics(dist_actions, dist_splitter).items():
                prefixed_name = self._prefix_distribution_metric_name(metric_name=metric_name, index=i)
                metrics[prefixed_name] = metric_value
        return metrics

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        for dist in self.distributions:
            if isinstance(dist, TemporallyCorrelatedActionDist):
                dist.reset_on_ep_start(mask)

    def get_temporal_correlation_state(self) -> tuple[Any, ...]:
        return tuple(
            dist.get_temporal_correlation_state()
            if isinstance(dist, TemporallyCorrelatedActionDist)
            else None
            for dist in self.distributions
        )

    def set_temporal_correlation_state(self, state: tuple[Any, ...]) -> None:
        if len(state) != len(self.distributions):
            raise ValueError(
                f"Expected temporal-correlation state for {len(self.distributions)} distributions, got {len(state)}"
            )
        for dist, dist_state in zip(self.distributions, state, strict=True):
            if isinstance(dist, TemporallyCorrelatedActionDist):
                dist.set_temporal_correlation_state(dist_state)
            elif dist_state is not None:
                raise ValueError("Received temporal-correlation state for a non-temporally-correlated distribution.")

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        for dist in self.distributions:
            if isinstance(dist, TemporallyCorrelatedActionDist):
                dist.reset_on_step(mask=mask, batch_shape=batch_shape)

    def set_std(self, std: float) -> None:
        for dist in self.distributions:
            set_std = getattr(dist, "set_std", None)
            if callable(set_std):
                set_std(std)

    def scale_std(self, multiplier: float) -> None:
        for dist in self.distributions:
            scale_std = getattr(dist, "scale_std", None)
            if callable(scale_std):
                scale_std(multiplier)

    def set_all_ent_loss_coefs(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        for dist in self.distributions:
            set_ent_loss_coef = getattr(dist, "set_ent_loss_coef", None)
            if callable(set_ent_loss_coef):
                set_ent_loss_coef(value)
        for idx, config in enumerate(self.continuous_configs):
            self.continuous_configs[idx] = _replace_ent_loss_coef(config, value)
        if self.bernoulli_config is not None:
            self.bernoulli_config = replace(self.bernoulli_config, ent_loss_coef=value)

    def set_sub_ent_loss_coef(self, sub_dist_idx: int, value: float) -> None:
        if value < 0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        if not (0 <= sub_dist_idx < len(self.distributions)):
            raise IndexError(
                f"sub_dist_idx out of range [0, {len(self.distributions) - 1}], got {sub_dist_idx}"
            )

        dist = self.distributions[sub_dist_idx]
        set_ent_loss_coef = getattr(dist, "set_ent_loss_coef", None)
        if not callable(set_ent_loss_coef):
            raise ValueError(f"Action sub-dist {sub_dist_idx} does not support entropy coefficient updates")
        set_ent_loss_coef(value)

        config = self.continuous_configs[sub_dist_idx]
        self.continuous_configs[sub_dist_idx] = _replace_ent_loss_coef(config, value)

    def set_all_stickiness(self, value: float) -> None:
        for dist in self.distributions:
            if not isinstance(dist, StickyActionDist):
                continue
            dist.set_stickiness(value)

    def set_sub_stickiness(self, sub_dist_idx: int, value: float) -> None:
        if not (0 <= sub_dist_idx < len(self.distributions)):
            raise IndexError(
                f"sub_dist_idx out of range [0, {len(self.distributions) - 1}], got {sub_dist_idx}"
            )

        dist = self.distributions[sub_dist_idx]
        if not isinstance(dist, StickyActionDist):
            raise ValueError(f"Action sub-dist {sub_dist_idx} does not support stickiness updates")

        dist.set_stickiness(value)

    def set_all_gumbel_temperatures(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"gumbel temperature must be > 0, got {value}")
        for dist in self.distributions:
            setter = getattr(dist, "set_gumbel_temperature", None)
            if callable(setter):
                setter(value)
        for idx, config in enumerate(self.continuous_configs):
            if isinstance(config, (GumbelSoftmaxSignMagnitudeBetaConfig,
                                   GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
                                   TernarySignMagnitudeBetaConfig)):
                self.continuous_configs[idx] = replace(config, gumbel_temperature=value)

    def set_sub_gumbel_temperature(self, sub_dist_idx: int, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"gumbel temperature must be > 0, got {value}")
        if not (0 <= sub_dist_idx < len(self.distributions)):
            raise IndexError(
                f"sub_dist_idx out of range [0, {len(self.distributions) - 1}], got {sub_dist_idx}"
            )
        setter = getattr(self.distributions[sub_dist_idx], "set_gumbel_temperature", None)
        if not callable(setter):
            raise ValueError(f"Action sub-dist {sub_dist_idx} does not support Gumbel temperature updates")
        setter(value)
        config = self.continuous_configs[sub_dist_idx]
        if isinstance(config, (GumbelSoftmaxSignMagnitudeBetaConfig,
                               GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
                               TernarySignMagnitudeBetaConfig)):
            self.continuous_configs[sub_dist_idx] = replace(config, gumbel_temperature=value)

    @staticmethod
    def _prefix_named_values(
            values: Mapping[str, Any],
            *,
            prefix: str,
    ) -> dict[str, Any]:
        return {f"{prefix}{name}": value for name, value in values.items()}

    def _resolve_action_splitters(
            self,
            action_splitter: ActionMetricsSplitterInput,
    ) -> list[ActionMetricsSplitter | None]:
        if action_splitter is None:
            return [None] * len(self.distributions)
        if callable(action_splitter):
            return [action_splitter] * len(self.distributions)
        if not isinstance(action_splitter, list):
            raise TypeError(
                f"action_splitter must be callable, list of callables, or None, got {type(action_splitter)}"
            )
        if len(action_splitter) != len(self.distributions):
            raise ValueError(
                f"Expected {len(self.distributions)} action splitters, got {len(action_splitter)}"
            )
        for splitter in action_splitter:
            if splitter is not None and not callable(splitter):
                raise TypeError(f"action_splitter list entries must be callable or None, got {type(splitter)}")
        return action_splitter

    @staticmethod
    def _prefix_distribution_metric_name(metric_name: str, index: int) -> str:
        if metric_name == "act":
            return f"act{index}"
        if metric_name.startswith("act_"):
            return f"act{index}_{metric_name.removeprefix('act_')}"
        if metric_name == "std":
            return f"std{index}"
        if metric_name.startswith("std_"):
            return f"std{index}_{metric_name.removeprefix('std_')}"
        return f"act{index}_{metric_name}"


def _make_continuous_action_distribution(
        *,
        config: ContinuousActionDistConfig,
        latent_dim: int,
        action_dim: int,
        action_net_initialization: ActionNetInitialization,
) -> ActionDist:
    spec = _continuous_action_dist_spec(config)
    constructor_kwargs = {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
    }
    constructor_kwargs.update(spec.constructor_overrides)
    return spec.distribution_type(
        latent_dim=latent_dim,
        action_dim=action_dim,
        action_net_initialization=action_net_initialization,
        **constructor_kwargs,
    )


def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
        action_space_dim: int,
        action_net_initialization: ActionNetInitialization,
        continuous_config: ContinuousActionDistConfig | None,
        bernoulli_config: BernoulliConfig | None = None,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        _assert_unit_box_range(action_space)
        if continuous_config is None:
            supported_configs = " | ".join(
                config_type.__name__
                for config_type in _CONTINUOUS_ACTION_DIST_SPECS
            )
            raise ValueError(
                f"Supply a ContinuousActionDistConfig ({supported_configs}) for continuous actions."
            )

        return _make_continuous_action_distribution(
            config=continuous_config,
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            action_net_initialization=action_net_initialization,
        )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space_dim,
            initial_prob=bernoulli_config.initial_prob if bernoulli_config is not None else None,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=bernoulli_config.ent_loss_coef if bernoulli_config is not None else 0.0,
            ent_loss_config=bernoulli_config.ent_loss_config if bernoulli_config is not None else None,
        )
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")


def _assert_unit_box_range(space: spaces.Box, atol: float = 1e-6) -> None:
    low = np.asarray(space.low, dtype=np.float64)
    high = np.asarray(space.high, dtype=np.float64)

    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError(f"Box action bounds must be finite, got low/high with non-finite values: {space}")

    if not (np.allclose(low, -1.0, atol=atol) and np.allclose(high, 1.0, atol=atol)):
        raise ValueError(
            "Box action space must have bounds low=-1 and high=1 for tanh-squashed policy output. "
            f"Got low in [{low.min():.6g}, {low.max():.6g}], high in [{high.min():.6g}, {high.max():.6g}]. "
        )
