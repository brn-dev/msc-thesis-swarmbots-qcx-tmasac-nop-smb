import abc
from dataclasses import dataclass, field
from typing import Any, Self

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import (
    AGENT_ACTIONS_DIM,
    BOUNDED_ACTION_HISTOGRAM,
    ActionDist,
    ActionMetricsSplitterInput,
    ActionNetInitialization,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.action_dists.action_sampling import sample_base_uniform
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig, compute_ent_loss, compute_ent_metrics
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class BoundedQuantileConfig:
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=EntropyLossConfig)


class BoundedQuantileActionDist(ActionDist, abc.ABC):
    def __init__(
            self,
            *,
            latent_dim: int,
            action_dim: int,
            parameters_per_action: int,
            action_net_initialization: ActionNetInitialization | None,
            ent_loss_coef: float,
            ent_loss_config: EntropyLossConfig | None,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=None,
            init_action_net=False,
        )
        if ent_loss_coef < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {ent_loss_coef}")
        _ = action_net_initialization
        self.parameters_per_action = parameters_per_action
        self.ent_loss_coef = ent_loss_coef
        self.ent_loss_config = ent_loss_config if ent_loss_config is not None else EntropyLossConfig()
        self.action_net = nn.Linear(latent_dim, action_dim * parameters_per_action)

        # Zero raw parameters are the exact uniform quantile map. This is stronger than
        # merely zeroing the bias: every state must initially produce the same map.
        nn.init.zeros_(self.action_net.weight)
        nn.init.zeros_(self.action_net.bias)
        self.raw_parameters: torch.Tensor | None = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raw_parameters = self.action_net(latent_pi)
        self.raw_parameters = raw_parameters.reshape(
            *latent_pi.shape[:-1],
            self.action_dim,
            self.parameters_per_action,
        ).float()
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
        raw_parameters = self._get_raw_parameters()
        u = torch.rand(raw_parameters.shape[:-1], device=raw_parameters.device, dtype=torch.float32)
        actions, _log_det = self._transform_forward_and_log_det(u)
        return actions

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        _ = previous_actions
        actions, _log_det = self._mode_and_log_det()
        return actions

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
            stratified_sample_dim: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (agent, previous_actions)
        raw_parameters = self.update_latent_features(latent_pi)._get_raw_parameters()
        if deterministic:
            actions, log_det = self._mode_and_log_det()
        else:
            u = sample_base_uniform(
                raw_parameters.shape[:-1],
                device=raw_parameters.device,
                dtype=torch.float32,
                stratified_sample_dim=stratified_sample_dim,
            )
            actions, log_det = self._transform_forward_and_log_det(u)
        if not deterministic and not use_rsample:
            actions = actions.detach()
            if torch.is_grad_enabled():
                return actions, self.log_prob(actions)
        return actions, -log_det.sum(dim=AGENT_ACTIONS_DIM)

    def get_on_policy_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=False,
        )
        return actions, self.log_prob(actions.detach(), previous_actions=previous_actions)

    @abc.abstractmethod
    def _transform_forward_and_log_det(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def _mode_and_log_det(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        return compute_action_metrics(actions, action_splitter, histogram=BOUNDED_ACTION_HISTOGRAM)

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        if self.ent_loss_coef <= 0.0:
            return {}, {}

        entropy_per_action = self._estimate_entropy_per_action()
        entropy_loss = self._entropy_loss(entropy_per_action)
        entropy_metrics = compute_ent_metrics(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
            agent_mask=agent_mask,
            metrics_action_splitter=resolve_action_metrics_splitter(action_splitter),
            name="ent",
        )
        return {"entropy": entropy_loss}, entropy_metrics

    def compute_extra_losses_without_metrics(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> LossDict:
        _ = (agent_mask, action_splitter)
        if self.ent_loss_coef <= 0.0:
            return {}
        entropy_per_action = self._estimate_entropy_per_action()
        return {"entropy": self._entropy_loss(entropy_per_action)}

    def set_ent_loss_coef(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"ent_loss_coef must be >= 0, got {value}")
        self.ent_loss_coef = value

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "parameters_per_action": self.parameters_per_action,
            "deterministic_action": "approximate_density_mode",
            "action_net_initialization": "zeros",
            "ent_loss_coef": self.ent_loss_coef,
            "ent_loss_config": serialize_dataclass(self.ent_loss_config),
            "entropy_estimator": "single_uniform_sample",
        }

    @property
    def compile_friendly(self) -> bool:
        return True

    def _get_raw_parameters(self) -> torch.Tensor:
        if self.raw_parameters is None:
            raise RuntimeError("Distribution parameters are not initialized. Call update_latent_features first.")
        return self.raw_parameters

    def _estimate_entropy_per_action(self) -> torch.Tensor:
        raw_parameters = self._get_raw_parameters()
        u = torch.rand(raw_parameters.shape[:-1], device=raw_parameters.device, dtype=torch.float32)
        _actions, log_det = self._transform_forward_and_log_det(u)
        return log_det

    def _entropy_loss(self, entropy_per_action: torch.Tensor) -> torch.Tensor:
        return self.ent_loss_coef * compute_ent_loss(
            config=self.ent_loss_config,
            entropy_per_action=entropy_per_action,
        )

    @staticmethod
    def _select_min_log_det_candidate(
            *,
            actions: torch.Tensor,
            log_det: torch.Tensor,
            quantiles: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        min_log_det = log_det.amin(dim=-1, keepdim=True)
        distance_from_median = (quantiles - 0.5).abs()
        tie_break_scores = torch.where(
            torch.isclose(log_det, min_log_det, rtol=1e-6, atol=1e-7),
            distance_from_median,
            torch.full_like(log_det, torch.inf),
        )
        best_indices = tie_break_scores.argmin(dim=-1, keepdim=True)
        return (
            actions.gather(dim=-1, index=best_indices).squeeze(-1),
            log_det.gather(dim=-1, index=best_indices).squeeze(-1),
        )
