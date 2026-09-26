import math
from dataclasses import dataclass, field
from typing import Optional, Self, Any

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput, ActionNetInitialization
from swarmbots.learn.action_dists.continuous_action_dist import ContinuousActionDist
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.temporally_correlated_action_dist import TemporallyCorrelatedActionDist
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal

LogStdNetInitialization = ActionNetInitialization


@dataclass(frozen=True)
class GSDEConfig:
    base_std: float
    latent_sde_dim: int | None = None
    std_learnable: bool = True
    normalize_latent_sde_by_dim: bool = True
    epsilon: float = 1e-6
    full_std: bool = True
    sde_learn_features: bool = True
    latent_sde_net_initialization: ActionNetInitialization = init_linear_orthogonal
    log_std_clamp_range: tuple[float, float] = (-20.0, 2.0)
    ent_loss_coef: float = 0.0
    ent_loss_config: EntropyLossConfig = field(default_factory=EntropyLossConfig)


class GSDEActionDist(ContinuousActionDist, TemporallyCorrelatedActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            base_std: float,
            latent_sde_dim: int | None = None,
            std_learnable: bool = True,
            normalize_latent_sde_by_dim: bool = True,
            squash_output: bool = False,
            epsilon: float = 1e-6,
            full_std: bool = True,
            sde_learn_features: bool = True,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            latent_sde_net_initialization: ActionNetInitialization = init_linear_orthogonal,
            log_std_clamp_range: tuple[float, float] = (-20.0, 2.0),
            ent_loss_coef: float = 0.0,
            ent_loss_config: EntropyLossConfig | None = None,
    ):
        assert latent_sde_dim is None or latent_sde_dim == latent_dim or sde_learn_features

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=ent_loss_config,
        )

        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        self.std_learnable = std_learnable
        self.normalize_latent_sde_by_dim = normalize_latent_sde_by_dim
        self.full_std = full_std
        self.sde_learn_features = sde_learn_features
        self.squash_output = squash_output
        self.epsilon = epsilon
        self.log_std_clamp_range = log_std_clamp_range

        if latent_dim == self.latent_sde_dim:
            self.latent_sde_net: nn.Module = nn.Identity()
        else:
            self.latent_sde_net = nn.Linear(latent_dim, self.latent_sde_dim)
            if latent_sde_net_initialization is not None:
                latent_sde_net_initialization(self.latent_sde_net)

        log_std_cols = action_dim if self.full_std else 1
        self.log_stds = nn.Parameter(
            torch.ones((self.latent_sde_dim, log_std_cols)) * math.log(base_std),
            requires_grad=std_learnable,
        )

        self.distribution: Optional[torchdist.Normal] = None

        self._latent_sde: Optional[torch.Tensor] = None
        self._last_gaussian_actions: Optional[torch.Tensor] = None

        self._exploration_noise: Optional[torch.Tensor] = None
        self._exploration_batch_shape: Optional[tuple[int, ...]] = None

    def set_std(self, std: float) -> None:
        if std <= 0:
            raise ValueError(f"std must be > 0, got {std}")
        with torch.no_grad():
            self.log_stds[:] = math.log(std)
        if self._exploration_batch_shape is not None:
            self.reset_noise(self._exploration_batch_shape)

    def scale_std(self, multiplier: float) -> None:
        if multiplier <= 0:
            raise ValueError(f"multiplier must be > 0, got {multiplier}")
        with torch.no_grad():
            self.log_stds += math.log(multiplier)
        if self._exploration_batch_shape is not None:
            self.reset_noise(self._exploration_batch_shape)

    def reset_noise(self, batch_shape: tuple[int, ...]) -> None:
        with torch.no_grad():
            self._exploration_noise = torch.randn(
                (*batch_shape, self.latent_sde_dim, self.action_dim),
                device=self.log_stds.device,
                dtype=self.log_stds.dtype,
            )
            self._exploration_batch_shape = batch_shape

    def get_temporal_correlation_state(self) -> torch.Tensor | None:
        return self._exploration_noise

    def set_temporal_correlation_state(self, state: torch.Tensor | None) -> None:
        self._exploration_noise = state
        self._exploration_batch_shape = None if state is None else tuple(state.shape[:-2])

    def reset_noise_masked(self, mask: torch.Tensor) -> None:
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must have dtype bool, got {mask.dtype}")

        mask = self._expand_reset_mask(mask)
        expected_batch_shape = tuple(mask.shape)
        if self._exploration_noise is None or self._exploration_batch_shape != expected_batch_shape:
            self.reset_noise(expected_batch_shape)
            return

        num_resets = int(mask.sum().item())
        if num_resets == 0:
            return

        with torch.no_grad():
            mask_flat = mask.to(self._exploration_noise.device, dtype=torch.bool).reshape(-1)
            noise = torch.randn(
                (num_resets, self.latent_sde_dim, self.action_dim),
                device=self._exploration_noise.device,
                dtype=self._exploration_noise.dtype,
            )
            exploration_noise_flat = self._exploration_noise.reshape(-1, self.latent_sde_dim, self.action_dim)
            exploration_noise_flat[mask_flat] = noise

    def reset_on_ep_start(self, mask: torch.Tensor) -> None:
        self.reset_noise_masked(mask)

    def reset_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        if mask is not None:
            self.reset_noise_masked(mask)
            return
        if batch_shape is not None:
            self.reset_noise(batch_shape)

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        latent_sde = self.latent_sde_net(latent_pi)
        if not self.sde_learn_features:
            latent_sde = latent_sde.detach()
        if self.normalize_latent_sde_by_dim:
            latent_sde = latent_sde / math.sqrt(self.latent_sde_dim)
        self._latent_sde = latent_sde
        return self.update_distribution_params(action_means, self.log_stds)

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        if self._latent_sde is None:
            raise RuntimeError("update_latent_features() must be called before update_distribution_params().")

        std_matrix = self._get_std_matrix(log_stds)

        latent_sde_sq = self._latent_sde ** 2
        variance = torch.matmul(latent_sde_sq, std_matrix ** 2)
        action_std = torch.sqrt(variance + self.epsilon)

        self.distribution = torchdist.Normal(loc=means, scale=action_std)
        return self

    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._sample(agent)

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._sample(agent)

    def _sample(self, agent: int | None) -> torch.Tensor:
        if self.squash_output:
            self._last_gaussian_actions = self._sample_gaussian_actions(agent)
            return TanhBijector.forward(self._last_gaussian_actions)

        return self._sample_gaussian_actions(agent)

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        gaussian_actions = self.distribution.mean
        if self.squash_output:
            self._last_gaussian_actions = gaussian_actions
            return TanhBijector.forward(gaussian_actions)
        return gaussian_actions

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
            gaussian_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.squash_output:
            return super().log_prob(actions)

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

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ):
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
        )
        log_probs = self.log_prob(actions, gaussian_actions=self._last_gaussian_actions)
        return actions, log_probs

    def _sample_gaussian_actions(self, agent: int | None) -> torch.Tensor:
        if self._latent_sde is None or self.distribution is None:
            raise RuntimeError("update_latent_features() must be called before sampling actions.")

        if self._exploration_noise is None:
            raise RuntimeError("reset_noise() must be called before sampling GSDE actions.")

        if agent is not None:
            exploration_noise = self._exploration_noise[..., agent:agent + 1, :, :]
        else:
            exploration_noise = self._exploration_noise

        exploration_matrices = exploration_noise * self._get_std_matrix(self.log_stds)
        noise = torch.einsum("...d,...da->...a", self._latent_sde, exploration_matrices)
        return self.distribution.mean + noise

    def _get_std_matrix(self, log_stds: torch.Tensor) -> torch.Tensor:
        log_stds = torch.clamp(log_stds, *self.log_std_clamp_range)
        if self.full_std:
            return torch.exp(log_stds)
        else:
            return torch.exp(log_stds).expand(self.latent_sde_dim, self.action_dim)

    def _expand_reset_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self._exploration_batch_shape is None:
            return mask

        expected_batch_shape = self._exploration_batch_shape
        mask_shape = tuple(mask.shape)
        if mask_shape == expected_batch_shape:
            return mask
        if not self._mask_shape_is_prefix(mask_shape, expected_batch_shape):
            return mask

        expanded_mask = mask
        for _ in range(len(expected_batch_shape) - mask.ndim):
            expanded_mask = expanded_mask.unsqueeze(-1)
        return expanded_mask.expand(expected_batch_shape)

    @staticmethod
    def _mask_shape_is_prefix(mask_shape: tuple[int, ...], expected_shape: tuple[int, ...]) -> bool:
        return (
            len(mask_shape) < len(expected_shape)
            and mask_shape == expected_shape[:len(mask_shape)]
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        std_matrix = self._get_std_matrix(self.log_stds.detach())
        return {
            **super().get_hyper_parameters(),
            "latent_sde_dim": self.latent_sde_dim,
            "std_learnable": self.std_learnable,
            "normalize_latent_sde_by_dim": self.normalize_latent_sde_by_dim,
            "full_std": self.full_std,
            "sde_learn_features": self.sde_learn_features,
            "squash_output": self.squash_output,
            "epsilon": self.epsilon,
            "log_std_clamp_range": list(self.log_std_clamp_range),
            "std_mean": float(std_matrix.mean().item()),
            "std_min": float(std_matrix.min().item()),
            "std_max": float(std_matrix.max().item()),
            "has_active_noise": self._exploration_noise is not None,
        }

    @property
    def compile_friendly(self) -> bool:
        return False

    @property
    def sampling_depends_on_agent(self) -> bool:
        return True
