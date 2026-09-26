import abc
from enum import StrEnum
from typing import Callable, Optional, Self, Any

import torch
import torch.distributions as torchdist
from torch import nn

from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.summary_statistics import HistogramConfig, compute_summary_statistics

ActionNetInitialization = Callable[[nn.Linear], None]
ActionMetricsSplitter = Callable[[torch.Tensor], dict[str, torch.Tensor]]
ActionMetricsSplitterInput = ActionMetricsSplitter | list[ActionMetricsSplitter | None] | None

AGENT_ACTIONS_DIM = -1
BOUNDED_ACTION_HISTOGRAM = HistogramConfig(bins=11, low=-1.0, high=1.0)


class ActionGradientEstimator(StrEnum):
    NONE = "none"
    PATHWISE = "pathwise"
    STRAIGHT_THROUGH = "straight_through"

    @property
    def supports_actor_gradients(self) -> bool:
        return self is not ActionGradientEstimator.NONE


def validate_probability_clamp_epsilon(epsilon: float) -> None:
    if not (0.0 < epsilon < 0.5):
        raise ValueError(f"epsilon must be in (0, 0.5), got {epsilon}")



class ActionDist(nn.Module, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
            init_action_net: bool = True,
    ):
        if init_action_net and action_net_initialization is None:
            raise ValueError('If init_action_net=True, then action_net_initialization must be given!')

        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if init_action_net:
            self.action_net_initialization = action_net_initialization
            self.action_net = nn.Linear(latent_dim, action_dim)
            action_net_initialization(self.action_net)
        else:
            self.action_net_initialization = None
            self.action_net = None

        self.distribution: Optional[torchdist.Distribution] = None

    @abc.abstractmethod
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not support rsample().")

    @abc.abstractmethod
    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[LossDict, LossMetrics]:
        _ = (agent_mask, action_splitter)
        return {}, {}

    def requires_previous_actions(self) -> bool:
        return False

    @property
    def sampling_depends_on_agent(self) -> bool:
        return False

    @property
    def compile_friendly(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "latent_dim": self.latent_dim,
            "action_dim": self.action_dim,
            "requires_previous_actions": self.requires_previous_actions(),
            "compile_friendly": self.compile_friendly,
            "action_net_type": type(self.action_net).__name__ if self.action_net is not None else None,
            "action_net_initialization": (
                self.action_net_initialization.__name__
                if (self.action_net_initialization is not None and hasattr(self.action_net_initialization, "__name__"))
                else (
                    str(self.action_net_initialization)
                    if self.action_net_initialization is not None
                    else None
                )
            ),
        }

    @abc.abstractmethod
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def get_actions(
            self,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> torch.Tensor:
        if deterministic:
            return self.mode(previous_actions=previous_actions)
        if use_rsample:
            return self.rsample(agent=agent, previous_actions=previous_actions)
        return self.sample(agent=agent, previous_actions=previous_actions)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.update_latent_features(latent_pi).get_actions(
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=use_rsample,
        )
        log_probs = self.log_prob(actions, previous_actions=previous_actions)
        return actions, log_probs

    def get_on_policy_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.get_actions_with_log_probs(
            latent_pi=latent_pi,
            deterministic=deterministic,
            agent=agent,
            previous_actions=previous_actions,
            use_rsample=False,
        )

    @staticmethod
    def validate_agent_mask(
            agent_mask: torch.Tensor | None,
            *,
            expected_shape: tuple[int, ...],
    ) -> None:
        if agent_mask is None:
            return
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if tuple(agent_mask.shape) != expected_shape:
            raise ValueError(f"Expected agent_mask shape {expected_shape}, got {tuple(agent_mask.shape)}")




def resolve_action_metrics_splitter(
        action_splitter: ActionMetricsSplitterInput,
) -> ActionMetricsSplitter | None:
    if action_splitter is None or callable(action_splitter):
        return action_splitter
    if not isinstance(action_splitter, list):
        raise TypeError(
            f"action_splitter must be callable, list of callables, or None, got {type(action_splitter)}"
        )
    if len(action_splitter) == 0:
        return None
    if len(action_splitter) > 1:
        raise ValueError("Expected one splitter for a single action distribution, got multiple entries.")
    splitter = action_splitter[0]
    if splitter is not None and not callable(splitter):
        raise TypeError(f"action_splitter list entry must be callable or None, got {type(splitter)}")
    return splitter


def compute_action_metrics(
        actions: torch.Tensor,
        action_splitter: ActionMetricsSplitterInput,
        *,
        histogram: HistogramConfig | int,
) -> dict[str, Any]:
    splitter = resolve_action_metrics_splitter(action_splitter)
    metrics: dict[str, Any] = {}
    if splitter is None:
        metrics["act"] = compute_summary_statistics(actions, make_histogram=histogram)
    else:
        for key, split_actions in splitter(actions).items():
            metrics[f"act_{key}"] = compute_summary_statistics(split_actions, make_histogram=histogram)
    return metrics
