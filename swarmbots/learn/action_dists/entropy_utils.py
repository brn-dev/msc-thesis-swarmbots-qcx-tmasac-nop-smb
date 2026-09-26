from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import torch

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitter
from swarmbots.learn.losses import LossMetrics
from swarmbots.learn.masking import masked_mean


AGENT_ACTIONS_DIM = -1


class AgentActionsReduction(Enum):
    SUM = 1
    MEAN = 2
    MIN = 3
    MAX = 4


class ReductionTiming(Enum):
    PRE = 1
    POST = 2


@dataclass(frozen=True)
class EntropyLossConfig:
    entropy_floor: Optional[float] = None
    max_entropy: Optional[float] = None
    agent_actions_reduction: AgentActionsReduction = AgentActionsReduction.SUM
    reduction_timing: ReductionTiming = ReductionTiming.POST  # whether reduction is applied before hinge or after hinge
    loss_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    metrics_reduction: Optional[AgentActionsReduction] = AgentActionsReduction.MEAN  # reduction for the metrics

    def resolved_entropy_floor(self) -> float | None:
        if self.entropy_floor is not None and self.max_entropy is not None:
            raise ValueError("Set either entropy_floor or deprecated max_entropy, not both.")
        return self.entropy_floor if self.entropy_floor is not None else self.max_entropy

def compute_ent_loss(
        config: EntropyLossConfig,
        entropy_per_action: torch.Tensor,
) -> torch.Tensor:
    return _compute_loss_per_agent(
        entropy_per_action=entropy_per_action,
        reduction=config.agent_actions_reduction,
        entropy_floor=config.resolved_entropy_floor(),
        reduction_timing=config.reduction_timing,
        loss_transform=config.loss_transform,
    )


def compute_ent_metrics(
        config: EntropyLossConfig,
        entropy_per_action: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
        metrics_action_splitter: ActionMetricsSplitter | None = None,
        name: str = 'ent'
) -> LossMetrics:
    with torch.no_grad():
        metrics_reduction = config.metrics_reduction or config.agent_actions_reduction
        metric_entropy_per_agent = _reduce_actions(entropy_per_action, metrics_reduction)
        _validate_agent_mask(agent_mask, expected_shape=tuple(metric_entropy_per_agent.shape))

        metrics: LossMetrics = {
            name: masked_mean(metric_entropy_per_agent, agent_mask).item()
        }

        if metrics_action_splitter is not None:
            split_tensors = metrics_action_splitter(entropy_per_action)
            for key, split_entropy in split_tensors.items():
                aligned_split_entropy = _align_split_entropy(
                    split_entropy=split_entropy,
                    entropy_per_action=entropy_per_action,
                    split_name=key,
                )
                split_entropy_per_agent = _reduce_actions(
                    values=aligned_split_entropy,
                    reduction=metrics_reduction,
                )
                metrics[f"{name}_{key}"] = masked_mean(split_entropy_per_agent, agent_mask).item()

        return metrics


def _compute_loss_per_agent(
        *,
        entropy_per_action: torch.Tensor,
        reduction: AgentActionsReduction,
        entropy_floor: float | None,
        reduction_timing: ReductionTiming,
        loss_transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    if reduction_timing == ReductionTiming.PRE:
        reduced_entropy = _reduce_actions(entropy_per_action, reduction)
        if entropy_floor is None:
            loss_per_agent = -reduced_entropy
        else:
            loss_per_agent = torch.relu(entropy_floor - reduced_entropy)

        if loss_transform is not None:
            loss_per_agent = loss_transform(loss_per_agent)

    elif reduction_timing == ReductionTiming.POST:
        if entropy_floor is None:
            loss_per_action = -entropy_per_action
        else:
            loss_per_action = torch.relu(entropy_floor - entropy_per_action)

        if loss_transform is not None:
            loss_per_action = loss_transform(loss_per_action)

        loss_per_agent = _reduce_actions(loss_per_action, reduction)

    else:
        raise ValueError(f"Unhandled reduction_timing={reduction_timing}")

    return loss_per_agent


def _reduce_actions(
        values: torch.Tensor,
        reduction: AgentActionsReduction,
) -> torch.Tensor:
    if reduction == AgentActionsReduction.SUM:
        return values.sum(dim=AGENT_ACTIONS_DIM)
    if reduction == AgentActionsReduction.MEAN:
        return values.mean(dim=AGENT_ACTIONS_DIM)
    if reduction == AgentActionsReduction.MIN:
        return values.amin(dim=AGENT_ACTIONS_DIM)
    if reduction == AgentActionsReduction.MAX:
        return values.amax(dim=AGENT_ACTIONS_DIM)
    raise ValueError(f"Unhandled reduction={reduction}")


def _align_split_entropy(
        *,
        split_entropy: torch.Tensor,
        entropy_per_action: torch.Tensor,
        split_name: str,
) -> torch.Tensor:
    if split_entropy.ndim == entropy_per_action.ndim:
        return split_entropy
    if split_entropy.ndim == entropy_per_action.ndim - 1:
        return split_entropy.unsqueeze(AGENT_ACTIONS_DIM)
    raise ValueError(
        "split entropy must return tensors with either the same rank as entropy_per_action "
        f"or one rank less. Got split {split_name!r} shape {tuple(split_entropy.shape)} "
        f"for entropy shape {tuple(entropy_per_action.shape)}."
    )


def _validate_agent_mask(
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
