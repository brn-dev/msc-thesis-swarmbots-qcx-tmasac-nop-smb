from typing import Generic, Protocol, TypeVar

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor
from swarmbots.learn.base_sampler import BaseSamplerConfig


class BaseWMSamples(Protocol):
    wm_actions: torch.Tensor
    next_local_obs: torch.Tensor
    wm_target_time_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: MaybeTensor
    wm_loss_agent_mask: MaybeTensor


WMSamplesType = TypeVar("WMSamplesType", bound=BaseWMSamples, covariant=True)
WMSamplerConfigType = TypeVar("WMSamplerConfigType", covariant=True, bound=BaseSamplerConfig)


class BaseWMSampler(Generic[WMSamplesType, WMSamplerConfigType]):
    pass
