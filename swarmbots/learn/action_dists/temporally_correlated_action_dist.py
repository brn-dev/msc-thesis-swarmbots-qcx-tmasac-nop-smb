import abc
from typing import Any

import torch


class TemporallyCorrelatedActionDist(abc.ABC):

    @abc.abstractmethod
    def get_temporal_correlation_state(self) -> Any:
        raise NotImplementedError()

    @abc.abstractmethod
    def set_temporal_correlation_state(self, state: Any) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_on_ep_start(self, mask: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        raise NotImplementedError()
