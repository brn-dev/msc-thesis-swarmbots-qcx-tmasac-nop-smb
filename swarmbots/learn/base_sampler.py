import abc
from dataclasses import dataclass
from typing import TypeVar, Generic, Generator, Any

import torch


@dataclass(frozen=True)
class BaseSamplerConfig:
    batch_size: int


SamplesType = TypeVar('SamplesType', covariant=True)
SamplerConfigType = TypeVar('SamplerConfigType', covariant=True, bound=BaseSamplerConfig)
BatchIndices = torch.Tensor | slice


class BaseSampler(Generic[SamplesType, SamplerConfigType], abc.ABC):

    def __init__(
        self,
        config: SamplerConfigType,
        n_samples: int,
        *,
        index_device: torch.device | str | None = None,
    ):
        if config.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {config.batch_size}")
        self.config = config
        self.n_samples = n_samples
        self.index_device = None if index_device is None else torch.device(index_device)

    @abc.abstractmethod
    def _fetch_samples(self, batch_indices: BatchIndices) -> SamplesType:
        raise NotImplementedError()

    def _fetch_all_samples(self) -> SamplesType:
        return self._fetch_samples(slice(None))

    def sample(self, drop_last: bool = True) -> Generator[SamplesType, Any, None]:
        batch_size = self.config.batch_size

        if batch_size == self.n_samples:
            yield self._fetch_all_samples()
            return

        indices = torch.randperm(self.n_samples, device=self.index_device)

        for start_idx in range(0, self.n_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]

            if drop_last and len(batch_indices) < batch_size:
                continue

            yield self._fetch_samples(batch_indices)
