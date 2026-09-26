from __future__ import annotations

import torch

from swarmbots.learn.torch_device import as_device


class TorchRunningMeanStd:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] | tuple[()] = (),
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        initial_count: float = 1e-4,
    ) -> None:
        self.mean = torch.zeros(shape, device=as_device(device), dtype=dtype)
        self.var = torch.ones(shape, device=self.mean.device, dtype=dtype)
        self.count = torch.tensor(float(initial_count), device=self.mean.device, dtype=dtype)

    def update(self, batch: torch.Tensor) -> None:
        if batch.ndim == 0:
            batch = batch.reshape(1)
        if batch.shape[0] <= 0:
            return

        batch = batch.to(device=self.mean.device, dtype=self.mean.dtype)
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(batch.shape[0]), device=self.mean.device, dtype=self.mean.dtype)
        self._update_from_moments(batch_mean=batch_mean, batch_var=batch_var, batch_count=batch_count)

    def to(self, device: torch.device | str) -> None:
        target_device = as_device(device)
        self.mean = self.mean.to(target_device)
        self.var = self.var.to(target_device)
        self.count = self.count.to(target_device)

    def _update_from_moments(
        self,
        *,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: torch.Tensor,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        var = m2 / total_count

        self.mean = mean
        self.var = var
        self.count = total_count
