from __future__ import annotations

from typing import Any

import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.tensor_conversion import to_torch_tensor
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class TorchNormalizeRewardWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.return_rms = TorchRunningMeanStd(shape=(), device=self.device)
        self.returns = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        self.prev_dones = torch.zeros((self._n_envs,), device=self.device, dtype=torch.bool)
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool) -> None:
        self._update_running_mean = setting

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        reset_mask = self._extract_reset_mask(kwargs)
        if reset_mask is None:
            self.returns.zero_()
            self.prev_dones.zero_()
        else:
            self.returns[reset_mask] = 0.0
            self.prev_dones[reset_mask] = False
        return obs, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.returns[self.prev_dones] = 0.0
        self.returns.mul_(self.gamma).add_(rewards.to(torch.float64))

        if self._update_running_mean:
            self.return_rms.update(self.returns)

        rewards = rewards / torch.sqrt(self.return_rms.var.to(device=rewards.device, dtype=rewards.dtype) + self.epsilon)
        self.prev_dones = torch.logical_or(terminations, truncations)
        return self.observations(observations), self.rewards(rewards), terminations, truncations, infos

    def set_device(self, device: torch.device | str) -> None:
        new_device = torch.device(device)
        self.return_rms.to(new_device)
        self.returns = self.returns.to(new_device)
        self.prev_dones = self.prev_dones.to(new_device)
        super().set_device(new_device)

    def _extract_reset_mask(self, kwargs: dict[str, Any]) -> torch.Tensor | None:
        options = kwargs.get("options", None)
        if not isinstance(options, dict) or "reset_mask" not in options:
            return None
        return to_torch_tensor(options["reset_mask"], device=self.device, dtype=torch.bool).reshape(self._n_envs)
