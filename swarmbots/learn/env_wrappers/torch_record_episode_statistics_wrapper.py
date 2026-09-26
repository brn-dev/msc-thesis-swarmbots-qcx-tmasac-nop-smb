from __future__ import annotations

import time
from typing import Any

import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.tensor_conversion import to_torch_tensor


class TorchRecordEpisodeStatisticsWrapper(TorchEnvWrapper):
    def __init__(self, env: BaseLearnEnvWrapper, stats_key: str = "episode") -> None:
        super().__init__(env)
        self._stats_key = stats_key
        self.episode_returns = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        self.episode_lengths = torch.zeros((self._n_envs,), device=self.device, dtype=torch.int64)
        self.episode_start_times = torch.full((self._n_envs,), time.perf_counter(), device=self.device, dtype=torch.float64)
        self.prev_dones = torch.zeros((self._n_envs,), device=self.device, dtype=torch.bool)

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        reset_mask = self._extract_reset_mask(kwargs)
        now = time.perf_counter()
        if reset_mask is None:
            self.episode_returns.zero_()
            self.episode_lengths.zero_()
            self.episode_start_times.fill_(now)
            self.prev_dones.zero_()
        else:
            self.episode_returns[reset_mask] = 0.0
            self.episode_lengths[reset_mask] = 0
            self.episode_start_times[reset_mask] = now
            self.prev_dones[reset_mask] = False
        return obs, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        now = time.perf_counter()

        if torch.any(self.prev_dones):
            self.episode_returns[self.prev_dones] = 0.0
            self.episode_lengths[self.prev_dones] = 0
            self.episode_start_times[self.prev_dones] = now

        self.episode_returns += rewards.to(dtype=self.episode_returns.dtype)
        self.episode_lengths += 1

        dones = torch.logical_or(terminations, truncations)
        self.prev_dones = dones

        if torch.any(dones):
            self._inject_episode_stats(infos=infos, dones=dones, now=now)

        return self.observations(observations), self.rewards(rewards), terminations, truncations, infos

    def set_device(self, device: torch.device | str) -> None:
        new_device = torch.device(device)
        self.episode_returns = self.episode_returns.to(new_device)
        self.episode_lengths = self.episode_lengths.to(new_device)
        self.episode_start_times = self.episode_start_times.to(new_device)
        self.prev_dones = self.prev_dones.to(new_device)
        super().set_device(new_device)

    def _extract_reset_mask(self, kwargs: dict[str, Any]) -> torch.Tensor | None:
        options = kwargs.get("options", None)
        if not isinstance(options, dict) or "reset_mask" not in options:
            return None
        return to_torch_tensor(options["reset_mask"], device=self.device, dtype=torch.bool).reshape(self._n_envs)

    def _inject_episode_stats(self, *, infos: dict[str, Any], dones: torch.Tensor, now: float) -> None:
        returns = torch.where(dones, self.episode_returns, torch.zeros_like(self.episode_returns))
        lengths = torch.where(dones, self.episode_lengths, torch.zeros_like(self.episode_lengths))
        elapsed = torch.where(
            dones,
            torch.full_like(self.episode_start_times, now) - self.episode_start_times,
            torch.zeros_like(self.episode_start_times),
        )

        stats_mask_key = f"_{self._stats_key}"
        if stats_mask_key in infos:
            existing_done_mask = to_torch_tensor(infos[stats_mask_key], device=self.device, dtype=torch.bool).reshape(-1)
            if tuple(existing_done_mask.shape) != (self._n_envs,):
                raise ValueError(
                    f"Expected infos['{stats_mask_key}'] shape ({self._n_envs},), got {tuple(existing_done_mask.shape)}"
                )
            if not torch.equal(existing_done_mask, dones):
                raise ValueError(f"Expected infos['{stats_mask_key}'] to match computed dones.")
        else:
            infos[stats_mask_key] = dones.clone()

        stats = {
            "r": returns.detach().clone(),
            "l": lengths.detach().clone(),
            "t": elapsed.detach().clone(),
        }
        if self._stats_key not in infos:
            infos[self._stats_key] = stats
            return

        existing_stats = infos[self._stats_key]
        if not isinstance(existing_stats, dict):
            raise ValueError(f"Expected infos['{self._stats_key}'] to be dict, got {type(existing_stats)}")
        for key, value in stats.items():
            if key in existing_stats:
                raise ValueError(f"infos['{self._stats_key}'] already contains '{key}'")
            existing_stats[key] = value
