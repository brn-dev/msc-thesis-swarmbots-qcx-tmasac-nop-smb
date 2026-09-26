from __future__ import annotations

from typing import Any

import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.tensor_conversion import to_torch_tensor

__all__ = ["TorchProgressGuidanceEpisodeStatsWrapper"]


class TorchProgressGuidanceEpisodeStatsWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        progress_key: str = "progress_reward",
        guidance_key: str = "guidance_reward",
        success_key: str = "success",
        stats_key: str = "episode",
    ) -> None:
        super().__init__(env)
        self.progress_key = progress_key
        self.guidance_key = guidance_key
        self.success_key = success_key
        self._stats_key = stats_key
        self.episode_progress_rewards = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        self.episode_guidance_rewards = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        self.prev_dones = torch.zeros((self._n_envs,), device=self.device, dtype=torch.bool)

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = super().reset(**kwargs)
        reset_mask = self._extract_reset_mask(kwargs)
        if reset_mask is None:
            self.episode_progress_rewards.zero_()
            self.episode_guidance_rewards.zero_()
            self.prev_dones.zero_()
        else:
            self.episode_progress_rewards[reset_mask] = 0.0
            self.episode_guidance_rewards[reset_mask] = 0.0
            self.prev_dones[reset_mask] = False
        return obs, info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        if not isinstance(infos, dict):
            raise ValueError(
                "`TorchProgressGuidanceEpisodeStatsWrapper` requires info type to be dict, "
                f"got {type(infos)}."
            )

        self.episode_progress_rewards[self.prev_dones] = 0.0
        self.episode_guidance_rewards[self.prev_dones] = 0.0

        progress_values = self._extract_step_values(infos, self.progress_key)
        guidance_values = self._extract_step_values(infos, self.guidance_key)
        if progress_values is not None:
            self.episode_progress_rewards += progress_values
        if guidance_values is not None:
            self.episode_guidance_rewards += guidance_values

        dones = torch.logical_or(terminations, truncations)
        self.prev_dones = dones
        if torch.any(dones):
            self._inject_episode_stats(infos=infos, dones=dones)

        return self.observations(observations), self.rewards(rewards), terminations, truncations, infos

    def set_device(self, device: torch.device | str) -> None:
        new_device = torch.device(device)
        self.episode_progress_rewards = self.episode_progress_rewards.to(new_device)
        self.episode_guidance_rewards = self.episode_guidance_rewards.to(new_device)
        self.prev_dones = self.prev_dones.to(new_device)
        super().set_device(new_device)

    def _extract_reset_mask(self, kwargs: dict[str, Any]) -> torch.Tensor | None:
        options = kwargs.get("options", None)
        if not isinstance(options, dict) or "reset_mask" not in options:
            return None
        return to_torch_tensor(options["reset_mask"], device=self.device, dtype=torch.bool).reshape(self._n_envs)

    def _extract_step_values(self, infos: dict[str, Any], key: str) -> torch.Tensor | None:
        values = self._extract_values_from_info_dict(infos, key)
        final_values, final_mask = self._extract_final_step_values(infos, key)
        if final_values is None:
            return values
        if values is None:
            values = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        else:
            values = values.clone()
        values[final_mask] = final_values[final_mask]
        return values

    def _extract_values_from_info_dict(self, infos: dict[str, Any], key: str) -> torch.Tensor | None:
        if key not in infos:
            return None

        mask_key = f"_{key}"
        present_mask = None
        if mask_key in infos:
            present_mask = to_torch_tensor(infos[mask_key], device=self.device, dtype=torch.bool).reshape(-1)
            if present_mask.shape[0] != self._n_envs:
                raise ValueError(
                    f"Expected infos['{mask_key}'] length {self._n_envs}, got shape {tuple(present_mask.shape)}"
                )

        values = self._info_values_to_tensor(infos[key], key=key, present_mask=present_mask)
        if present_mask is None:
            return values

        masked_values = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        if torch.any(present_mask):
            masked_values[present_mask] = values[present_mask]
        return masked_values

    def _info_values_to_tensor(
        self,
        raw_values: Any,
        *,
        key: str,
        present_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            values = to_torch_tensor(raw_values, device=self.device, dtype=torch.float64).reshape(-1)
        except (BufferError, TypeError, ValueError):
            return self._masked_info_entries_to_tensor(raw_values, key=key, present_mask=present_mask)

        if values.shape[0] == 1:
            return values.expand(self._n_envs)
        if values.shape[0] != self._n_envs:
            raise ValueError(f"Expected infos['{key}'] length {self._n_envs}, got shape {tuple(values.shape)}")
        return values

    def _masked_info_entries_to_tensor(
        self,
        raw_values: Any,
        *,
        key: str,
        present_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        entries = self._flatten_info_entries(raw_values)
        if len(entries) == 1:
            if present_mask is not None and not torch.any(present_mask):
                return torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
            return to_torch_tensor(entries, device=self.device, dtype=torch.float64).reshape(-1).expand(self._n_envs)
        if len(entries) != self._n_envs:
            raise ValueError(f"Expected infos['{key}'] length {self._n_envs}, got length {len(entries)}")
        if present_mask is None:
            return to_torch_tensor(entries, device=self.device, dtype=torch.float64).reshape(-1)

        values = torch.zeros((self._n_envs,), device=self.device, dtype=torch.float64)
        present_indices = torch.nonzero(present_mask, as_tuple=False).flatten().tolist()
        if present_indices:
            present_values = [entries[index] for index in present_indices]
            values[present_mask] = to_torch_tensor(present_values, device=self.device, dtype=torch.float64).reshape(-1)
        return values

    @staticmethod
    def _flatten_info_entries(raw_values: Any) -> list[Any]:
        if hasattr(raw_values, "reshape") and not isinstance(raw_values, torch.Tensor):
            try:
                flat_values = raw_values.reshape(-1)
                return [flat_values[index] for index in range(len(flat_values))]
            except (TypeError, ValueError, AttributeError):
                pass
        try:
            return list(raw_values)
        except TypeError:
            return [raw_values]

    def _extract_final_step_values(
        self,
        infos: dict[str, Any],
        key: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        final_info = infos.get("final_info", None)
        if not isinstance(final_info, dict):
            return None, torch.zeros((self._n_envs,), device=self.device, dtype=torch.bool)

        values = self._extract_values_from_info_dict(final_info, key)
        if values is None:
            return None, torch.zeros((self._n_envs,), device=self.device, dtype=torch.bool)

        if "_final_info" in infos:
            final_mask = to_torch_tensor(infos["_final_info"], device=self.device, dtype=torch.bool).reshape(-1)
            if final_mask.shape[0] != self._n_envs:
                raise ValueError(
                    f"Expected infos['_final_info'] length {self._n_envs}, got shape {tuple(final_mask.shape)}"
                )
        else:
            final_mask = torch.ones((self._n_envs,), device=self.device, dtype=torch.bool)
        return values, final_mask

    def _inject_episode_stats(self, *, infos: dict[str, Any], dones: torch.Tensor) -> None:
        progress_sum = torch.where(dones, self.episode_progress_rewards, torch.zeros_like(self.episode_progress_rewards))
        guidance_sum = torch.where(dones, self.episode_guidance_rewards, torch.zeros_like(self.episode_guidance_rewards))
        success_values = self._extract_step_values(infos, self.success_key)
        if success_values is None:
            success_stats = None
        else:
            success_stats = torch.where(dones, success_values, torch.zeros_like(success_values))

        stats_mask_key = f"_{self._stats_key}"
        if stats_mask_key in infos:
            existing_done_mask = to_torch_tensor(infos[stats_mask_key], device=self.device, dtype=torch.bool).reshape(-1)
            if tuple(existing_done_mask.shape) != (self._n_envs,):
                raise ValueError(
                    f"Expected infos['{stats_mask_key}'] length {self._n_envs}, got {tuple(existing_done_mask.shape)}"
                )
            if not torch.equal(existing_done_mask, dones):
                raise ValueError(f"Expected infos['{stats_mask_key}'] to match computed dones.")
        else:
            infos[stats_mask_key] = dones.clone()

        progress_values = progress_sum.detach().clone()
        guidance_values = guidance_sum.detach().clone()

        if self._stats_key not in infos:
            episode_stats = {
                self.progress_key: progress_values,
                self.guidance_key: guidance_values,
            }
            if success_stats is not None:
                episode_stats[self.success_key] = success_stats.detach().clone()
            infos[self._stats_key] = episode_stats
            return

        stats = infos[self._stats_key]
        if not isinstance(stats, dict):
            raise ValueError(f"Expected infos['{self._stats_key}'] to be dict, got {type(stats)}")
        keys_to_inject = [self.progress_key, self.guidance_key]
        if success_stats is not None:
            keys_to_inject.append(self.success_key)
        if any(key in stats for key in keys_to_inject):
            raise ValueError(
                f"infos['{self._stats_key}'] already contains one of {keys_to_inject}"
            )
        stats[self.progress_key] = progress_values
        stats[self.guidance_key] = guidance_values
        if success_stats is not None:
            stats[self.success_key] = success_stats.detach().clone()
