from __future__ import annotations

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.logger import warn
from gymnasium.vector.vector_env import (
    ArrayType,
    AutoresetMode,
    VectorEnv,
    VectorWrapper,
)

__all__ = ["ProgressGuidanceEpisodeStatsWrapper"]


class ProgressGuidanceEpisodeStatsWrapper(VectorWrapper):
    def __init__(
            self,
            env: VectorEnv,
            progress_key: str = "progress_reward",
            guidance_key: str = "guidance_reward",
            success_key: str = "success",
            stats_key: str = "episode",
    ):
        super().__init__(env)
        self.progress_key = progress_key
        self.guidance_key = guidance_key
        self.success_key = success_key
        self._stats_key = stats_key

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` tag in its metadata, therefore, "
                "ProgressGuidanceEpisodeStatsWrapper assumes `AutoresetMode.SAME_STEP`."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.SAME_STEP}

        self.episode_progress_rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self.episode_guidance_rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self.prev_dones = np.zeros((self.num_envs,), dtype=bool)

    def reset(
            self,
            seed: int | list[int] | None = None,
            options: dict | None = None,
    ) -> tuple[ObsType, dict]:
        obs, info = super().reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options["reset_mask"]
            assert isinstance(reset_mask, np.ndarray), (
                f"`options['reset_mask']` must be a numpy array, got {type(reset_mask)}"
            )
            assert reset_mask.shape == (self.num_envs,), (
                f"`options['reset_mask']` must have shape ({self.num_envs},), got {reset_mask.shape}"
            )
            assert reset_mask.dtype == np.bool_, (
                f"`options['reset_mask']` must have dtype np.bool_, got {reset_mask.dtype}"
            )

            self.episode_progress_rewards[reset_mask] = 0.0
            self.episode_guidance_rewards[reset_mask] = 0.0
            self.prev_dones[reset_mask] = False
        else:
            self.episode_progress_rewards[:] = 0.0
            self.episode_guidance_rewards[:] = 0.0
            self.prev_dones[:] = False

        return obs, info

    def step(
            self,
            actions: ActType,
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        assert isinstance(infos, dict), (
            "`ProgressGuidanceEpisodeStatsWrapper` requires `info` type to be `dict`, "
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

        dones = np.logical_or(terminations, truncations)
        self.prev_dones = dones

        if np.any(dones):
            self._inject_episode_stats(infos, dones)

        return observations, rewards, terminations, truncations, infos

    def _extract_step_values(
            self,
            infos: dict,
            key: str,
    ) -> np.ndarray | None:
        values = self._extract_values_from_info_dict(infos, key)
        final_values, final_mask = self._extract_final_step_values(infos, key)
        if final_values is None:
            return values
        if values is None:
            values = np.zeros((self.num_envs,), dtype=np.float64)
        else:
            values = values.copy()
        values[final_mask] = final_values[final_mask]
        return values

    def _extract_values_from_info_dict(
            self,
            infos: dict,
            key: str,
    ) -> np.ndarray | None:
        if key not in infos:
            return None

        mask_key = f"_{key}"
        raw_values = np.asarray(infos[key], dtype=np.float64)
        if raw_values.shape == ():
            out = np.full((self.num_envs,), raw_values.item(), dtype=np.float64)
        else:
            out = raw_values.reshape(-1)
            if out.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected infos['{key}'] to have length {self.num_envs}, got shape {raw_values.shape}"
                )

        if mask_key not in infos:
            return out

        present_mask = np.asarray(infos[mask_key], dtype=bool).reshape(-1)
        if present_mask.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected infos['{mask_key}'] to have length {self.num_envs}, got shape {present_mask.shape}"
            )
        masked_out = np.zeros((self.num_envs,), dtype=np.float64)
        if np.any(present_mask):
            masked_out[present_mask] = out[present_mask]
        return masked_out

    def _extract_final_step_values(
            self,
            infos: dict,
            key: str,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        final_info = infos.get("final_info", None)
        if not isinstance(final_info, dict):
            return None, np.zeros((self.num_envs,), dtype=bool)

        values = self._extract_values_from_info_dict(final_info, key)
        if values is None:
            return None, np.zeros((self.num_envs,), dtype=bool)

        if "_final_info" in infos:
            final_mask = np.asarray(infos["_final_info"], dtype=bool).reshape(-1)
            if final_mask.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected infos['_final_info'] to have length {self.num_envs}, got {final_mask.shape}"
                )
        else:
            final_mask = np.ones((self.num_envs,), dtype=bool)
        return values, final_mask

    def _inject_episode_stats(
            self,
            infos: dict,
            dones: np.ndarray,
    ) -> None:
        progress_sum = np.where(dones, self.episode_progress_rewards, 0.0)
        guidance_sum = np.where(dones, self.episode_guidance_rewards, 0.0)
        success_values = self._extract_step_values(infos, self.success_key)
        if success_values is None:
            success_stats = None
        else:
            success_stats = np.where(dones, success_values, 0.0)

        stats_mask_key = f"_{self._stats_key}"
        if stats_mask_key in infos:
            done_mask = np.asarray(infos[stats_mask_key], dtype=bool).reshape(-1)
            if done_mask.shape[0] != self.num_envs:
                raise ValueError(
                    f"Expected infos['{stats_mask_key}'] to have length {self.num_envs}, got {done_mask.shape}"
                )
            if not np.array_equal(done_mask, dones):
                raise ValueError(
                    f"Expected infos['{stats_mask_key}'] to match computed dones, got mismatch."
                )
        else:
            done_mask = dones
            infos[stats_mask_key] = done_mask

        if self._stats_key not in infos:
            episode_stats = {
                self.progress_key: np.where(done_mask, progress_sum, 0.0),
                self.guidance_key: np.where(done_mask, guidance_sum, 0.0),
            }
            if success_stats is not None:
                episode_stats[self.success_key] = success_stats
            infos[self._stats_key] = episode_stats
            return

        stats = infos[self._stats_key]
        if not isinstance(stats, dict):
            raise ValueError(
                f"Expected infos['{self._stats_key}'] to be a dict, got {type(stats)}"
            )
        keys_to_inject = [self.progress_key, self.guidance_key]
        if success_stats is not None:
            keys_to_inject.append(self.success_key)
        if any(key in stats for key in keys_to_inject):
            raise ValueError(
                f"infos['{self._stats_key}'] already contains one of {keys_to_inject}"
            )
        stats[self.progress_key] = np.where(done_mask, progress_sum, 0.0)
        stats[self.guidance_key] = np.where(done_mask, guidance_sum, 0.0)
        if success_stats is not None:
            stats[self.success_key] = success_stats
