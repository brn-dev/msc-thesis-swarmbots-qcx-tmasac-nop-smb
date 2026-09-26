from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv, VectorWrapper

from swarmbots.scenario_presets.scenario_obs_layouts import (
    DUAL_PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
    PAYLOAD_GLOBAL_OBS_ADAPTER_NAME,
)


class MoveToPayloadGlobalObsAdapter(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv,
        *,
        payload_z: float,
    ) -> None:
        super().__init__(env)

        single_space = env.single_observation_space
        vector_space = env.observation_space
        if not isinstance(single_space, spaces.Dict):
            raise ValueError(f"Expected Dict single_observation_space, got {type(single_space)}")
        if not isinstance(vector_space, spaces.Dict):
            raise ValueError(f"Expected Dict observation_space, got {type(vector_space)}")
        if tuple(single_space["global_obs"].shape) != (2,):
            raise ValueError(f"Expected move-to global_obs shape (2,), got {single_space['global_obs'].shape}")
        if tuple(vector_space["global_obs"].shape) != (self.num_envs, 2):
            raise ValueError(
                f"Expected move-to vector global_obs shape ({self.num_envs}, 2), got "
                f"{vector_space['global_obs'].shape}"
            )

        self.payload_z = float(payload_z)
        self.single_observation_space = self._replace_global_obs_space(single_space, shape=(9,))
        self.observation_space = self._replace_global_obs_space(vector_space, shape=(self.num_envs, 9))

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._adapt_obs(obs), info

    def step(self, actions: Any) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self._adapt_obs(obs), rewards, terminations, truncations, self._adapt_infos(infos)

    def get_settings(self) -> dict[str, Any]:
        get_settings = getattr(self.env, "get_settings", None)
        if get_settings is None:
            raise AttributeError(f"{type(self.env).__name__} does not expose get_settings()")

        settings = deepcopy(get_settings())
        scenario_settings = settings.setdefault("scenario", {})
        scenario_settings["global_obs_adapter"] = PAYLOAD_GLOBAL_OBS_ADAPTER_NAME
        scenario_settings["adapted_payload_z"] = self.payload_z
        return settings

    @staticmethod
    def _replace_global_obs_space(obs_space: spaces.Dict, *, shape: tuple[int, ...]) -> spaces.Dict:
        new_spaces = dict(obs_space.spaces)
        new_spaces["global_obs"] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        return spaces.Dict(new_spaces)

    def _adapt_infos(self, infos: dict[str, Any]) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        adapted_infos = dict(infos)
        final_obs = infos["final_obs"]
        if isinstance(final_obs, dict):
            adapted_infos["final_obs"] = self._adapt_obs(final_obs)
            return adapted_infos

        final_obs_entries = np.asarray(final_obs, dtype=object).copy()
        final_obs_mask = np.asarray(infos["_final_obs"], dtype=bool).reshape(-1)
        for env_idx in np.nonzero(final_obs_mask)[0].tolist():
            final_obs_entries[env_idx] = self._adapt_obs(final_obs_entries[env_idx])
        adapted_infos["final_obs"] = final_obs_entries
        return adapted_infos

    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        adapted_obs = dict(obs)
        adapted_obs["global_obs"] = self._payload_like_global_obs(obs["global_obs"])
        return adapted_obs

    def _payload_like_global_obs(self, move_to_global_obs: Any) -> Any:
        if isinstance(move_to_global_obs, torch.Tensor):
            payload_global_obs = torch.zeros(
                (*move_to_global_obs.shape[:-1], 9),
                device=move_to_global_obs.device,
                dtype=move_to_global_obs.dtype,
            )
            payload_global_obs[..., :2] = move_to_global_obs[..., :2]
            payload_global_obs[..., 2] = self.payload_z
            payload_global_obs[..., 3] = 1.0
            payload_global_obs[..., 7] = 1.0
            return payload_global_obs

        move_to_global_obs_array = np.asarray(move_to_global_obs)
        payload_global_obs_array = np.zeros((*move_to_global_obs_array.shape[:-1], 9), dtype=np.float32)
        payload_global_obs_array[..., :2] = move_to_global_obs_array[..., :2]
        payload_global_obs_array[..., 2] = self.payload_z
        payload_global_obs_array[..., 3] = 1.0
        payload_global_obs_array[..., 7] = 1.0
        return payload_global_obs_array


class MoveToDualPayloadGlobalObsAdapter(MoveToPayloadGlobalObsAdapter):
    def __init__(
        self,
        env: VectorEnv,
        *,
        payload_z: float,
    ) -> None:
        super().__init__(env, payload_z=payload_z)
        self.single_observation_space = self._replace_global_obs_space(env.single_observation_space, shape=(18,))
        self.observation_space = self._replace_global_obs_space(env.observation_space, shape=(self.num_envs, 18))

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        scenario_settings = settings.setdefault("scenario", {})
        scenario_settings["global_obs_adapter"] = DUAL_PAYLOAD_GLOBAL_OBS_ADAPTER_NAME
        scenario_settings["num_payloads"] = 2
        return settings

    def _payload_like_global_obs(self, move_to_global_obs: Any) -> Any:
        single_payload_obs = super()._payload_like_global_obs(move_to_global_obs)
        if isinstance(single_payload_obs, torch.Tensor):
            return torch.cat((single_payload_obs, single_payload_obs), dim=-1)
        return np.concatenate((single_payload_obs, single_payload_obs), axis=-1)
