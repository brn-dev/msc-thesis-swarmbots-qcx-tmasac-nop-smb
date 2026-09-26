from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import (
    BaseLearnEnvWrapper,
    TorchObs,
)
from swarmbots.learn.tensor_conversion import to_torch_tensor


class TorchEnvWrapper(BaseLearnEnvWrapper):
    def __init__(self, env: BaseLearnEnvWrapper) -> None:
        self.env = env
        self.device = env.device
        self.obs_dtype = env.obs_dtype
        self.reward_dtype = env.reward_dtype
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._n_envs = int(getattr(env, "_n_envs", env.num_envs))
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.has_scenario_id = bool(getattr(env, "has_scenario_id", False))
        self.scenario_names = getattr(env, "scenario_names", None)
        self.scenario_observation_dims = getattr(env, "scenario_observation_dims", None)
        self.has_agent_mask = env.has_agent_mask
        self.n_agents = env.n_agents

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.observations(obs), info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return (
            self.observations(obs),
            self.rewards(rewards),
            terminations,
            truncations,
            self._transform_infos(infos),
        )

    def observations(self, observations: TorchObs) -> TorchObs:
        return observations

    def rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards

    def set_device(self, device: torch.device | str) -> None:
        self.env.set_device(device)
        self.device = self.env.device

    def _actions_to_env(self, actions: torch.Tensor) -> Any:
        return self.env._actions_to_env(actions)

    def _transform_infos(self, infos: dict[str, Any]) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        final_obs_mask = to_torch_tensor(
            infos["_final_obs"], device=self.device, dtype=torch.bool
        ).reshape(self._n_envs)
        if not torch.any(final_obs_mask):
            return infos

        transformed_infos = dict(infos)
        transformed_infos["_final_obs"] = final_obs_mask

        final_obs_value = infos["final_obs"]
        if isinstance(final_obs_value, dict):
            final_obs = self._obs_to_torch(final_obs_value)
            transformed_final_obs = self.observations(final_obs)
            transformed_infos["final_obs"] = {
                key: value.detach().clone()
                for key, value in transformed_final_obs.items()
            }
            return transformed_infos

        final_obs_entries = np.asarray(final_obs_value, dtype=object).copy()
        for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
            final_obs = self._obs_to_torch(final_obs_entries[env_idx])
            transformed_final_obs = self.observations(final_obs)
            final_obs_entries[env_idx] = {
                key: value.detach().clone()
                for key, value in transformed_final_obs.items()
            }

        transformed_infos["final_obs"] = final_obs_entries
        return transformed_infos
