from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.tensor_conversion import to_torch_tensor


def _step_transition_tensors(
    local_obs: torch.Tensor,
    previous_local_obs: torch.Tensor,
    previous_actions: torch.Tensor,
    dones: torch.Tensor,
    global_obs: torch.Tensor | None,
    previous_global_obs: torch.Tensor | None,
    new_obs_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    local_done_mask = dones.reshape(dones.shape[0], 1, 1)
    transition_previous_local = previous_local_obs.masked_fill(local_done_mask, 0)
    transition_previous_actions = previous_actions.masked_fill(local_done_mask, 0)
    if new_obs_first:
        stacked_local = torch.cat((local_obs, transition_previous_actions, transition_previous_local), dim=-1)
    else:
        stacked_local = torch.cat((transition_previous_local, transition_previous_actions, local_obs), dim=-1)

    if global_obs is None:
        return stacked_local, transition_previous_local, transition_previous_actions, None, None
    if previous_global_obs is None:
        previous_global_obs = torch.zeros_like(global_obs)
    transition_previous_global = previous_global_obs.masked_fill(dones.reshape(dones.shape[0], 1), 0)
    stacked_global = (
        torch.cat((global_obs, transition_previous_global), dim=-1)
        if new_obs_first
        else torch.cat((transition_previous_global, global_obs), dim=-1)
    )
    return (
        stacked_local,
        transition_previous_local,
        transition_previous_actions,
        stacked_global,
        transition_previous_global,
    )


class TorchTransitionObsWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        new_obs_first: bool = True,
        normalize_prev_binary_actions: bool = False,
        compile_tensor_operations: bool | None = None,
        tensor_operations_compile_mode: str = "default",
    ) -> None:
        super().__init__(env)
        if "local_obs" not in self.observation_space.spaces:
            raise ValueError('Expected "local_obs" key in observation space')

        self.new_obs_first = new_obs_first
        self.normalize_prev_binary_actions = bool(normalize_prev_binary_actions)
        compile_operations = (
            self.device.type == "cuda"
            if compile_tensor_operations is None
            else compile_tensor_operations
        )
        if compile_operations:
            if not hasattr(torch, "compile") or not callable(torch.compile):
                raise RuntimeError("Compiling transition-observation tensor operations requires torch.compile.")
            if not tensor_operations_compile_mode:
                raise ValueError("tensor_operations_compile_mode must be non-empty when compilation is enabled.")
            self._step_transition_tensors = torch.compile(
                _step_transition_tensors,
                mode=tensor_operations_compile_mode,
                fullgraph=True,
                dynamic=False,
            )
        else:
            self._step_transition_tensors = _step_transition_tensors
        local_space: spaces.Box = self.observation_space["local_obs"]  # type: ignore[assignment]
        self._n_agents, self._n_local_obs = map(int, local_space.shape[1:])
        self._has_global_obs = "global_obs" in self.observation_space.spaces
        self._n_action_features = int(self.action_space.total_agent_action_dim)
        self._prev_action_binary_indices = self._build_binary_action_indices()
        self._observation_space = self._with_transition_obs_space(self.observation_space)
        self.local_obs_dim = int(self._observation_space["local_obs"].shape[-1])
        if self._has_global_obs:
            self.global_obs_dim = int(self._observation_space["global_obs"].shape[-1])

        self._prev_local = torch.zeros(
            (self._n_envs, self._n_agents, self._n_local_obs),
            device=self.device,
            dtype=torch.float32,
        )
        current_prev_actions = torch.zeros(
            (self._n_envs, self._n_agents, self._n_action_features),
            device=self.device,
            dtype=torch.float32,
        )
        self._current_prev_local = torch.zeros_like(self._prev_local)
        self._current_prev_actions = current_prev_actions
        self._prev_global: torch.Tensor | None = None
        self._current_prev_global: torch.Tensor | None = None
        if self._has_global_obs:
            global_obs_dim = int(self.env.observation_space["global_obs"].shape[-1])
            self._prev_global = torch.zeros((self._n_envs, global_obs_dim), device=self.device, dtype=torch.float32)
            self._current_prev_global = torch.zeros_like(self._prev_global)

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        reset_mask = self._extract_reset_mask(kwargs)
        old_current_prev_local = self._current_prev_local
        old_current_prev_actions = self._current_prev_actions
        old_current_prev_global = self._current_prev_global

        obs, info = self.env.reset(**kwargs)
        local = obs["local_obs"]
        if reset_mask is None:
            reset_mask = torch.ones((self._n_envs,), device=self.device, dtype=torch.bool)

        self._prev_local = self._prev_local.clone()
        self._current_prev_local = self._current_prev_local.clone()
        self._current_prev_actions = self._current_prev_actions.clone()
        self._prev_local[reset_mask] = local[reset_mask]
        self._current_prev_local[reset_mask] = 0.0
        self._current_prev_actions[reset_mask] = 0.0

        prev_local = old_current_prev_local.masked_fill(reset_mask.view(self._n_envs, 1, 1), 0)
        prev_actions = old_current_prev_actions.masked_fill(reset_mask.view(self._n_envs, 1, 1), 0)

        prev_global = None
        if self._has_global_obs and "global_obs" in obs:
            global_obs = obs["global_obs"]
            if self._prev_global is None:
                self._prev_global = torch.zeros_like(global_obs)
            if self._current_prev_global is None:
                self._current_prev_global = torch.zeros_like(global_obs)
            self._prev_global = self._prev_global.clone()
            self._current_prev_global = self._current_prev_global.clone()
            self._prev_global[reset_mask] = global_obs[reset_mask]
            self._current_prev_global[reset_mask] = 0.0
            if old_current_prev_global is None:
                prev_global = torch.zeros_like(global_obs)
            else:
                prev_global = old_current_prev_global.masked_fill(reset_mask.view(self._n_envs, 1), 0)
        return self._stack_obs(obs, prev_local=prev_local, prev_actions=prev_actions, prev_global=prev_global), info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        prev_local = self._prev_local
        prev_actions = self._actions_to_features(actions, dtype=prev_local.dtype)
        prev_global = self._prev_global if self._has_global_obs else None

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        infos = self._transform_infos_with_transition_state(
            infos=infos,
            prev_local=prev_local,
            prev_actions=prev_actions,
            prev_global=prev_global,
        )
        dones = torch.logical_or(terminations, truncations)

        (
            stacked_local_obs,
            obs_prev_local,
            obs_prev_actions,
            stacked_global_obs,
            obs_prev_global,
        ) = self._step_transition_tensors(
            obs["local_obs"],
            prev_local,
            prev_actions,
            dones,
            obs.get("global_obs"),
            prev_global,
            self.new_obs_first,
        )

        self._prev_local = obs["local_obs"].clone()
        self._current_prev_local = obs_prev_local.clone()
        self._current_prev_actions = obs_prev_actions.clone()
        if self._has_global_obs and self._prev_global is not None and "global_obs" in obs:
            self._prev_global = obs["global_obs"].clone()
            if obs_prev_global is not None:
                self._current_prev_global = obs_prev_global.clone()

        stacked_obs = dict(obs)
        stacked_obs["local_obs"] = stacked_local_obs
        if stacked_global_obs is not None:
            stacked_obs["global_obs"] = stacked_global_obs
        return stacked_obs, rewards, terminations, truncations, infos

    def set_device(self, device: torch.device | str) -> None:
        new_device = torch.device(device)
        self._prev_local = self._prev_local.to(new_device)
        self._current_prev_local = self._current_prev_local.to(new_device)
        self._current_prev_actions = self._current_prev_actions.to(new_device)
        if self._prev_global is not None:
            self._prev_global = self._prev_global.to(new_device)
        if self._current_prev_global is not None:
            self._current_prev_global = self._current_prev_global.to(new_device)
        super().set_device(new_device)

    def _extract_reset_mask(self, kwargs: dict[str, Any]) -> torch.Tensor | None:
        options = kwargs.get("options", None)
        if not isinstance(options, dict) or "reset_mask" not in options:
            return None
        return to_torch_tensor(options["reset_mask"], device=self.device, dtype=torch.bool).reshape(self._n_envs)

    def _with_transition_obs_space(self, obs_space: spaces.Dict) -> spaces.Dict:
        new_spaces: dict[str, spaces.Space] = dict(obs_space.spaces)
        local_space: spaces.Box = obs_space["local_obs"]  # type: ignore[assignment]
        local_shape = tuple(int(x) for x in local_space.shape)
        new_spaces["local_obs"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(local_shape[0], local_shape[1], 2 * local_shape[2] + self._n_action_features),
            dtype=local_space.dtype,
        )

        if self._has_global_obs and "global_obs" in obs_space.keys():
            global_space: spaces.Box = obs_space["global_obs"]  # type: ignore[assignment]
            global_shape = tuple(int(x) for x in global_space.shape)
            new_spaces["global_obs"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(global_shape[0], 2 * global_shape[1]),
                dtype=global_space.dtype,
            )

        return spaces.Dict(new_spaces)

    def _stack_obs(
        self,
        obs: TorchObs,
        *,
        prev_local: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_global: torch.Tensor | None,
    ) -> TorchObs:
        stacked = dict(obs)
        local = obs["local_obs"]
        if self.new_obs_first:
            stacked["local_obs"] = torch.cat((local, prev_actions, prev_local), dim=-1)
        else:
            stacked["local_obs"] = torch.cat((prev_local, prev_actions, local), dim=-1)

        if self._has_global_obs and "global_obs" in obs:
            global_obs = obs["global_obs"]
            if prev_global is None:
                prev_global = torch.zeros_like(global_obs)
            if self.new_obs_first:
                stacked["global_obs"] = torch.cat((global_obs, prev_global), dim=-1)
            else:
                stacked["global_obs"] = torch.cat((prev_global, global_obs), dim=-1)
        return stacked

    def _actions_to_features(self, actions: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        features = actions.to(device=self.device, dtype=dtype)
        if self.normalize_prev_binary_actions and self._prev_action_binary_indices.numel() > 0:
            features = features.clone()
            binary_indices = self._prev_action_binary_indices.to(device=features.device)
            features[..., binary_indices] = features[..., binary_indices] * 2.0 - 1.0
        return features

    def _build_binary_action_indices(self) -> torch.Tensor:
        indices: list[int] = []
        offset = 0
        for sub_space, action_dim in zip(self.action_space.sub_spaces, self.action_space.agent_action_dims, strict=True):
            if isinstance(sub_space, spaces.MultiBinary):
                indices.extend(range(offset, offset + action_dim))
            offset += action_dim
        return torch.as_tensor(indices, dtype=torch.long)

    def _transform_infos_with_transition_state(
        self,
        *,
        infos: dict[str, Any],
        prev_local: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_global: torch.Tensor | None,
    ) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        final_obs_mask = to_torch_tensor(infos["_final_obs"], device=self.device, dtype=torch.bool).reshape(self._n_envs)

        transformed_infos = dict(infos)
        transformed_infos["_final_obs"] = final_obs_mask

        final_obs_value = infos["final_obs"]
        if isinstance(final_obs_value, dict):
            final_obs = self._obs_to_torch(final_obs_value)
            stacked_final_obs = self._stack_obs(
                final_obs,
                prev_local=prev_local,
                prev_actions=prev_actions,
                prev_global=prev_global,
            )
            transformed_infos["final_obs"] = {
                key: value.detach().clone()
                for key, value in stacked_final_obs.items()
            }
            return transformed_infos

        final_obs_entries = np.asarray(final_obs_value, dtype=object).copy()
        for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
            final_obs = {
                key: value.unsqueeze(0)
                for key, value in self._obs_to_torch(final_obs_entries[env_idx]).items()
            }
            stacked_final_obs = self._stack_obs(
                final_obs,
                prev_local=prev_local[env_idx:env_idx + 1],
                prev_actions=prev_actions[env_idx:env_idx + 1],
                prev_global=None if prev_global is None else prev_global[env_idx:env_idx + 1],
            )
            final_obs_entries[env_idx] = {
                key: value.squeeze(0).detach().clone()
                for key, value in stacked_final_obs.items()
            }

        transformed_infos["final_obs"] = final_obs_entries
        return transformed_infos
