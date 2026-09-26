from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.tensor_conversion import to_torch_tensor


class TorchShuffleAgentsWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        *,
        preserve_inactive_prefix_structure: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        if preserve_inactive_prefix_structure and not self.has_agent_mask:
            raise ValueError("preserve_inactive_prefix_structure=True requires an agent_mask observation.")

        self.preserve_inactive_prefix_structure = bool(preserve_inactive_prefix_structure)
        self._rng = torch.Generator(device=self.device)
        if seed is not None:
            self._rng.manual_seed(int(seed))

        self._agent_indices = torch.arange(self.n_agents, device=self.device, dtype=torch.long)
        self._agent_permutation = self._agent_indices.expand(self._n_envs, self.n_agents).clone()
        self._inv_agent_permutation = self._agent_permutation.clone()

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        seed = kwargs.get("seed", None)
        if seed is not None:
            self._rng.manual_seed(int(seed))

        obs, info = self.env.reset(**kwargs)
        reset_mask = self._reset_mask_from_kwargs(kwargs)
        self._reset_agent_permutations(reset_mask=reset_mask, agent_mask=obs.get("agent_mask", None))
        return self._shuffle_obs(obs, permutation=self._agent_permutation), info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        old_permutation = self._agent_permutation.clone()
        unshuffled_actions = self._unshuffle_actions(actions)
        obs, rewards, terminations, truncations, infos = self.env.step(unshuffled_actions)

        transformed_infos = self._shuffle_final_obs(infos=infos, permutation=old_permutation)
        dones = torch.logical_or(terminations, truncations)
        self._reset_agent_permutations(reset_mask=dones, agent_mask=obs.get("agent_mask", None))
        return (
            self._shuffle_obs(obs, permutation=self._agent_permutation),
            rewards,
            terminations,
            truncations,
            transformed_infos,
        )

    def observations(self, observations: TorchObs) -> TorchObs:
        return self._shuffle_obs(observations, permutation=self._agent_permutation)

    def set_device(self, device: torch.device | str) -> None:
        super().set_device(device)
        self._rng = torch.Generator(device=self.device)
        self._agent_indices = self._agent_indices.to(self.device)
        self._agent_permutation = self._agent_permutation.to(self.device)
        self._inv_agent_permutation = self._inv_agent_permutation.to(self.device)

    def _reset_mask_from_kwargs(self, kwargs: dict[str, Any]) -> torch.Tensor:
        options = kwargs.get("options", None)
        if options is None or "reset_mask" not in options:
            return torch.ones((self._n_envs,), device=self.device, dtype=torch.bool)
        return to_torch_tensor(options["reset_mask"], device=self.device, dtype=torch.bool).reshape(self._n_envs)

    def _reset_agent_permutations(self, *, reset_mask: torch.Tensor, agent_mask: torch.Tensor | None) -> None:
        world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return

        if self.preserve_inactive_prefix_structure:
            if agent_mask is None:
                raise ValueError("preserve_inactive_prefix_structure=True requires agent_mask in observations.")
            new_permutation = self._sample_active_only_permutations(agent_mask[world_idx])
        else:
            random_scores = torch.rand(
                (int(world_idx.numel()), self.n_agents),
                generator=self._rng,
                device=self.device,
            )
            new_permutation = random_scores.argsort(dim=1)

        self._agent_permutation[world_idx] = new_permutation
        self._inv_agent_permutation[world_idx] = new_permutation.argsort(dim=1)

    def _sample_active_only_permutations(self, agent_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(agent_mask.shape[0])
        base_permutation = self._agent_indices.expand(batch_size, self.n_agents).clone()
        random_scores = torch.rand(
            (batch_size, self.n_agents),
            generator=self._rng,
            device=self.device,
        )
        active_order = random_scores.masked_fill(~agent_mask, torch.inf).argsort(dim=1)
        active_counts = agent_mask.sum(dim=1, keepdim=True)
        active_value_mask = self._agent_indices.view(1, self.n_agents) < active_counts
        base_permutation[agent_mask] = active_order[active_value_mask]
        return base_permutation

    def _unshuffle_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        index = self._inv_agent_permutation.unsqueeze(-1).expand(-1, -1, actions.shape[-1])
        return actions.gather(dim=1, index=index)

    def _shuffle_obs(self, obs: TorchObs, *, permutation: torch.Tensor) -> TorchObs:
        shuffled = dict(obs)
        shuffled["local_obs"] = self._gather_agents(obs["local_obs"], permutation=permutation)
        shuffled["hidden_local_vars"] = self._gather_agents(obs["hidden_local_vars"], permutation=permutation)
        if "agent_mask" in obs:
            shuffled["agent_mask"] = self._gather_agents(obs["agent_mask"], permutation=permutation)
        return shuffled

    def _shuffle_final_obs(self, *, infos: dict[str, Any], permutation: torch.Tensor) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        final_obs_mask = to_torch_tensor(infos["_final_obs"], device=self.device, dtype=torch.bool).reshape(self._n_envs)
        if not torch.any(final_obs_mask):
            return infos

        transformed_infos = dict(infos)
        transformed_infos["_final_obs"] = final_obs_mask
        final_obs_value = infos["final_obs"]
        if isinstance(final_obs_value, dict):
            final_obs = self._obs_to_torch(final_obs_value)
            shuffled_final_obs = self._shuffle_obs(final_obs, permutation=permutation)
            transformed_infos["final_obs"] = {
                key: value.detach().clone()
                for key, value in shuffled_final_obs.items()
            }
            return transformed_infos

        final_obs_entries = np.asarray(final_obs_value, dtype=object).copy()
        for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
            final_obs = self._obs_to_torch(final_obs_entries[env_idx])
            shuffled_final_obs = self._shuffle_obs(final_obs, permutation=permutation[env_idx])
            final_obs_entries[env_idx] = {
                key: value.detach().clone()
                for key, value in shuffled_final_obs.items()
            }
        transformed_infos["final_obs"] = final_obs_entries
        return transformed_infos

    @staticmethod
    def _gather_agents(value: torch.Tensor, *, permutation: torch.Tensor) -> torch.Tensor:
        if permutation.ndim == 1:
            return value.index_select(0, permutation)

        index = permutation
        while index.ndim < value.ndim:
            index = index.unsqueeze(-1)
        return value.gather(dim=1, index=index.expand_as(value))
