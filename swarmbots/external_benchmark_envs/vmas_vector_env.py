from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from swarmbots.external_benchmark_envs.observation_spaces import (
    make_benchmark_observation_space,
)
from swarmbots.learn.tensor_conversion import normalize_reset_mask_options
from swarmbots.learn.torch_device import as_device


class VMASVectorEnv(VectorEnv):
    metadata: ClassVar[dict[str, Any]] = {
        "autoreset_mode": AutoresetMode.SAME_STEP,
        "render_modes": ["human", "rgb_array"],
    }
    supports_per_env_reset_seeds = False

    def __init__(
        self,
        *,
        scenario: str = "balance",
        num_envs: int,
        device: torch.device | str = "cuda",
        episode_length: int = 200,
        scenario_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if num_envs <= 0:
            raise ValueError(f"num_envs must be > 0, got {num_envs}")
        if episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {episode_length}")
        try:
            import vmas
        except ImportError as error:
            raise ImportError(
                "VMAS requires the 'vmas' optional dependency. Install it with `uv sync --extra vmas`."
            ) from error

        self.device = as_device(device)
        self.num_envs = int(num_envs)
        self.scenario = scenario
        self.episode_length = int(episode_length)
        self.action_backend = "torch"
        self.vmas_env = vmas.make_env(
            scenario=scenario,
            num_envs=self.num_envs,
            device=self.device,
            continuous_actions=True,
            max_steps=self.episode_length,
            terminated_truncated=True,
            grad_enabled=False,
            **({} if scenario_kwargs is None else scenario_kwargs),
        )
        self.n_agents = int(self.vmas_env.n_agents)
        observation_spaces = list(self.vmas_env.observation_space)
        action_spaces = list(self.vmas_env.action_space)
        self._validate_homogeneous_spaces(observation_spaces, name="observation")
        self._validate_homogeneous_spaces(action_spaces, name="action")

        local_obs_dim = int(np.prod(observation_spaces[0].shape, dtype=np.int64))
        action_dim = int(np.prod(action_spaces[0].shape, dtype=np.int64))
        self.single_observation_space = make_benchmark_observation_space(
            n_agents=self.n_agents,
            local_obs_dim=local_obs_dim,
            global_state_dim=self.n_agents * local_obs_dim,
        )
        self.observation_space = batch_space(
            self.single_observation_space, n=self.num_envs
        )
        self.single_action_space = spaces.Box(
            low=np.stack(
                [
                    np.asarray(action_space.low).reshape(-1)
                    for action_space in action_spaces
                ]
            ),
            high=np.stack(
                [
                    np.asarray(action_space.high).reshape(-1)
                    for action_space in action_spaces
                ]
            ),
            shape=(self.n_agents, action_dim),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        options = normalize_reset_mask_options(options, num_envs=self.num_envs)
        reset_mask = None if options is None else options.get("reset_mask")
        if reset_mask is None:
            observations = self.vmas_env.reset(seed=seed)
        else:
            if seed is not None:
                raise ValueError(
                    "VMAS does not support seeding individual vector lanes during partial resets."
                )
            reset_mask_tensor = torch.as_tensor(
                reset_mask, device=self.device, dtype=torch.bool
            ).reshape(self.num_envs)
            observations = None
            for env_index in (
                torch.nonzero(reset_mask_tensor, as_tuple=False).flatten().tolist()
            ):
                observations = self.vmas_env.reset_at(env_index)
            if observations is None:
                raise ValueError(
                    "VMAS partial reset requires at least one active reset lane."
                )
        return self._convert_observations(observations), {}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if tuple(actions.shape) != tuple(self.action_space.shape):
            raise ValueError(
                f"Expected actions shape {self.action_space.shape}, got {tuple(actions.shape)}"
            )

        agent_actions = [
            actions[:, agent_index, :] for agent_index in range(self.n_agents)
        ]
        observations, rewards, terminations, truncations, _ = self.vmas_env.step(
            agent_actions
        )
        converted_observations = self._convert_observations(observations)
        terminal_observations = {
            key: value.detach().clone() for key, value in converted_observations.items()
        }
        terminations = terminations.to(device=self.device, dtype=torch.bool)
        truncations = truncations.to(device=self.device, dtype=torch.bool)
        dones = torch.logical_or(terminations, truncations)

        if torch.any(dones):
            reset_observations = observations
            for env_index in torch.nonzero(dones, as_tuple=False).flatten().tolist():
                reset_observations = self.vmas_env.reset_at(env_index)
            converted_observations = self._convert_observations(reset_observations)

        team_rewards = torch.stack(rewards, dim=1).mean(dim=1).to(dtype=torch.float32)
        infos: dict[str, Any] = {}
        if torch.any(dones):
            infos["final_obs"] = terminal_observations
            infos["_final_obs"] = dones.detach().clone()
        return converted_observations, team_rewards, terminations, truncations, infos

    def close(self) -> None:
        close = getattr(self.vmas_env, "close", None)
        if close is not None:
            close()

    def _convert_observations(
        self, observations: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        local_obs = torch.stack(
            [observation.reshape(self.num_envs, -1) for observation in observations],
            dim=1,
        ).to(dtype=torch.float32)
        return {
            "local_obs": local_obs,
            "global_obs": local_obs.new_empty((self.num_envs, 0)),
            "hidden_local_vars": local_obs.new_empty((self.num_envs, self.n_agents, 0)),
            "hidden_global_vars": local_obs.flatten(start_dim=1),
        }

    @staticmethod
    def _validate_homogeneous_spaces(agent_spaces: list[Any], *, name: str) -> None:
        required_box_attributes = ("shape", "low", "high", "dtype")
        if not agent_spaces or not all(
            all(
                hasattr(agent_space, attribute) for attribute in required_box_attributes
            )
            for agent_space in agent_spaces
        ):
            raise ValueError(
                f"MAT/TMASAC require per-agent Box {name} spaces, got {agent_spaces}"
            )
        shapes = {agent_space.shape for agent_space in agent_spaces}
        if len(shapes) != 1:
            raise ValueError(
                f"MAT/TMASAC require equal per-agent {name} shapes, got "
                f"{[agent_space.shape for agent_space in agent_spaces]}"
            )
