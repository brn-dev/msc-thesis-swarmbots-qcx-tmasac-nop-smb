from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from swarmbots.external_benchmark_envs.observation_spaces import (
    make_benchmark_observation_space,
)


class MultiAgentMujocoEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        *,
        scenario: str = "HalfCheetah",
        agent_conf: str = "2x3",
        agent_obsk: int = 1,
        episode_length: int = 1_000,
        render_mode: str | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if episode_length <= 0:
            raise ValueError(f"episode_length must be > 0, got {episode_length}")

        try:
            from gymnasium_robotics.envs.multiagent_mujoco import mamujoco_v1
        except ImportError as error:
            raise ImportError(
                "Multi-Agent MuJoCo requires the 'mamujoco' optional dependency. "
                "Install it with `uv sync --extra mamujoco`."
            ) from error

        self.parallel_env = mamujoco_v1.parallel_env(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            render_mode=render_mode,
            **({} if env_kwargs is None else env_kwargs),
        )
        self.scenario = scenario
        self.agent_conf = agent_conf
        self.agent_obsk = int(agent_obsk)
        self.episode_length = int(episode_length)
        self.render_mode = render_mode
        self.agent_names = tuple(self.parallel_env.possible_agents)
        if not self.agent_names:
            raise ValueError(
                "Multi-Agent MuJoCo created an environment without agents."
            )

        local_obs_spaces = [
            self.parallel_env.observation_space(agent) for agent in self.agent_names
        ]
        action_spaces = [
            self.parallel_env.action_space(agent) for agent in self.agent_names
        ]
        self._validate_homogeneous_box_spaces(local_obs_spaces, name="observation")
        self._validate_homogeneous_box_spaces(action_spaces, name="action")

        initial_observations, _ = self.parallel_env.reset()
        global_state_dim = int(np.asarray(self.parallel_env.state()).size)
        local_obs_dim = int(np.asarray(initial_observations[self.agent_names[0]]).size)
        action_dim = int(np.prod(action_spaces[0].shape, dtype=np.int64))

        self.observation_space = make_benchmark_observation_space(
            n_agents=len(self.agent_names),
            local_obs_dim=local_obs_dim,
            global_state_dim=global_state_dim,
        )
        self.action_space = spaces.Box(
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
            shape=(len(self.agent_names), action_dim),
            dtype=np.float32,
        )
        self._elapsed_steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        _ = options
        observations, _ = self.parallel_env.reset(seed=seed)
        self._elapsed_steps = 0
        return self._convert_observations(observations), {}

    def step(
        self,
        actions: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.float32, bool, bool, dict[str, Any]]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != self.action_space.shape:
            raise ValueError(
                f"Expected actions shape {self.action_space.shape}, got {actions.shape}"
            )
        agent_actions = {
            agent: actions[agent_index]
            for agent_index, agent in enumerate(self.agent_names)
        }
        observations, rewards, terminations, truncations, _ = self.parallel_env.step(
            agent_actions
        )
        self._elapsed_steps += 1

        terminated = self._shared_done(terminations, name="termination")
        truncated = self._shared_done(truncations, name="truncation")
        truncated = truncated or self._elapsed_steps >= self.episode_length
        reward = np.float32(np.mean([rewards[agent] for agent in self.agent_names]))
        return (
            self._convert_observations(observations),
            reward,
            terminated,
            truncated,
            {},
        )

    def render(self) -> Any:
        return self.parallel_env.render()

    def close(self) -> None:
        self.parallel_env.close()

    def _convert_observations(
        self, observations: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        local_obs = np.stack(
            [
                np.asarray(observations[agent], dtype=np.float32).reshape(-1)
                for agent in self.agent_names
            ]
        )
        return {
            "local_obs": local_obs,
            "global_obs": np.empty((0,), dtype=np.float32),
            "hidden_local_vars": np.empty((len(self.agent_names), 0), dtype=np.float32),
            "hidden_global_vars": np.asarray(
                self.parallel_env.state(), dtype=np.float32
            ).reshape(-1),
        }

    def _shared_done(self, values: dict[str, bool], *, name: str) -> bool:
        agent_values = [bool(values[agent]) for agent in self.agent_names]
        if any(value != agent_values[0] for value in agent_values[1:]):
            raise RuntimeError(
                f"Multi-Agent MuJoCo returned inconsistent per-agent {name}s: {values}"
            )
        return agent_values[0]

    @staticmethod
    def _validate_homogeneous_box_spaces(
        agent_spaces: list[spaces.Space], *, name: str
    ) -> None:
        if not all(isinstance(agent_space, spaces.Box) for agent_space in agent_spaces):
            raise ValueError(
                f"MAT/TMASAC require continuous Box {name} spaces, got {agent_spaces}"
            )
        shapes = {agent_space.shape for agent_space in agent_spaces}
        if len(shapes) != 1:
            raise ValueError(
                f"MAT/TMASAC require equal per-agent {name} shapes, got "
                f"{[agent_space.shape for agent_space in agent_spaces]}"
            )
