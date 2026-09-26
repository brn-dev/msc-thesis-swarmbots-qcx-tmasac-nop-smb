

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces


class TestingSwarmBotsEnv(gymnasium.Env):
    def __init__(
        self,
        n_agents: int,
        n_local_obs: int,
        n_global_obs: int,
        actuators_dim: int,
        connectors_dim: int,
        n_hidden_local_vars: int = 0,
        n_hidden_global_vars: int = 0,
        max_steps: int = 100,
        continuous_connector_actions: bool = False,
    ):
        self.n_agents = n_agents
        self.n_local_obs = n_local_obs
        self.n_global_obs = n_global_obs
        self.n_hidden_local_vars = n_hidden_local_vars
        self.n_hidden_global_vars = n_hidden_global_vars
        self.max_steps = max_steps

        self._step_count = 0

        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_agents, n_local_obs),
                dtype=np.float32
            ),
            "global_obs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_global_obs,),
                dtype=np.float32
            ),
            "hidden_local_vars": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_agents, n_hidden_local_vars),
                dtype=np.float32
            ),
            "hidden_global_vars": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_hidden_global_vars,),
                dtype=np.float32
            ),
        })

        connector_space: spaces.Box | spaces.MultiBinary
        if continuous_connector_actions:
            connector_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(n_agents, connectors_dim),
                dtype=np.float32,
            )
        else:
            connector_space = spaces.MultiBinary((n_agents, connectors_dim))

        self.action_space = spaces.Dict({
            "actuators": spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_agents, actuators_dim),
                dtype=np.float32
            ),
            "connectors": connector_space,
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        
        truncated = self._step_count >= self.max_steps
        terminated = False
        reward = self._step_count if not truncated else -100
        
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> dict[str, np.ndarray]:
        local_obs = np.full(
            (self.n_agents, self.n_local_obs), 
            self._step_count, 
            dtype=np.float32
        )
        global_obs = np.full(
            (self.n_global_obs,), 
            self._step_count, 
            dtype=np.float32
        )
        hidden_local_vars = np.full(
            (self.n_agents, self.n_hidden_local_vars),
            self._step_count,
            dtype=np.float32
        )
        hidden_global_vars = np.full(
            (self.n_hidden_global_vars,),
            self._step_count,
            dtype=np.float32
        )

        return {
            "local_obs": local_obs,
            "global_obs": global_obs,
            "hidden_local_vars": hidden_local_vars,
            "hidden_global_vars": hidden_global_vars,
        }

    def close(self) -> None:
        pass
