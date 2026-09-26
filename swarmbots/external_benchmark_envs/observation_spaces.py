from __future__ import annotations

import numpy as np
from gymnasium import spaces


def make_benchmark_observation_space(
    *,
    n_agents: int,
    local_obs_dim: int,
    global_state_dim: int,
) -> spaces.Dict:
    return spaces.Dict(
        {
            "local_obs": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_agents, local_obs_dim),
                dtype=np.float32,
            ),
            "global_obs": spaces.Box(
                low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
            ),
            "hidden_local_vars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_agents, 0),
                dtype=np.float32,
            ),
            "hidden_global_vars": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(global_state_dim,),
                dtype=np.float32,
            ),
        }
    )
