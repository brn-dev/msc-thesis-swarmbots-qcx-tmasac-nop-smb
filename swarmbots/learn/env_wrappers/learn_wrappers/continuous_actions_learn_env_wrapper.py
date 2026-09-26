from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import (
    BaseLearnEnvWrapper,
)
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import to_backend_array


class ContinuousActionsLearnEnvWrapper(BaseLearnEnvWrapper):
    def __init__(
        self,
        env: VectorEnv,
        *,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError(
                f"Expected a continuous Box action space, got {env.action_space}"
            )
        if len(env.action_space.shape) != 3:
            raise ValueError(
                "Expected vector action shape (n_envs, n_agents, action_dim), "
                f"got {env.action_space.shape}"
            )
        if not np.issubdtype(env.action_space.dtype, np.floating):
            raise ValueError(
                f"Expected floating-point actions, got dtype={env.action_space.dtype}"
            )
        native_action_low = np.asarray(env.action_space.low, dtype=np.float32)
        native_action_high = np.asarray(env.action_space.high, dtype=np.float32)
        if not (
            np.all(np.isfinite(native_action_low))
            and np.all(np.isfinite(native_action_high))
        ):
            raise ValueError(
                "ContinuousActionsLearnEnvWrapper requires finite action bounds."
            )

        action_space = VectorHybridActionSpace(
            spaces={
                "actions": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=env.action_space.shape,
                    dtype=np.float32,
                )
            }
        )
        super().__init__(
            env=env,
            action_space=action_space,
            device=device,
            obs_dtype=obs_dtype,
            reward_dtype=reward_dtype,
        )
        self.action_dim = action_space.total_agent_action_dim
        self.action_backend = str(getattr(env, "action_backend", "numpy")).lower()
        self._action_center = torch.as_tensor(
            native_action_low + (native_action_high - native_action_low) / 2.0,
            device=self.device,
        )
        self._action_half_range = torch.as_tensor(
            (native_action_high - native_action_low) / 2.0,
            device=self.device,
        )

    def set_device(self, device: torch.device | str) -> None:
        super().set_device(device)
        self._action_center = self._action_center.to(self.device)
        self._action_half_range = self._action_half_range.to(self.device)

    def _actions_to_env(self, actions: torch.Tensor) -> Any:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        native_actions = (
            self._action_center + self._action_half_range * actions.detach()
        )
        return to_backend_array(
            native_actions,
            backend=self.action_backend,
            dtype=np.float32 if self.action_backend == "numpy" else None,
        )
