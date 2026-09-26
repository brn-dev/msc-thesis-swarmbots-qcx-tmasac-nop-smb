import abc
import itertools
from typing import Any, Generic, TypeAlias, TypeVar

import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import (
    normalize_reset_mask_options,
    to_torch_tensor,
)
from swarmbots.learn.torch_device import as_device

TorchObs: TypeAlias = dict[str, torch.Tensor]
ArrayObs: TypeAlias = dict[str, Any]

ActSpace = TypeVar("ActSpace", bound=VectorHybridActionSpace)


class BaseLearnEnvWrapper(VectorWrapper, Generic[ActSpace], abc.ABC):
    def __init__(
        self,
        env: VectorEnv,
        action_space: VectorHybridActionSpace,
        *,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(env)
        self.device = as_device(device)
        self.obs_dtype = obs_dtype
        self.reward_dtype = reward_dtype

        autoreset_mode = env.metadata.get("autoreset_mode", None)
        if autoreset_mode != AutoresetMode.SAME_STEP:
            raise ValueError(
                "SwarmBotsLearnVectorEnv requires VectorEnv autoreset_mode=SAME_STEP "
                f"(got autoreset_mode={autoreset_mode!r}). The rollout pipeline expects terminal observations in "
                "infos['final_obs'] and immediate same-step resets."
            )

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError(
                f"Observation space must be a gymnasium.spaces.Dict, got {obs_space}"
            )

        required_obs_keys = {
            "local_obs",
            "global_obs",
            "hidden_local_vars",
            "hidden_global_vars",
        }
        missing_obs_keys = required_obs_keys.difference(obs_space.keys())
        if missing_obs_keys:
            raise ValueError(
                f"Observation space is missing required keys {sorted(missing_obs_keys)}, got keys {list(obs_space.keys())}"
            )

        self._observation_space: spaces.Dict = obs_space

        local_obs_shape = self._observation_space["local_obs"].shape
        if len(local_obs_shape) != 3:
            raise ValueError(
                f"Expected local_obs shape (n_envs, n_agents, n_local_obs), got {local_obs_shape}"
            )
        global_obs_shape = self._observation_space["global_obs"].shape
        if len(global_obs_shape) != 2:
            raise ValueError(
                f"Expected global_obs shape (n_envs, n_global_obs), got {global_obs_shape}"
            )
        hidden_local_vars_shape = self._observation_space["hidden_local_vars"].shape
        if len(hidden_local_vars_shape) != 3:
            raise ValueError(
                "Expected hidden_local_vars shape (n_envs, n_agents, n_hidden_local_vars), "
                f"got {hidden_local_vars_shape}"
            )
        hidden_global_vars_shape = self._observation_space["hidden_global_vars"].shape
        if len(hidden_global_vars_shape) != 2:
            raise ValueError(
                "Expected hidden_global_vars shape (n_envs, n_hidden_global_vars), "
                f"got {hidden_global_vars_shape}"
            )

        self.local_obs_dim = local_obs_shape[2]
        self.global_obs_dim = global_obs_shape[1]
        self.hidden_local_vars_dim = hidden_local_vars_shape[2]
        self.hidden_global_vars_dim = hidden_global_vars_shape[1]
        self.has_scenario_id = "scenario_id" in self._observation_space.keys()
        self.scenario_names = getattr(env, "scenario_names", None)
        self.scenario_observation_dims = getattr(env, "scenario_observation_dims", None)
        if self.has_scenario_id:
            scenario_id_shape = self._observation_space["scenario_id"].shape
            if tuple(scenario_id_shape) != (local_obs_shape[0],):
                raise ValueError(
                    f"Expected scenario_id shape ({local_obs_shape[0]},), got {scenario_id_shape}"
                )
        self.has_agent_mask = "agent_mask" in self._observation_space.keys()
        if self.has_agent_mask:
            agent_mask_shape = self._observation_space["agent_mask"].shape
            if len(agent_mask_shape) != 2:
                raise ValueError(
                    f"Expected agent_mask shape (n_envs, n_agents), got {agent_mask_shape}"
                )

        if not isinstance(action_space, VectorHybridActionSpace):
            raise ValueError(
                f"Action space must be a VectorHybridActionSpace, got {action_space}"
            )
        self._action_space: VectorHybridActionSpace = action_space

        self._n_envs: int = int(getattr(env, "num_envs"))

        for space in itertools.chain(obs_space.values(), action_space.values()):
            assert space.shape[0] == self._n_envs, space

        self.n_agents = obs_space["local_obs"].shape[1]
        if hidden_local_vars_shape[1] != self.n_agents:
            raise ValueError(
                f"Expected hidden_local_vars second dim to match n_agents ({self.n_agents}), "
                f"got {hidden_local_vars_shape}"
            )

        if self.has_agent_mask:
            agent_mask_shape = self._observation_space["agent_mask"].shape
            if agent_mask_shape[1] != self.n_agents:
                raise ValueError(
                    f"Expected agent_mask second dim to match n_agents ({self.n_agents}), got {agent_mask_shape}"
                )

        for space in action_space.values():
            assert space.shape[1] == self.n_agents, space
            assert len(space.shape) == 3, space

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> VectorHybridActionSpace:
        return self._action_space

    def reset(self, **kwargs) -> tuple[TorchObs, dict[str, Any]]:
        options = normalize_reset_mask_options(
            kwargs.get("options", None), num_envs=self._n_envs
        )
        if options is not kwargs.get("options", None):
            kwargs = dict(kwargs)
            kwargs["options"] = options
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_torch(obs), info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        env_actions = self._actions_to_env(actions)
        obs, rewards, terminations, truncations, infos = self.env.step(env_actions)

        obs_t = self._obs_to_torch(obs)
        rewards_t = to_torch_tensor(
            rewards, device=self.device, dtype=self.reward_dtype
        ).reshape(self._n_envs)
        term_t = to_torch_tensor(
            terminations, device=self.device, dtype=torch.bool
        ).reshape(self._n_envs)
        trunc_t = to_torch_tensor(
            truncations, device=self.device, dtype=torch.bool
        ).reshape(self._n_envs)

        return obs_t, rewards_t, term_t, trunc_t, infos

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def set_device(self, device: torch.device | str) -> None:
        self.device = as_device(device)

    def _obs_to_torch(self, obs: ArrayObs) -> TorchObs:
        obs_t: TorchObs = {
            "local_obs": to_torch_tensor(
                obs["local_obs"], device=self.device, dtype=self.obs_dtype
            ),
            "global_obs": to_torch_tensor(
                obs["global_obs"], device=self.device, dtype=self.obs_dtype
            ),
            "hidden_local_vars": to_torch_tensor(
                obs["hidden_local_vars"], device=self.device, dtype=self.obs_dtype
            ),
            "hidden_global_vars": to_torch_tensor(
                obs["hidden_global_vars"], device=self.device, dtype=self.obs_dtype
            ),
        }
        if "agent_mask" in obs and obs["agent_mask"] is not None:
            obs_t["agent_mask"] = to_torch_tensor(
                obs["agent_mask"], device=self.device, dtype=torch.bool
            )
        if "scenario_id" in obs and obs["scenario_id"] is not None:
            obs_t["scenario_id"] = to_torch_tensor(
                obs["scenario_id"], device=self.device, dtype=torch.long
            )
        return obs_t

    @abc.abstractmethod
    def _actions_to_env(self, actions: torch.Tensor) -> Any:
        raise NotImplementedError()
