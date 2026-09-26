from copy import deepcopy
from typing import Any, SupportsFloat, Literal, Iterable

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame
from mujoco import MjvOption

from swarmbots.mj_env.scenarios.base_scenario import (
    BaseScenario,
    RewardWeights,
    RewardWeightsUpdateResult,
    SwarmActDict,
    SwarmObsDict,
)
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.utils.recording_resolution import (
    DEFAULT_RECORDING_HEIGHT,
    DEFAULT_RECORDING_WIDTH,
    ensure_mujoco_offscreen_framebuffer,
)


class SwarmBotsEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        scenario: BaseScenario,
        episode_length: int = 500,
        render_mode: str | None = None,
        width: int = DEFAULT_RECORDING_WIDTH,
        height: int = DEFAULT_RECORDING_HEIGHT,
        camera: int | list[int] | Literal['all'] = 'all',
        scene_option: MjvOption = None,
        simulation_unstable_reward: float = -1.0,
        return_scenario_state_as_infos: bool = False,
        first_episode_length: int | None = None,
        reset_retry_count: int = 3,
    ):
        if first_episode_length is not None and first_episode_length > episode_length:
            raise ValueError(f'first_episode_length can not be longer than episode_length '
                             f'({episode_length}), got {first_episode_length}')
        if reset_retry_count <= 0:
            raise ValueError(f"Expected reset_retry_count > 0, got {reset_retry_count}")

        self.episode_length = episode_length
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.cameras = list(camera) if isinstance(camera, list) else [camera]
        self.scene_option = scene_option
        self.simulation_unstable_reward = simulation_unstable_reward
        self.return_scenario_state_as_infos = return_scenario_state_as_infos
        self.reset_retry_count = reset_retry_count

        self.first_episode_length = first_episode_length
        self.is_first_episode = True

        self.current_step = 0

        self.scenario = scenario
        # BaseScenario already compiles a model/data pair to derive indices and spaces.
        # Reusing that pair here avoids recompiling the same MuJoCo scene for every env.
        self.model = self.scenario.dummy_model
        self.data = self.scenario.dummy_data
        ensure_mujoco_offscreen_framebuffer(self.model, width=self.width, height=self.height)

        self.scenario_state: dict | None = None
        self.swarm_connections: SwarmConnections | None = None

        self.observation_space: spaces.Dict = scenario.get_obs_space()
        self.action_space: spaces.Dict = scenario.get_action_space()

        if isinstance(camera, int):
            self.cameras = [camera]
        if camera == 'all':
            self.cameras = list(range(self.scenario.swarm.config.num_units))
        elif isinstance(camera, Iterable):
            self.cameras = list(camera)

        self._renderer = None

        self._zeros_obs: SwarmObsDict = {
            "global_obs": np.zeros(
                self.observation_space["global_obs"].shape,
                dtype=self.observation_space["global_obs"].dtype,
            ),
            "local_obs": np.zeros(
                self.observation_space["local_obs"].shape,
                dtype=self.observation_space["local_obs"].dtype,
            ),
            "hidden_local_vars": np.zeros(
                self.observation_space["hidden_local_vars"].shape,
                dtype=self.observation_space["hidden_local_vars"].dtype,
            ),
            "hidden_global_vars": np.zeros(
                self.observation_space["hidden_global_vars"].shape,
                dtype=self.observation_space["hidden_global_vars"].dtype,
            ),
        }
        if "agent_mask" in self.observation_space.keys() and self.observation_space["agent_mask"] is not None:
            self._zeros_obs["agent_mask"] = np.ones(
                self.observation_space["agent_mask"].shape,
                dtype=bool,
            )

    @property
    def action_repeat(self) -> int:
        return self.scenario.action_repeat

    def get_settings(self):
        return {
            'scenario': self.scenario.get_settings(),
            'episode_length': self.episode_length,
            'action_repeat': self.action_repeat,
            'simulation_unstable_reward': self.simulation_unstable_reward,
            'swarm_pool': {
                'pool_size': self.get_swarm_pool_size(),
                'active_pool_size': self.get_active_swarm_pool_size(),
            },
        }

    def update_reward_weights(self, reward_weights: RewardWeights) -> RewardWeightsUpdateResult:
        return self.scenario.update_reward_weights(reward_weights)

    def get_reward_weights(self) -> RewardWeights:
        return self.scenario.get_reward_weights()

    def get_swarm_pool_size(self) -> int | None:
        get_pool_size = getattr(self.scenario.swarm, "get_pool_size", None)
        if not callable(get_pool_size):
            return None
        return get_pool_size()

    def get_active_swarm_pool_size(self) -> int | None:
        get_active_pool_size = getattr(self.scenario.swarm, "get_active_pool_size", None)
        if not callable(get_active_pool_size):
            return None
        return get_active_pool_size()

    def set_active_swarm_pool_size(self, active_pool_size: int) -> int:
        set_active_pool_size = getattr(self.scenario.swarm, "set_active_pool_size", None)
        if not callable(set_active_pool_size):
            raise ValueError("Scenario swarm does not support a fixed starting-configuration pool.")
        return int(set_active_pool_size(active_pool_size))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SwarmObsDict, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.scenario.rng = np.random.default_rng(seed)
        self.current_step = 0

        last_error: mujoco.FatalError | None = None
        for _ in range(self.reset_retry_count):
            try:
                self.scenario_state, self.swarm_connections = self.scenario.reset_scenario(self.model, self.data)
                last_error = None
                break
            except mujoco.FatalError as error:
                last_error = error

        if last_error is not None:
            raise last_error

        obs = self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections)
        return obs, {}

    def step(
        self, action: SwarmActDict
    ) -> tuple[SwarmObsDict, SupportsFloat, bool, bool, dict[str, Any]]:
        self.scenario.apply_action(
            model=self.model,
            data=self.data,
            action=action,
            state=self.scenario_state,
            connections=self.swarm_connections,
        )

        mujoco.mj_step(self.model, self.data, self.action_repeat)

        mj_warning: mujoco.MjWarningStat = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC]
        if mj_warning.number > 0 or np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            print('Simulation Unstable!')
            # noinspection PyTypeChecker
            err_obs: SwarmObsDict = {key: value.copy() for key, value in self._zeros_obs.items()}
            if "agent_mask" in err_obs:
                units_active_mask = self.scenario_state.get("units_active_mask")
                if units_active_mask is not None:
                    err_obs["agent_mask"] = np.asarray(units_active_mask, dtype=bool).copy()
            return (
                err_obs,
                self.simulation_unstable_reward,
                True,
                False,
                {"error": "simulation_unstable", "mj_warning.lastinfo": mj_warning.lastinfo},
            )

        reward, terminated = self.scenario.evaluate_step(
            action, self.model, self.data, self.scenario_state, self.swarm_connections
        )

        self.current_step += 1
        if self.first_episode_length is not None and self.is_first_episode:
            truncated = self.current_step >= self.first_episode_length
        else:
            truncated = self.current_step >= self.episode_length

        if terminated or truncated:
            self.is_first_episode = False

        if self.render_mode == "human":
            self.render()

        obs = self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections).copy()

        if self.return_scenario_state_as_infos:
            info: dict[str, Any] = dict(self.scenario_state)
        else:
            info = {}
        info["progress_reward"] = float(self.scenario_state["weighted_progress_reward"])
        weighted_forward_reward = self.scenario_state.get("weighted_forward_reward")
        if weighted_forward_reward is None:
            weighted_forward_reward = self.scenario_state.get("weighted_forward_progress_reward")
        if weighted_forward_reward is not None:
            info["forward_reward"] = float(weighted_forward_reward)
            info["forward_progress_reward"] = float(weighted_forward_reward)
        weighted_wall_pass_reward = self.scenario_state.get("weighted_wall_pass_reward")
        if weighted_wall_pass_reward is not None:
            info["wall_pass_reward"] = float(weighted_wall_pass_reward)
        weighted_wall_climb_reward = self.scenario_state.get("weighted_wall_climb_reward")
        if weighted_wall_climb_reward is not None:
            info["wall_climb_reward"] = float(weighted_wall_climb_reward)
        weighted_wall_success_reward = self.scenario_state.get("weighted_wall_success_reward")
        if weighted_wall_success_reward is not None:
            info["wall_success_reward"] = float(weighted_wall_success_reward)
        weighted_goal_success_reward = self.scenario_state.get("weighted_goal_success_reward")
        if weighted_goal_success_reward is not None:
            info["goal_success_reward"] = float(weighted_goal_success_reward)
        weighted_units_without_connections_reward = self.scenario_state.get("weighted_units_without_connections_reward")
        if weighted_units_without_connections_reward is not None:
            info["units_without_connections_reward"] = float(weighted_units_without_connections_reward)
        success = self.scenario_state.get("success")
        if success is not None:
            info["success"] = bool(success)
        info["guidance_reward"] = float(self.scenario_state["weighted_guidance_reward"])
        reward_terms = self.scenario_state.get("reward_terms")
        if isinstance(reward_terms, dict):
            info["reward_terms"] = {
                str(label): float(value)
                for label, value in reward_terms.items()
            }

        return obs, reward, terminated, truncated, info

    def render(
            self,
            scene_option: mujoco.MjvOption = None
    ) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )

        scene_option = scene_option or self.scene_option

        frames = []

        if self.render_mode == "depth_array":
            self._renderer.enable_depth_rendering()

        for camera in self.cameras:
            self._renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
            if self.render_mode != "depth_array":
                self.scenario.add_render_geoms(self._renderer.scene)
            frames.append(self._renderer.render())

        if self.render_mode == "depth_array":
            self._renderer.disable_depth_rendering()

        return np.concatenate(frames, axis=1)

    def set_render_resolution(self, *, width: int, height: int) -> None:
        self.width = int(width)
        self.height = int(height)
        ensure_mujoco_offscreen_framebuffer(self.model, width=self.width, height=self.height)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()

    def clone_for_worker_pool(self) -> "SwarmBotsEnv":
        """Clone an already-built env without rebuilding the underlying scenario."""
        clone = object.__new__(type(self))
        for attr_name, attr_value in self.__dict__.items():
            if attr_name in {"scenario", "model", "data", "_renderer"}:
                continue
            clone.__dict__[attr_name] = deepcopy(attr_value)

        clone.scenario = self.scenario.clone_for_worker_pool()
        clone.model = clone.scenario.dummy_model
        clone.data = clone.scenario.dummy_data
        ensure_mujoco_offscreen_framebuffer(clone.model, width=clone.width, height=clone.height)
        clone._renderer = None
        return clone
