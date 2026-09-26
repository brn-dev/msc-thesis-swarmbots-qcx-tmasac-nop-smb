from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space


Array = np.ndarray | torch.Tensor


class MultiScenarioVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(self, envs: Mapping[str, VectorEnv]) -> None:
        if not envs:
            raise ValueError("MultiScenarioVectorEnv requires at least one scenario environment.")
        self.scenario_names = tuple(envs)
        self.envs = tuple(envs.values())
        self.scenario_ids_by_name = {
            scenario_name: scenario_id
            for scenario_id, scenario_name in enumerate(self.scenario_names)
        }
        self._scenario_num_envs = tuple(int(env.num_envs) for env in self.envs)
        if any(num_envs <= 0 for num_envs in self._scenario_num_envs):
            raise ValueError(
                "Every scenario environment must contain at least one lane, got "
                f"{dict(zip(self.scenario_names, self._scenario_num_envs, strict=True))}."
            )
        self.num_envs = sum(self._scenario_num_envs)
        self._scenario_slices = self._build_scenario_slices(self._scenario_num_envs)
        self._latest_child_observations: list[dict[str, Array] | None] = [None] * len(self.envs)
        self._validate_envs()
        self._deferred_child_steps = all(
            bool(getattr(env, "supports_deferred_step", False))
            for env in self.envs
        )

        first_env = self.envs[0]
        first_single_obs = first_env.single_observation_space
        assert isinstance(first_single_obs, spaces.Dict)
        self._field_dims = {
            key: tuple(int(env.single_observation_space[key].shape[-1]) for env in self.envs)
            for key in ("global_obs", "hidden_local_vars", "hidden_global_vars")
        }
        self._padded_field_dims = {
            key: max(dimensions)
            for key, dimensions in self._field_dims.items()
        }
        single_obs_spaces = dict(first_single_obs.spaces)
        for key, padded_dim in self._padded_field_dims.items():
            original_space = first_single_obs[key]
            assert isinstance(original_space, spaces.Box)
            shape = (*original_space.shape[:-1], padded_dim)
            single_obs_spaces[key] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=shape,
                dtype=original_space.dtype,
            )
        single_obs_spaces["scenario_id"] = spaces.Discrete(len(self.envs))
        self.single_observation_space = spaces.Dict(single_obs_spaces)
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

        self.single_action_space = deepcopy(first_env.single_action_space)
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)
        self.action_backend = str(getattr(first_env, "action_backend", "numpy")).lower()

    @property
    def scenario_observation_dims(self) -> dict[str, dict[str, int]]:
        return {
            scenario_name: {
                key: self._field_dims[key][scenario_id]
                for key in self._field_dims
            }
            for scenario_id, scenario_name in enumerate(self.scenario_names)
        }

    def reset(
            self,
            *,
            seed: int | list[int] | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Array], dict[str, Any]]:
        self._validate_reset_arguments(seed=seed, options=options)
        options = self._normalize_reset_options(options)
        child_observations: list[dict[str, Array]] = []
        child_infos: list[dict[str, Any]] = []
        for scenario_id, (env, scenario_slice) in enumerate(
                zip(self.envs, self._scenario_slices, strict=True)
        ):
            child_options = self._slice_reset_options(options, scenario_slice)
            if self._reset_mask_is_empty(child_options):
                cached_observations = self._latest_child_observations[scenario_id]
                if cached_observations is None:
                    raise ValueError(
                        "Cannot skip every lane of a scenario before that scenario has been reset."
                    )
                child_observations.append(cached_observations)
                child_infos.append({})
                continue
            child_seed: int | list[int] | None
            if seed is None:
                child_seed = None
            elif isinstance(seed, int):
                child_seed = seed + int(scenario_slice.start or 0)
            else:
                child_seed = seed[scenario_slice]
            obs, infos = env.reset(seed=child_seed, options=child_options)
            adapted_observations = self._adapt_observations(obs, scenario_id=scenario_id)
            self._latest_child_observations[scenario_id] = adapted_observations
            child_observations.append(adapted_observations)
            child_infos.append(infos)
        observations = self._concatenate_observations(child_observations)
        infos = self._merge_infos(child_infos)
        self._inject_scenario_info(infos, observations["scenario_id"])
        return observations, infos

    def step(
            self,
            actions: Mapping[str, Array],
    ) -> tuple[dict[str, Array], Array, Array, Array, dict[str, Any]]:
        child_actions = [
            {
                key: value[scenario_slice]
                for key, value in actions.items()
            }
            for scenario_slice in self._scenario_slices
        ]
        child_step_results = self._step_children(child_actions)
        child_observations: list[dict[str, Array]] = []
        child_rewards: list[Array] = []
        child_terminations: list[Array] = []
        child_truncations: list[Array] = []
        child_infos: list[dict[str, Any]] = []
        for scenario_id, (
                obs,
                rewards,
                terminations,
                truncations,
                infos,
        ) in enumerate(child_step_results):
            adapted_obs = self._adapt_observations(obs, scenario_id=scenario_id)
            self._latest_child_observations[scenario_id] = adapted_obs
            child_observations.append(adapted_obs)
            child_rewards.append(rewards)
            child_terminations.append(terminations)
            child_truncations.append(truncations)
            child_infos.append(
                self._adapt_final_observations(
                    infos,
                    scenario_id=scenario_id,
                )
            )

        self._normalize_final_observation_infos(
            child_infos=child_infos,
            child_observations=child_observations,
        )
        observations = self._concatenate_observations(child_observations)
        infos = self._merge_infos(child_infos)
        self._inject_scenario_info(infos, observations["scenario_id"])
        return (
            observations,
            self._concatenate(child_rewards),
            self._concatenate(child_terminations),
            self._concatenate(child_truncations),
            infos,
        )

    def _step_children(
            self,
            child_actions: Sequence[Mapping[str, Array]],
    ) -> list[tuple[dict[str, Array], Array, Array, Array, dict[str, Any]]]:
        if not self._deferred_child_steps:
            return [
                env.step(actions)
                for env, actions in zip(self.envs, child_actions, strict=True)
            ]

        begun_envs: list[VectorEnv] = []
        try:
            for env, actions in zip(self.envs, child_actions, strict=True):
                begin_step = getattr(env, "begin_step")
                begin_step(actions)
                begun_envs.append(env)

            results = []
            for env in self.envs:
                end_step = getattr(env, "end_step")
                results.append(end_step())
                begun_envs.remove(env)
            return results
        except Exception:
            for env in begun_envs:
                if not bool(getattr(env, "has_pending_step", False)):
                    continue
                with suppress(Exception):
                    getattr(env, "end_step")()
            raise

    def close_extras(self, **kwargs: Any) -> None:
        for env in self.envs:
            env.close(**kwargs)

    def get_settings(self) -> dict[str, Any]:
        return {
            "scenarios": {
                scenario_name: env.get_settings()
                for scenario_name, env in zip(self.scenario_names, self.envs, strict=True)
            },
            "scenario_ids": dict(self.scenario_ids_by_name),
            "scenario_num_envs": dict(zip(self.scenario_names, self._scenario_num_envs, strict=True)),
            "scenario_observation_dims": self.scenario_observation_dims,
        }

    def get_swarm_pool_size(self) -> int:
        return self._common_child_value("get_swarm_pool_size")

    def get_active_swarm_pool_size(self) -> int:
        return self._common_child_value("get_active_swarm_pool_size")

    def set_active_swarm_pool_size(self, active_pool_size: int) -> int:
        values = [
            int(env.set_active_swarm_pool_size(active_pool_size))
            for env in self.envs
        ]
        if len(set(values)) != 1:
            raise ValueError(
                f"Scenario environments resolved different active swarm pool sizes: {values}."
            )
        return values[0]

    def _validate_envs(self) -> None:
        first_env = self.envs[0]
        if not isinstance(first_env.single_observation_space, spaces.Dict):
            raise ValueError("Scenario environments must expose Dict single_observation_space values.")
        if "scenario_id" in first_env.single_observation_space.spaces:
            raise ValueError(
                "scenario_id is reserved for MultiScenarioVectorEnv and must not be provided by child envs."
            )
        for scenario_name, env in zip(self.scenario_names, self.envs, strict=True):
            if env.metadata.get("autoreset_mode") != AutoresetMode.SAME_STEP:
                raise ValueError(
                    f"Scenario {scenario_name!r} must use SAME_STEP autoreset."
                )
            if not isinstance(env.single_observation_space, spaces.Dict):
                raise ValueError(
                    f"Scenario {scenario_name!r} must expose a Dict single_observation_space."
                )
            if set(env.single_observation_space.keys()) != set(
                    first_env.single_observation_space.keys()
            ):
                raise ValueError(
                    "All scenarios must expose the same observation keys before padding."
                )
            for key in (
                "local_obs",
                "global_obs",
                "hidden_local_vars",
                "hidden_global_vars",
            ):
                if key not in env.single_observation_space.spaces:
                    raise ValueError(
                        f"Scenario {scenario_name!r} observation space is missing {key!r}."
                    )
            for key in ("global_obs", "hidden_local_vars", "hidden_global_vars"):
                scenario_space = env.single_observation_space[key]
                first_space = first_env.single_observation_space[key]
                if not isinstance(scenario_space, spaces.Box):
                    raise ValueError(f"Scenario observation {key!r} must be a Box space.")
                if (
                        scenario_space.shape[:-1] != first_space.shape[:-1]
                        or scenario_space.dtype != first_space.dtype
                ):
                    raise ValueError(
                        f"All scenarios must use the same {key} shape prefix and dtype."
                    )
            if env.single_observation_space["local_obs"] != first_env.single_observation_space["local_obs"]:
                raise ValueError("All scenarios must use the same local observation space.")
            env_agent_mask_space = (
                env.single_observation_space["agent_mask"]
                if "agent_mask" in env.single_observation_space.spaces
                else None
            )
            first_agent_mask_space = (
                first_env.single_observation_space["agent_mask"]
                if "agent_mask" in first_env.single_observation_space.spaces
                else None
            )
            if env_agent_mask_space != first_agent_mask_space:
                raise ValueError("All scenarios must use the same agent-mask space.")
            if env.single_action_space != first_env.single_action_space:
                raise ValueError("All scenarios must use the same action space.")
            action_backend = str(getattr(env, "action_backend", "numpy")).lower()
            first_backend = str(getattr(first_env, "action_backend", "numpy")).lower()
            if action_backend != first_backend:
                raise ValueError("All scenarios must use the same action backend.")

    def _common_child_value(self, method_name: str) -> int:
        values = [
            int(getattr(env, method_name)())
            for env in self.envs
        ]
        if len(set(values)) != 1:
            raise ValueError(
                f"Scenario environments returned different {method_name} values: {values}."
            )
        return values[0]

    def _adapt_observations(
            self,
            observations: Mapping[str, Array],
            *,
            scenario_id: int,
    ) -> dict[str, Array]:
        adapted = dict(observations)
        for key, padded_dim in self._padded_field_dims.items():
            adapted[key] = self._pad_last_dim(adapted[key], padded_dim=padded_dim)
        template = adapted["global_obs"]
        adapted["scenario_id"] = self._full(
            (template.shape[0],),
            scenario_id,
            like=template,
            dtype=torch.long if isinstance(template, torch.Tensor) else np.int64,
        )
        return adapted

    def _adapt_final_observations(
            self,
            infos: dict[str, Any],
            *,
            scenario_id: int,
    ) -> dict[str, Any]:
        adapted_infos = dict(infos)
        final_obs = infos.get("final_obs", None)
        if isinstance(final_obs, Mapping):
            adapted_infos["final_obs"] = self._adapt_observations(
                final_obs,
                scenario_id=scenario_id,
            )
        elif final_obs is None:
            return adapted_infos
        else:
            final_obs_entries = np.asarray(final_obs, dtype=object)
            adapted_entries = np.full(final_obs_entries.shape, None, dtype=object)
            for index in np.ndindex(final_obs_entries.shape):
                entry = final_obs_entries[index]
                if entry is not None:
                    if not isinstance(entry, Mapping):
                        raise ValueError(
                            "MultiScenarioVectorEnv requires dict-style final_obs entries."
                        )
                    adapted_entries[index] = self._adapt_single_observation(
                        entry,
                        scenario_id=scenario_id,
                    )
            adapted_infos["final_obs"] = adapted_entries
        return adapted_infos

    def _adapt_single_observation(
            self,
            observation: Mapping[str, Array],
            *,
            scenario_id: int,
    ) -> dict[str, Array]:
        adapted = dict(observation)
        for key, padded_dim in self._padded_field_dims.items():
            adapted[key] = self._pad_last_dim(adapted[key], padded_dim=padded_dim)
        template = adapted["global_obs"]
        adapted["scenario_id"] = self._full(
            (),
            scenario_id,
            like=template,
            dtype=torch.long if isinstance(template, torch.Tensor) else np.int64,
        )
        return adapted

    def _concatenate_observations(
            self,
            observations: Sequence[Mapping[str, Array]],
    ) -> dict[str, Array]:
        return {
            key: self._concatenate([obs[key] for obs in observations])
            for key in observations[0]
        }

    def _normalize_final_observation_infos(
            self,
            *,
            child_infos: Sequence[dict[str, Any]],
            child_observations: Sequence[Mapping[str, Array]],
    ) -> None:
        for child_idx, infos in enumerate(child_infos):
            final_obs = infos.get("final_obs", None)
            final_obs_mask = infos.get("_final_obs", None)
            if final_obs is None and final_obs_mask is not None and self._has_true(final_obs_mask):
                raise ValueError(
                    f"Scenario {self.scenario_names[child_idx]!r} reported final observations "
                    "in '_final_obs' but did not provide 'final_obs'."
                )
            if final_obs is not None and final_obs_mask is None:
                raise ValueError(
                    f"Scenario {self.scenario_names[child_idx]!r} provided 'final_obs' "
                    "without the required '_final_obs' mask."
                )

        present_final_observations = [
            infos["final_obs"]
            for infos in child_infos
            if "final_obs" in infos
        ]
        if not present_final_observations:
            return

        use_object_entries = any(
            not isinstance(final_obs, Mapping)
            for final_obs in present_final_observations
        )
        for child_idx, (infos, fallback_obs) in enumerate(
                zip(child_infos, child_observations, strict=True)
        ):
            final_obs = infos.get("final_obs", None)
            if use_object_entries:
                if isinstance(final_obs, Mapping):
                    infos["final_obs"] = self._batched_observations_to_object_entries(
                        final_obs,
                        num_envs=self._scenario_num_envs[child_idx],
                    )
                elif final_obs is None:
                    infos["final_obs"] = np.full(
                        self._scenario_num_envs[child_idx],
                        None,
                        dtype=object,
                    )
            elif final_obs is None:
                infos["final_obs"] = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value.copy()
                    for key, value in fallback_obs.items()
                }

            if "_final_obs" not in infos:
                template = fallback_obs["scenario_id"]
                infos["_final_obs"] = self._full(
                    template.shape,
                    False,
                    like=template,
                    dtype=torch.bool if isinstance(template, torch.Tensor) else np.bool_,
                )

    def _merge_infos(self, infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        keys = set().union(*(info.keys() for info in infos))
        for key in sorted(keys):
            values = [info.get(key, None) for info in infos]
            present = [value is not None for value in values]
            exemplar = next((value for value in values if value is not None), None)
            if isinstance(exemplar, Mapping):
                child_dicts = [
                    {} if value is None else dict(value)
                    for value in values
                ]
                merged[key] = self._merge_infos(child_dicts)
                continue
            if self._is_batched_value(exemplar):
                filled_values = [
                    self._zeros_for_missing(exemplar, num_envs)
                    if value is None
                    else value
                    for value, num_envs in zip(values, self._scenario_num_envs, strict=True)
                ]
                merged[key] = self._concatenate(filled_values)
                if (
                        not all(present)
                        and not key.startswith("_")
                        and f"_{key}" not in keys
                ):
                    merged[f"_{key}"] = self._concatenate([
                        self._full(
                            (num_envs,),
                            is_present,
                            like=filled_value,
                            dtype=(
                                torch.bool
                                if isinstance(filled_value, torch.Tensor)
                                else np.bool_
                            ),
                        )
                        for is_present, filled_value, num_envs in zip(
                            present,
                            filled_values,
                            self._scenario_num_envs,
                            strict=True,
                        )
                    ])
                continue
            merged[key] = values
        return merged

    @staticmethod
    def _batched_observations_to_object_entries(
            observations: Mapping[str, Array],
            *,
            num_envs: int,
    ) -> np.ndarray:
        entries = np.empty(num_envs, dtype=object)
        for env_idx in range(num_envs):
            entries[env_idx] = {
                key: (
                    value[env_idx].clone()
                    if isinstance(value, torch.Tensor)
                    else np.array(value[env_idx], copy=True)
                )
                for key, value in observations.items()
            }
        return entries

    def _inject_scenario_info(self, infos: dict[str, Any], scenario_ids: Array) -> None:
        infos["scenario_id"] = scenario_ids
        infos["scenario_name"] = np.concatenate([
            np.full(num_envs, scenario_name, dtype=object)
            for scenario_name, num_envs in zip(
                self.scenario_names,
                self._scenario_num_envs,
                strict=True,
            )
        ])

    @staticmethod
    def _build_scenario_slices(num_envs: Sequence[int]) -> tuple[slice, ...]:
        slices: list[slice] = []
        start = 0
        for count in num_envs:
            slices.append(slice(start, start + count))
            start += count
        return tuple(slices)

    @staticmethod
    def _slice_reset_options(
            options: dict[str, Any] | None,
            scenario_slice: slice,
    ) -> dict[str, Any] | None:
        if options is None:
            return None
        child_options = dict(options)
        reset_mask = child_options.get("reset_mask", None)
        if reset_mask is not None:
            child_options["reset_mask"] = reset_mask[scenario_slice]
        return child_options

    def _validate_reset_arguments(
            self,
            *,
            seed: int | list[int] | None,
            options: dict[str, Any] | None,
    ) -> None:
        if isinstance(seed, list) and len(seed) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} reset seeds, got {len(seed)}."
            )
        if isinstance(seed, list):
            unsupported_scenarios = [
                scenario_name
                for scenario_name, env in zip(
                    self.scenario_names,
                    self.envs,
                    strict=True,
                )
                if not bool(getattr(env, "supports_per_env_reset_seeds", True))
            ]
            if unsupported_scenarios:
                raise ValueError(
                    "Explicit per-environment reset seeds are not supported by scenario "
                    f"environments {unsupported_scenarios}. Use a single integer seed instead."
                )
        if options is None or "reset_mask" not in options:
            return

        reset_mask = options["reset_mask"]
        if not isinstance(reset_mask, (np.ndarray, torch.Tensor)):
            raise ValueError(
                "options['reset_mask'] must be a numpy array or torch tensor."
            )
        if tuple(reset_mask.shape) != (self.num_envs,):
            raise ValueError(
                f"Expected reset_mask shape ({self.num_envs},), got {tuple(reset_mask.shape)}."
            )
        expected_dtype = torch.bool if isinstance(reset_mask, torch.Tensor) else np.bool_
        if reset_mask.dtype != expected_dtype:
            raise ValueError(
                f"Expected reset_mask dtype {expected_dtype}, got {reset_mask.dtype}."
            )

    @staticmethod
    def _normalize_reset_options(
            options: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if options is None:
            return None
        reset_mask = options.get("reset_mask", None)
        if not isinstance(reset_mask, torch.Tensor):
            return options
        normalized_options = dict(options)
        normalized_options["reset_mask"] = reset_mask.detach().cpu().numpy()
        return normalized_options

    @staticmethod
    def _reset_mask_is_empty(options: dict[str, Any] | None) -> bool:
        if options is None:
            return False
        reset_mask = options.get("reset_mask", None)
        if reset_mask is None:
            return False
        return not bool(np.any(reset_mask))

    @staticmethod
    def _concatenate(values: Sequence[Array]) -> Array:
        if isinstance(values[0], torch.Tensor):
            return torch.cat(tuple(values), dim=0)
        return np.concatenate(tuple(np.asarray(value) for value in values), axis=0)

    @staticmethod
    def _pad_last_dim(value: Array, *, padded_dim: int) -> Array:
        if value.shape[-1] == padded_dim:
            return value
        shape = (*value.shape[:-1], padded_dim)
        if isinstance(value, torch.Tensor):
            padded = torch.zeros(shape, dtype=value.dtype, device=value.device)
        else:
            padded = np.zeros(shape, dtype=value.dtype)
        padded[..., :value.shape[-1]] = value
        return padded

    @staticmethod
    def _is_batched_value(value: Any) -> bool:
        return isinstance(value, (np.ndarray, torch.Tensor)) and value.ndim > 0

    @staticmethod
    def _has_true(value: Array) -> bool:
        if isinstance(value, torch.Tensor):
            return bool(torch.any(value).item())
        return bool(np.any(value))

    @staticmethod
    def _zeros_for_missing(exemplar: Array, num_envs: int) -> Array:
        shape = (num_envs, *exemplar.shape[1:])
        if isinstance(exemplar, torch.Tensor):
            return torch.zeros(shape, dtype=exemplar.dtype, device=exemplar.device)
        return np.zeros(shape, dtype=exemplar.dtype)

    @staticmethod
    def _full(
            shape: tuple[int, ...],
            value: Any,
            *,
            like: Array,
            dtype: torch.dtype | np.dtype[Any] | type[np.generic],
    ) -> Array:
        if isinstance(like, torch.Tensor):
            assert isinstance(dtype, torch.dtype)
            return torch.full(shape, value, dtype=dtype, device=like.device)
        return np.full(shape, value, dtype=dtype)
