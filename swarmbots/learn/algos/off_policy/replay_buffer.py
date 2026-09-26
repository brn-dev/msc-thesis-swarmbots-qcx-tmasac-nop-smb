from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from gymnasium import spaces

from swarmbots.learn.algos.off_policy.replay_buffer_tensor_ops import (
    build_replay_buffer_tensor_operations,
)
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_device import as_device


MaybeTensor = torch.Tensor | None


class NoEpisodeSegmentCandidatesError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OffPolicyReplayBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    previous_actions: MaybeTensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    episode_start_mask: torch.Tensor | None = None
    scenario_ids: MaybeTensor = None
    next_scenario_ids: MaybeTensor = None

    @property
    def episode_ends(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)

    @property
    def terminal_mask(self) -> torch.Tensor:
        return self.terminations


@dataclass(frozen=True, slots=True)
class OffPolicyReplayEpisodeSegmentBatch:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    actions: torch.Tensor
    rewards: torch.Tensor
    terminations: torch.Tensor
    truncations: torch.Tensor
    previous_actions: MaybeTensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor
    next_hidden_local_vars: torch.Tensor
    next_hidden_global_vars: torch.Tensor
    next_agent_mask: MaybeTensor
    episode_start_mask: MaybeTensor
    train_mask: torch.Tensor
    initial_temporal_state: Any
    burn_in_steps: int
    scenario_ids: MaybeTensor = None
    next_scenario_ids: MaybeTensor = None

    @property
    def sequence_length(self) -> int:
        return int(self.actions.shape[1])

    @property
    def segment_length(self) -> int:
        return self.sequence_length - self.burn_in_steps

    @property
    def burn_in_mask(self) -> torch.Tensor:
        return torch.logical_not(self.train_mask)

    @property
    def episode_ends(self) -> torch.Tensor:
        return torch.logical_or(self.terminations, self.truncations)

    @property
    def terminal_mask(self) -> torch.Tensor:
        return self.terminations

    @property
    def origin_batch(self) -> OffPolicyReplayBatch:
        return OffPolicyReplayBatch(
            local_obs=self.local_obs[:, 0],
            global_obs=self.global_obs[:, 0],
            hidden_local_vars=self.hidden_local_vars[:, 0],
            hidden_global_vars=self.hidden_global_vars[:, 0],
            agent_mask=None if self.agent_mask is None else self.agent_mask[:, 0],
            actions=self.actions[:, 0],
            rewards=self.rewards[:, 0],
            terminations=self.terminations[:, 0],
            truncations=self.truncations[:, 0],
            previous_actions=None if self.previous_actions is None else self.previous_actions[:, 0],
            next_local_obs=self.next_local_obs[:, 0],
            next_global_obs=self.next_global_obs[:, 0],
            next_hidden_local_vars=self.next_hidden_local_vars[:, 0],
            next_hidden_global_vars=self.next_hidden_global_vars[:, 0],
            next_agent_mask=None if self.next_agent_mask is None else self.next_agent_mask[:, 0],
            episode_start_mask=(
                None if self.episode_start_mask is None else self.episode_start_mask[:, 0]
            ),
            scenario_ids=None if self.scenario_ids is None else self.scenario_ids[:, 0],
            next_scenario_ids=None if self.next_scenario_ids is None else self.next_scenario_ids[:, 0],
        )


class OffPolicyReplayBuffer:
    def __init__(
            self,
            capacity_per_env: int,
            observation_space: spaces.Dict,
            action_space: VectorHybridActionSpace,
            *,
            store_previous_actions: bool = False,
            temporal_state_store_interval: int | None = None,
            temporal_state_storage_dtype: torch.dtype | None = None,
            storage_device: torch.device | str = "cpu",
            storage_dtype: torch.dtype = torch.float32,
            storage_pin_memory: bool = False,
            train_device: torch.device | str = "auto",
            train_dtype: torch.dtype = torch.float32,
            non_blocking_train_transfer: bool | None = None,
            compile_tensor_operations: bool | None = None,
    ) -> None:
        if capacity_per_env <= 0:
            raise ValueError(f"capacity_per_env must be > 0, got {capacity_per_env}")
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(f"observation_space must be a gymnasium.spaces.Dict, got {observation_space}")
        if temporal_state_store_interval is not None and temporal_state_store_interval <= 0:
            raise ValueError(
                f"temporal_state_store_interval must be > 0 when set, got {temporal_state_store_interval}"
            )
        if (
                temporal_state_storage_dtype is not None
                and not temporal_state_storage_dtype.is_floating_point
        ):
            raise ValueError(
                "temporal_state_storage_dtype must be a floating-point dtype, got "
                f"{temporal_state_storage_dtype}"
            )

        self.capacity_per_env = capacity_per_env
        self.observation_capacity_per_env = capacity_per_env + 1
        self.observation_space = observation_space
        self.action_space = action_space
        self.store_previous_actions = store_previous_actions
        self.temporal_state_store_interval = temporal_state_store_interval
        self.temporal_state_storage_dtype = (
            storage_dtype
            if temporal_state_storage_dtype is None
            else temporal_state_storage_dtype
        )
        self.storage_device = as_device(storage_device)
        self.storage_dtype = storage_dtype
        self.storage_pin_memory = storage_pin_memory
        self.train_device = as_device(train_device)
        self.train_dtype = train_dtype
        if self.storage_pin_memory and self.storage_device.type != "cpu":
            raise ValueError("storage_pin_memory=True requires storage_device='cpu'.")
        self.non_blocking_train_transfer = (
            self.storage_pin_memory and self.train_device.type == "cuda"
            if non_blocking_train_transfer is None
            else non_blocking_train_transfer
        )
        self.compile_tensor_operations = (
            self.storage_device.type == "cuda"
            if compile_tensor_operations is None
            else compile_tensor_operations
        )
        self._tensor_operations = build_replay_buffer_tensor_operations(
            compile_operations=self.compile_tensor_operations,
        )

        self.local_obs_space = observation_space["local_obs"]
        self.n_envs = int(self.local_obs_space.shape[0])
        self.n_agents = int(self.local_obs_space.shape[1])
        self.agent_obs_shape = tuple(self.local_obs_space.shape[2:])
        self.global_obs_shape = tuple(observation_space["global_obs"].shape[1:])
        self.hidden_local_vars_shape = tuple(observation_space["hidden_local_vars"].shape[2:])
        self.hidden_global_vars_shape = tuple(observation_space["hidden_global_vars"].shape[1:])
        self.has_scenario_ids = "scenario_id" in observation_space.keys()
        self.has_agent_mask = "agent_mask" in observation_space.keys() and observation_space["agent_mask"] is not None

        if action_space.n_envs != self.n_envs:
            raise ValueError(f"Expected action_space.n_envs={self.n_envs}, got {action_space.n_envs}")
        if action_space.n_agents != self.n_agents:
            raise ValueError(f"Expected action_space.n_agents={self.n_agents}, got {action_space.n_agents}")
        self.n_agent_actions = action_space.total_agent_action_dim

        obs_shape = (self.n_envs, self.observation_capacity_per_env)
        transition_shape = (self.n_envs, self.capacity_per_env)
        self.local_obs = self._new_storage_tensor(
            (*obs_shape, self.n_agents, *self.agent_obs_shape),
            dtype=self.storage_dtype,
        )
        self.global_obs = self._new_storage_tensor(
            (*obs_shape, *self.global_obs_shape),
            dtype=self.storage_dtype,
        )
        self.hidden_local_vars = self._new_storage_tensor(
            (*obs_shape, self.n_agents, *self.hidden_local_vars_shape),
            dtype=self.storage_dtype,
        )
        self.hidden_global_vars = self._new_storage_tensor(
            (*obs_shape, *self.hidden_global_vars_shape),
            dtype=self.storage_dtype,
        )
        self.scenario_ids: MaybeTensor = None
        if self.has_scenario_ids:
            self.scenario_ids = self._new_storage_tensor(obs_shape, dtype=torch.long)
        self.agent_mask: MaybeTensor = None
        if self.has_agent_mask:
            self.agent_mask = self._new_storage_tensor(
                (*obs_shape, self.n_agents),
                dtype=torch.bool,
            )

        self.actions = self._new_storage_tensor(
            (*transition_shape, self.n_agents, self.n_agent_actions),
            dtype=self.storage_dtype,
        )
        self.rewards = self._new_storage_tensor(transition_shape, dtype=self.storage_dtype)
        self.terminations = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.truncations = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.episode_starts: MaybeTensor = None
        if temporal_state_store_interval is not None:
            self.episode_starts = self._new_storage_tensor(transition_shape, dtype=torch.bool)
        self.previous_actions: MaybeTensor = None
        if store_previous_actions:
            self.previous_actions = self._new_storage_tensor(
                (*transition_shape, self.n_agents, self.n_agent_actions),
                dtype=self.storage_dtype,
            )
        self.temporal_states: Any = None
        self._temporal_state_available: torch.Tensor | None = None
        self._temporal_state_indices: torch.Tensor | None = None
        self._temporal_state_slots_in_use: torch.Tensor | None = None
        self.temporal_state_capacity_per_env = 0
        if temporal_state_store_interval is not None:
            self._temporal_state_available = self._new_storage_tensor(obs_shape, dtype=torch.bool)
            self._temporal_state_indices = torch.full(
                obs_shape,
                -1,
                dtype=torch.long,
                device=self.storage_device,
            )
            self.temporal_state_capacity_per_env = (
                self.observation_capacity_per_env + temporal_state_store_interval - 1
            ) // temporal_state_store_interval
            self._temporal_state_slots_in_use = self._new_storage_tensor(
                (self.n_envs, self.temporal_state_capacity_per_env),
                dtype=torch.bool,
            )

        self._write_slot = 0
        self._current_obs_slots = torch.zeros(self.n_envs, dtype=torch.long, device=self.storage_device)
        self._obs_write_slots = torch.zeros(self.n_envs, dtype=torch.long, device=self.storage_device)
        self._transition_obs_slots = torch.zeros(
            transition_shape,
            dtype=torch.long,
            device=self.storage_device,
        )
        self._transition_next_obs_slots = torch.zeros(
            transition_shape,
            dtype=torch.long,
            device=self.storage_device,
        )
        self._terminal_obs_indices = torch.full(
            transition_shape,
            -1,
            dtype=torch.long,
            device=self.storage_device,
        )
        initial_terminal_capacity = min(self.total_capacity, max(64, self.n_envs * 4))
        self._terminal_obs_slots_in_use = self._new_storage_tensor(
            (initial_terminal_capacity,),
            dtype=torch.bool,
        )
        self._terminal_local_obs = self._new_storage_tensor(
            (initial_terminal_capacity, self.n_agents, *self.agent_obs_shape),
            dtype=self.storage_dtype,
        )
        self._terminal_global_obs = self._new_storage_tensor(
            (initial_terminal_capacity, *self.global_obs_shape),
            dtype=self.storage_dtype,
        )
        self._terminal_hidden_local_vars = self._new_storage_tensor(
            (initial_terminal_capacity, self.n_agents, *self.hidden_local_vars_shape),
            dtype=self.storage_dtype,
        )
        self._terminal_hidden_global_vars = self._new_storage_tensor(
            (initial_terminal_capacity, *self.hidden_global_vars_shape),
            dtype=self.storage_dtype,
        )
        self._terminal_scenario_ids: MaybeTensor = None
        if self.scenario_ids is not None:
            self._terminal_scenario_ids = self._new_storage_tensor(
                (initial_terminal_capacity,),
                dtype=torch.long,
            )
        self._terminal_agent_mask: MaybeTensor = None
        if self.agent_mask is not None:
            self._terminal_agent_mask = self._new_storage_tensor(
                (initial_terminal_capacity, self.n_agents),
                dtype=torch.bool,
            )
        self._segment_candidate_cache: dict[
            tuple[int, int, bool, bool, int | None],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._size_per_env = 0
        self._size_per_env_tensor = torch.zeros((), dtype=torch.long, device=self.storage_device)
        self._logical_transition_slot_offset = torch.zeros(
            (),
            dtype=torch.long,
            device=self.storage_device,
        )
        self._has_current_obs = False
        self._current_episode_start_mask: MaybeTensor = None
        if temporal_state_store_interval is not None:
            self._current_episode_start_mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.storage_device)
        self._vector_steps_added = 0
        self.total_transitions_added = 0

    @property
    def size_per_env(self) -> int:
        return self._size_per_env

    @property
    def has_current_obs(self) -> bool:
        return self._has_current_obs

    def __len__(self) -> int:
        return self._size_per_env * self.n_envs

    @property
    def total_capacity(self) -> int:
        return self.capacity_per_env * self.n_envs

    def reset(self) -> None:
        self._write_slot = 0
        self._current_obs_slots.zero_()
        self._obs_write_slots.zero_()
        self._transition_obs_slots.zero_()
        self._transition_next_obs_slots.zero_()
        if self.episode_starts is not None:
            self.episode_starts.zero_()
        if self._temporal_state_available is not None:
            self._temporal_state_available.zero_()
        if self._temporal_state_indices is not None:
            self._temporal_state_indices.fill_(-1)
        if self._temporal_state_slots_in_use is not None:
            self._temporal_state_slots_in_use.zero_()
        self.temporal_states = None
        self._terminal_obs_indices.fill_(-1)
        self._terminal_obs_slots_in_use.zero_()
        self._segment_candidate_cache.clear()
        self._size_per_env = 0
        self._size_per_env_tensor.zero_()
        self._logical_transition_slot_offset.zero_()
        self._has_current_obs = False
        if self._current_episode_start_mask is not None:
            self._current_episode_start_mask.fill_(True)
        self._vector_steps_added = 0
        self.total_transitions_added = 0

    def add(
            self,
            *,
            obs: dict[str, torch.Tensor],
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminations: torch.Tensor,
            truncations: torch.Tensor,
            next_obs: dict[str, torch.Tensor],
            terminal_obs: dict[str, torch.Tensor] | None = None,
            previous_actions: MaybeTensor = None,
            episode_start_mask: torch.Tensor | None = None,
            temporal_state: Any = None,
            next_temporal_state: Any = None,
            copy_current_obs: bool | None = None,
    ) -> None:
        """
        Append one full vector-env step from a continuous stream.

        After the stream is initialized, the current obs for this transition is the previous appended next_obs/reset
        obs. In that normal streaming case, this method ignores obs for storage and does not copy it into replay again.
        terminal_obs may either be full vector-env observations or rows packed in done-env order.
        """
        slot = self._write_slot
        dones = torch.logical_or(terminations, truncations)
        done_env_indices = torch.nonzero(dones, as_tuple=False).flatten()
        should_copy_current_obs = not self._has_current_obs if copy_current_obs is None else copy_current_obs
        storage_episode_start_mask: torch.Tensor | None = None
        if self.episode_starts is not None:
            if episode_start_mask is None:
                assert self._current_episode_start_mask is not None
                storage_episode_start_mask = self._current_episode_start_mask.clone()
            else:
                storage_episode_start_mask = episode_start_mask.to(device=self.storage_device, dtype=torch.bool)

        if len(done_env_indices) > 0 and terminal_obs is None:
            raise ValueError("terminal_obs is required for done transitions under SAME_STEP autoreset.")
        if not should_copy_current_obs and not self._has_current_obs:
            raise ValueError("copy_current_obs=False requires the current observation slot to be initialized.")
        if copy_current_obs is True and self._has_current_obs:
            raise ValueError(
                "copy_current_obs=True is only valid when starting a replay stream. Continue with "
                "copy_current_obs=False/None so next_obs/reset obs becomes the next transition's current obs."
            )

        self._clear_terminal_obs_for_slot_(slot)

        if should_copy_current_obs:
            self._current_obs_slots = self._obs_write_slots.clone()
            self._copy_obs_rows_at_slots_(obs=obs, obs_slots=self._current_obs_slots)
            self._obs_write_slots = self._advance_obs_slots(self._obs_write_slots)

        transition_obs_slots = self._current_obs_slots.clone()
        next_obs_slots = self._obs_write_slots.clone()
        self._copy_obs_rows_at_slots_(obs=next_obs, obs_slots=next_obs_slots)
        self._obs_write_slots = self._advance_obs_slots(self._obs_write_slots)

        if len(done_env_indices) > 0:
            assert terminal_obs is not None
            storage_done_env_indices = done_env_indices.to(device=self.storage_device)
            self._store_terminal_obs_at_slot_(
                slot=slot,
                terminal_obs=terminal_obs,
                done_env_indices=storage_done_env_indices,
            )

        self._current_obs_slots = next_obs_slots

        self._transition_obs_slots[:, slot] = transition_obs_slots
        self._transition_next_obs_slots[:, slot] = next_obs_slots
        if self.episode_starts is not None:
            assert storage_episode_start_mask is not None
            self.episode_starts[:, slot] = storage_episode_start_mask

        if self._should_store_temporal_state(self._vector_steps_added) and temporal_state is not None:
            self._copy_temporal_state_rows_at_slots_(state=temporal_state, obs_slots=transition_obs_slots)
        if self._should_store_temporal_state(self._vector_steps_added + 1) and next_temporal_state is not None:
            self._copy_temporal_state_rows_at_slots_(state=next_temporal_state, obs_slots=next_obs_slots)

        self.actions[:, slot] = actions.to(device=self.storage_device, dtype=self.storage_dtype)
        self.rewards[:, slot] = rewards.to(device=self.storage_device, dtype=self.storage_dtype)
        self.terminations[:, slot] = terminations.to(device=self.storage_device, dtype=torch.bool)
        self.truncations[:, slot] = truncations.to(device=self.storage_device, dtype=torch.bool)

        if self.previous_actions is not None:
            if previous_actions is None:
                self.previous_actions[:, slot].zero_()
            else:
                self.previous_actions[:, slot] = previous_actions.to(device=self.storage_device, dtype=self.storage_dtype)

        self._write_slot = (slot + 1) % self.capacity_per_env
        if self._size_per_env < self.capacity_per_env:
            self._size_per_env += 1
            self._size_per_env_tensor.fill_(self._size_per_env)
        if self._size_per_env == self.capacity_per_env:
            self._logical_transition_slot_offset.fill_(self._write_slot)
        self._has_current_obs = True
        if self._current_episode_start_mask is not None:
            self._current_episode_start_mask = dones.to(device=self.storage_device, dtype=torch.bool)
        self._vector_steps_added += 1
        self.total_transitions_added += self.n_envs
        self._segment_candidate_cache.clear()

    def sample(
            self,
            batch_size: int,
            *,
            replacement: bool = True,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayBatch:
        indices = self._sample_transition_indices(
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        return self._fetch_indices(indices)

    def sample_episode_windows(
            self,
            batch_size: int,
            *,
            num_next_steps: int,
            replacement: bool = True,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        if num_next_steps <= 0:
            raise ValueError(f"num_next_steps must be > 0, got {num_next_steps}")
        indices = self._sample_transition_indices(
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        return self._fetch_episode_windows(indices, num_next_steps=num_next_steps)

    def sample_episode_segments(
            self,
            batch_size: int,
            *,
            segment_length: int,
            burn_in_steps: int = 0,
            replacement: bool = True,
            require_initial_temporal_state: bool = True,
            allow_episode_boundaries: bool = False,
            max_train_truncations: int | None = None,
            generator: torch.Generator | None = None,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if segment_length <= 0:
            raise ValueError(f"segment_length must be > 0, got {segment_length}")
        if burn_in_steps < 0:
            raise ValueError(f"burn_in_steps must be >= 0, got {burn_in_steps}")
        if max_train_truncations is not None and max_train_truncations < 0:
            raise ValueError(
                "max_train_truncations must be >= 0 when set, got "
                f"{max_train_truncations}"
            )
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if require_initial_temporal_state and self._temporal_state_available is None:
            raise ValueError(
                "sample_episode_segments(require_initial_temporal_state=True) requires "
                "temporal_state_store_interval to be set on the replay buffer."
            )

        total_sequence_length = burn_in_steps + segment_length
        candidate_env_indices, candidate_logical_starts = self._replay_segment_candidates(
            total_sequence_length=total_sequence_length,
            burn_in_steps=burn_in_steps,
            require_initial_temporal_state=require_initial_temporal_state,
            allow_episode_boundaries=allow_episode_boundaries,
            max_train_truncations=max_train_truncations,
        )
        num_candidates = int(candidate_env_indices.numel())
        if num_candidates == 0:
            raise NoEpisodeSegmentCandidatesError(
                "Cannot sample replay segments: no contiguous replay windows satisfy the requested "
                "length, temporal-state-start, and episode-boundary constraints."
            )
        if not replacement and batch_size > num_candidates:
            raise ValueError(
                f"Cannot sample batch_size={batch_size} episode segments without replacement from "
                f"{num_candidates} candidates."
            )

        candidate_indices = self._sample_indices(
            population_size=num_candidates,
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )
        env_indices = candidate_env_indices[candidate_indices]
        logical_starts = candidate_logical_starts[candidate_indices]
        sequence_offsets = torch.arange(total_sequence_length, dtype=torch.long, device=self.storage_device)
        flat_indices = self._tensor_operations.episode_segment_indices(
            env_indices,
            logical_starts,
            self._size_per_env_tensor,
            sequence_offsets,
        )
        flat_batch = self._fetch_indices(flat_indices)

        initial_temporal_state = None
        if require_initial_temporal_state:
            start_transition_slots = self._logical_to_transition_slots(logical_starts)
            start_obs_slots = self._transition_obs_slots[env_indices, start_transition_slots]
            initial_temporal_state = self._to_train_temporal_state(
                self._temporal_state_rows(env_indices=env_indices, obs_slots=start_obs_slots)
            )

        train_mask = torch.ones(
            (batch_size, total_sequence_length),
            dtype=torch.bool,
            device=self.storage_device,
        )
        if burn_in_steps > 0:
            train_mask[:, :burn_in_steps] = False
        return self._reshape_episode_segment_batch(
            flat_batch=flat_batch,
            batch_size=batch_size,
            total_sequence_length=total_sequence_length,
            burn_in_steps=burn_in_steps,
            train_mask=self._to_train(train_mask, dtype=torch.bool),
            initial_temporal_state=initial_temporal_state,
        )

    def _sample_transition_indices(
            self,
            *,
            batch_size: int,
            replacement: bool,
            generator: torch.Generator | None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if not replacement and batch_size > len(self):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} without replacement from {len(self)} transitions."
            )
        return self._sample_indices(
            population_size=len(self),
            batch_size=batch_size,
            replacement=replacement,
            generator=generator,
        )

    def _sample_indices(
            self,
            *,
            population_size: int,
            batch_size: int,
            replacement: bool,
            generator: torch.Generator | None,
    ) -> torch.Tensor:
        sampling_device = self.storage_device if generator is None else torch.device(generator.device)
        if replacement:
            indices = torch.randint(
                population_size,
                (batch_size,),
                generator=generator,
                device=sampling_device,
            )
        else:
            indices = torch.randperm(
                population_size,
                generator=generator,
                device=sampling_device,
            )[:batch_size]
        return indices.to(device=self.storage_device)

    def get_all(self) -> OffPolicyReplayBatch:
        if len(self) == 0:
            raise ValueError("Cannot fetch from an empty replay buffer.")
        return self._fetch_indices(torch.arange(len(self), device=self.storage_device))

    def _copy_obs_rows_at_slots_(
            self,
            *,
            obs: dict[str, torch.Tensor],
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor | None = None,
            source_env_indices: torch.Tensor | None = None,
    ) -> None:
        if target_env_indices is None:
            target_env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.storage_device)
        if source_env_indices is None:
            source_env_indices = target_env_indices
        obs_slots = obs_slots.to(device=self.storage_device, dtype=torch.long)
        target_env_indices = target_env_indices.to(device=self.storage_device, dtype=torch.long)
        source_env_indices = source_env_indices.to(dtype=torch.long)

        local_obs = self._select_obs_rows_for_storage(obs["local_obs"], source_env_indices, dtype=self.storage_dtype)
        global_obs = self._select_obs_rows_for_storage(obs["global_obs"], source_env_indices, dtype=self.storage_dtype)
        hidden_local_vars = self._select_obs_rows_for_storage(
            obs["hidden_local_vars"],
            source_env_indices,
            dtype=self.storage_dtype,
        )
        hidden_global_vars = self._select_obs_rows_for_storage(
            obs["hidden_global_vars"],
            source_env_indices,
            dtype=self.storage_dtype,
        )

        self._clear_temporal_state_available_(obs_slots=obs_slots, target_env_indices=target_env_indices)
        self.local_obs[target_env_indices, obs_slots] = local_obs
        self.global_obs[target_env_indices, obs_slots] = global_obs
        self.hidden_local_vars[target_env_indices, obs_slots] = hidden_local_vars
        self.hidden_global_vars[target_env_indices, obs_slots] = hidden_global_vars
        if self.scenario_ids is not None:
            scenario_ids = obs.get("scenario_id", None)
            if scenario_ids is None:
                raise ValueError("obs must include scenario_id because this replay buffer stores scenario IDs.")
            self.scenario_ids[target_env_indices, obs_slots] = self._select_obs_rows_for_storage(
                scenario_ids,
                source_env_indices,
                dtype=torch.long,
            )
        if self.agent_mask is not None:
            agent_mask = obs.get("agent_mask", None)
            if agent_mask is None:
                raise ValueError("obs must include agent_mask because this replay buffer stores agent_mask.")
            self.agent_mask[target_env_indices, obs_slots] = self._select_obs_rows_for_storage(
                agent_mask,
                source_env_indices,
                dtype=torch.bool,
            )

    def _clear_terminal_obs_for_slot_(self, slot: int) -> None:
        terminal_slots = self._terminal_obs_indices[:, slot]
        occupied_terminal_slots = terminal_slots[terminal_slots >= 0]
        self._terminal_obs_slots_in_use[occupied_terminal_slots] = False
        self._terminal_obs_indices[:, slot] = -1

    def _store_terminal_obs_at_slot_(
            self,
            *,
            slot: int,
            terminal_obs: dict[str, torch.Tensor],
            done_env_indices: torch.Tensor,
    ) -> None:
        source_env_indices = self._terminal_obs_source_indices(
            terminal_obs=terminal_obs,
            done_env_indices=done_env_indices,
        )
        terminal_rows = {
            "local_obs": self._select_obs_rows_for_storage(
                terminal_obs["local_obs"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "global_obs": self._select_obs_rows_for_storage(
                terminal_obs["global_obs"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "hidden_local_vars": self._select_obs_rows_for_storage(
                terminal_obs["hidden_local_vars"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
            "hidden_global_vars": self._select_obs_rows_for_storage(
                terminal_obs["hidden_global_vars"],
                source_env_indices,
                dtype=self.storage_dtype,
            ),
        }
        if self.scenario_ids is not None:
            scenario_ids = terminal_obs.get("scenario_id", None)
            if scenario_ids is None:
                raise ValueError(
                    "terminal_obs must include scenario_id because this replay buffer stores scenario IDs."
                )
            terminal_rows["scenario_id"] = self._select_obs_rows_for_storage(
                scenario_ids,
                source_env_indices,
                dtype=torch.long,
            )
        if self.agent_mask is not None:
            agent_mask = terminal_obs.get("agent_mask", None)
            if agent_mask is None:
                raise ValueError("terminal_obs must include agent_mask because this replay buffer stores agent_mask.")
            terminal_rows["agent_mask"] = self._select_obs_rows_for_storage(
                agent_mask,
                source_env_indices,
                dtype=torch.bool,
            )

        terminal_slots = self._allocate_terminal_obs_slots(int(done_env_indices.numel()))
        self._terminal_obs_indices[done_env_indices, slot] = terminal_slots
        self._terminal_local_obs[terminal_slots] = terminal_rows["local_obs"]
        self._terminal_global_obs[terminal_slots] = terminal_rows["global_obs"]
        self._terminal_hidden_local_vars[terminal_slots] = terminal_rows["hidden_local_vars"]
        self._terminal_hidden_global_vars[terminal_slots] = terminal_rows["hidden_global_vars"]
        if self._terminal_scenario_ids is not None:
            self._terminal_scenario_ids[terminal_slots] = terminal_rows["scenario_id"]
        if self._terminal_agent_mask is not None:
            self._terminal_agent_mask[terminal_slots] = terminal_rows["agent_mask"]

    def _allocate_terminal_obs_slots(self, count: int) -> torch.Tensor:
        free_slots = torch.nonzero(~self._terminal_obs_slots_in_use, as_tuple=False).flatten()
        while free_slots.numel() < count:
            self._grow_terminal_obs_storage()
            free_slots = torch.nonzero(~self._terminal_obs_slots_in_use, as_tuple=False).flatten()
        allocated_slots = free_slots[:count]
        self._terminal_obs_slots_in_use[allocated_slots] = True
        return allocated_slots

    def _grow_terminal_obs_storage(self) -> None:
        current_capacity = int(self._terminal_obs_slots_in_use.shape[0])
        new_capacity = min(self.total_capacity, current_capacity * 2)
        if new_capacity <= current_capacity:
            raise RuntimeError("Terminal observation storage exhausted.")

        def grow(tensor: torch.Tensor) -> torch.Tensor:
            grown = self._new_storage_tensor(
                (new_capacity, *tensor.shape[1:]),
                dtype=tensor.dtype,
            )
            grown[:current_capacity].copy_(tensor)
            return grown

        self._terminal_obs_slots_in_use = grow(self._terminal_obs_slots_in_use)
        self._terminal_local_obs = grow(self._terminal_local_obs)
        self._terminal_global_obs = grow(self._terminal_global_obs)
        self._terminal_hidden_local_vars = grow(self._terminal_hidden_local_vars)
        self._terminal_hidden_global_vars = grow(self._terminal_hidden_global_vars)
        if self._terminal_scenario_ids is not None:
            self._terminal_scenario_ids = grow(self._terminal_scenario_ids)
        if self._terminal_agent_mask is not None:
            self._terminal_agent_mask = grow(self._terminal_agent_mask)

    def _terminal_obs_source_indices(
            self,
            *,
            terminal_obs: dict[str, torch.Tensor],
            done_env_indices: torch.Tensor,
    ) -> torch.Tensor:
        terminal_batch_size = int(terminal_obs["local_obs"].shape[0])
        done_count = int(done_env_indices.numel())
        if terminal_batch_size == self.n_envs:
            return done_env_indices
        if terminal_batch_size == done_count:
            return torch.arange(done_count, dtype=torch.long, device=self.storage_device)
        raise ValueError(
            "terminal_obs must be either full vector-env observations or rows packed in done-env order; "
            f"got leading dimension {terminal_batch_size} for {done_count} done envs across {self.n_envs} envs."
        )

    def _select_obs_rows_for_storage(
            self,
            tensor: torch.Tensor,
            source_env_indices: torch.Tensor,
            *,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        source_env_indices = source_env_indices.to(device=tensor.device, dtype=torch.long)
        return tensor[source_env_indices].to(device=self.storage_device, dtype=dtype)

    def _fetch_indices(self, indices: torch.Tensor) -> OffPolicyReplayBatch:
        (
            _env_indices,
            _transition_slots,
            local_obs,
            global_obs,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask,
            actions,
            rewards,
            terminations,
            truncations,
            previous_actions,
            next_local_obs,
            next_global_obs,
            next_hidden_local_vars,
            next_hidden_global_vars,
            next_agent_mask,
            episode_start_mask,
            scenario_ids,
            next_scenario_ids,
        ) = self._tensor_operations.gather_replay_storage(
            indices,
            self._size_per_env_tensor,
            self._logical_transition_slot_offset,
            self.capacity_per_env,
            self._transition_obs_slots,
            self._transition_next_obs_slots,
            self.local_obs,
            self.global_obs,
            self.hidden_local_vars,
            self.hidden_global_vars,
            self.scenario_ids,
            self.agent_mask,
            self.actions,
            self.rewards,
            self.terminations,
            self.truncations,
            self.previous_actions,
            self.episode_starts,
            self._terminal_obs_indices,
            self._terminal_local_obs,
            self._terminal_global_obs,
            self._terminal_hidden_local_vars,
            self._terminal_hidden_global_vars,
            self._terminal_scenario_ids,
            self._terminal_agent_mask,
        )

        return OffPolicyReplayBatch(
            local_obs=self._to_train(local_obs),
            global_obs=self._to_train(global_obs),
            hidden_local_vars=self._to_train(hidden_local_vars),
            hidden_global_vars=self._to_train(hidden_global_vars),
            agent_mask=None if agent_mask is None else self._to_train(
                agent_mask,
                dtype=torch.bool,
            ),
            actions=self._to_train(actions),
            rewards=self._to_train(rewards),
            terminations=self._to_train(terminations, dtype=torch.bool),
            truncations=self._to_train(truncations, dtype=torch.bool),
            previous_actions=None if previous_actions is None else self._to_train(previous_actions),
            next_local_obs=self._to_train(next_local_obs),
            next_global_obs=self._to_train(next_global_obs),
            next_hidden_local_vars=self._to_train(next_hidden_local_vars),
            next_hidden_global_vars=self._to_train(next_hidden_global_vars),
            next_agent_mask=None if next_agent_mask is None else self._to_train(next_agent_mask, dtype=torch.bool),
            episode_start_mask=None if episode_start_mask is None else self._to_train(
                episode_start_mask,
                dtype=torch.bool,
            ),
            scenario_ids=None if scenario_ids is None else self._to_train(scenario_ids, dtype=torch.long),
            next_scenario_ids=(
                None
                if next_scenario_ids is None
                else self._to_train(next_scenario_ids, dtype=torch.long)
            ),
        )

    def _fetch_episode_windows(
            self,
            indices: torch.Tensor,
            *,
            num_next_steps: int,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        batch_size = int(indices.numel())
        sequence_offsets = torch.arange(num_next_steps, dtype=torch.long, device=self.storage_device)
        flat_indices, within_replay = self._tensor_operations.episode_window_indices(
            indices,
            self._size_per_env_tensor,
            sequence_offsets,
        )
        flat_batch = self._fetch_indices(flat_indices)
        within_replay = self._to_train(within_replay, dtype=torch.bool)
        window_batch = self._reshape_episode_segment_batch(
            flat_batch=flat_batch,
            batch_size=batch_size,
            total_sequence_length=num_next_steps,
            burn_in_steps=0,
            train_mask=within_replay,
            initial_temporal_state=None,
        )
        previous_episode_end = torch.cat((
            torch.zeros((batch_size, 1), dtype=torch.bool, device=window_batch.actions.device),
            window_batch.episode_ends[:, :-1],
        ), dim=1).to(dtype=torch.long).cumsum(dim=1) > 0
        valid_steps = within_replay & ~previous_episode_end
        return self._mask_episode_window_batch(window_batch, valid_steps=valid_steps)

    @staticmethod
    def _mask_episode_window_batch(
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            valid_steps: torch.Tensor,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        def mask_tensor(tensor: torch.Tensor, value: bool | float = 0.0) -> torch.Tensor:
            invalid = ~valid_steps.reshape((*valid_steps.shape, *((1,) * (tensor.ndim - 2))))
            return tensor.masked_fill(invalid, value)

        def mask_optional_tensor(
                tensor: torch.Tensor | None,
                value: bool | float = 0.0,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None
            return mask_tensor(tensor, value)

        return OffPolicyReplayEpisodeSegmentBatch(
            local_obs=mask_tensor(batch.local_obs),
            global_obs=mask_tensor(batch.global_obs),
            hidden_local_vars=mask_tensor(batch.hidden_local_vars),
            hidden_global_vars=mask_tensor(batch.hidden_global_vars),
            agent_mask=mask_optional_tensor(batch.agent_mask, True),
            actions=mask_tensor(batch.actions),
            rewards=mask_tensor(batch.rewards),
            terminations=mask_tensor(batch.terminations, False),
            truncations=mask_tensor(batch.truncations, False),
            previous_actions=mask_optional_tensor(batch.previous_actions),
            next_local_obs=mask_tensor(batch.next_local_obs),
            next_global_obs=mask_tensor(batch.next_global_obs),
            next_hidden_local_vars=mask_tensor(batch.next_hidden_local_vars),
            next_hidden_global_vars=mask_tensor(batch.next_hidden_global_vars),
            next_agent_mask=mask_optional_tensor(batch.next_agent_mask, True),
            episode_start_mask=mask_optional_tensor(batch.episode_start_mask, False),
            train_mask=valid_steps,
            initial_temporal_state=None,
            burn_in_steps=0,
            scenario_ids=mask_optional_tensor(batch.scenario_ids),
            next_scenario_ids=mask_optional_tensor(batch.next_scenario_ids),
        )

    def _replay_segment_candidates(
            self,
            *,
            total_sequence_length: int,
            burn_in_steps: int = 0,
            require_initial_temporal_state: bool,
            allow_episode_boundaries: bool,
            max_train_truncations: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (
            total_sequence_length,
            burn_in_steps,
            require_initial_temporal_state,
            allow_episode_boundaries,
            max_train_truncations,
        )
        cached_candidates = self._segment_candidate_cache.get(cache_key)
        if cached_candidates is not None:
            return cached_candidates
        if self._size_per_env < total_sequence_length:
            empty = torch.empty((0,), dtype=torch.long, device=self.storage_device)
            return empty, empty
        if require_initial_temporal_state and self.temporal_states is None:
            empty = torch.empty((0,), dtype=torch.long, device=self.storage_device)
            return empty, empty

        logical_positions = torch.arange(self._size_per_env, dtype=torch.long, device=self.storage_device)
        temporal_state_available = None
        if require_initial_temporal_state:
            assert self._temporal_state_available is not None
            temporal_state_available = self._temporal_state_available
        valid = self._tensor_operations.episode_segment_candidate_mask(
            logical_positions,
            self._logical_transition_slot_offset,
            self.capacity_per_env,
            total_sequence_length,
            burn_in_steps,
            self.terminations,
            self.truncations,
            self._transition_obs_slots,
            temporal_state_available,
            allow_episode_boundaries,
            max_train_truncations,
        )

        candidates = torch.nonzero(valid, as_tuple=True)
        self._segment_candidate_cache[cache_key] = candidates
        return candidates

    def _reshape_episode_segment_batch(
            self,
            *,
            flat_batch: OffPolicyReplayBatch,
            batch_size: int,
            total_sequence_length: int,
            burn_in_steps: int,
            train_mask: torch.Tensor,
            initial_temporal_state: Any,
    ) -> OffPolicyReplayEpisodeSegmentBatch:
        def reshape_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, total_sequence_length, *tensor.shape[1:])

        def reshape_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return reshape_tensor(tensor)

        return OffPolicyReplayEpisodeSegmentBatch(
            local_obs=reshape_tensor(flat_batch.local_obs),
            global_obs=reshape_tensor(flat_batch.global_obs),
            hidden_local_vars=reshape_tensor(flat_batch.hidden_local_vars),
            hidden_global_vars=reshape_tensor(flat_batch.hidden_global_vars),
            agent_mask=reshape_optional_tensor(flat_batch.agent_mask),
            actions=reshape_tensor(flat_batch.actions),
            rewards=reshape_tensor(flat_batch.rewards),
            terminations=reshape_tensor(flat_batch.terminations),
            truncations=reshape_tensor(flat_batch.truncations),
            previous_actions=reshape_optional_tensor(flat_batch.previous_actions),
            next_local_obs=reshape_tensor(flat_batch.next_local_obs),
            next_global_obs=reshape_tensor(flat_batch.next_global_obs),
            next_hidden_local_vars=reshape_tensor(flat_batch.next_hidden_local_vars),
            next_hidden_global_vars=reshape_tensor(flat_batch.next_hidden_global_vars),
            next_agent_mask=reshape_optional_tensor(flat_batch.next_agent_mask),
            episode_start_mask=reshape_optional_tensor(flat_batch.episode_start_mask),
            train_mask=train_mask,
            initial_temporal_state=initial_temporal_state,
            burn_in_steps=burn_in_steps,
            scenario_ids=reshape_optional_tensor(flat_batch.scenario_ids),
            next_scenario_ids=reshape_optional_tensor(flat_batch.next_scenario_ids),
        )

    def _new_storage_tensor(self, shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        if self.storage_pin_memory:
            return torch.empty(shape, dtype=dtype, device=self.storage_device, pin_memory=True).zero_()
        return torch.zeros(shape, dtype=dtype, device=self.storage_device)

    def _to_train(self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        target_dtype = dtype
        if target_dtype is None and tensor.is_floating_point():
            target_dtype = self.train_dtype
        if (
                self.storage_pin_memory
                and self.train_device.type == "cuda"
                and tensor.device.type == "cpu"
                and not tensor.is_pinned()
        ):
            tensor = tensor.pin_memory()
        return tensor.to(
            device=self.train_device,
            dtype=target_dtype if target_dtype is not None else tensor.dtype,
            non_blocking=self.non_blocking_train_transfer,
        )

    def _to_train_temporal_state(self, state: Any) -> Any:
        if state is None:
            return None
        if torch.is_tensor(state):
            return self._to_train(state)
        if isinstance(state, tuple):
            return tuple(self._to_train_temporal_state(item) for item in state)
        if isinstance(state, list):
            return [self._to_train_temporal_state(item) for item in state]
        if isinstance(state, Mapping):
            return type(state)((key, self._to_train_temporal_state(value)) for key, value in state.items())
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _clear_temporal_state_available_(
            self,
            *,
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor,
    ) -> None:
        if self._temporal_state_available is None:
            return
        assert self._temporal_state_indices is not None
        assert self._temporal_state_slots_in_use is not None
        self._tensor_operations.release_temporal_state_slots(
            self._temporal_state_available,
            self._temporal_state_indices,
            self._temporal_state_slots_in_use,
            target_env_indices,
            obs_slots,
        )

    def _should_store_temporal_state(self, observation_step_idx: int) -> bool:
        return (
            self.temporal_state_store_interval is not None
            and observation_step_idx % self.temporal_state_store_interval == 0
        )

    def _copy_temporal_state_rows_at_slots_(
            self,
            *,
            state: Any,
            obs_slots: torch.Tensor,
            target_env_indices: torch.Tensor | None = None,
            source_env_indices: torch.Tensor | None = None,
    ) -> None:
        if self._temporal_state_available is None:
            return
        assert self._temporal_state_indices is not None
        assert self._temporal_state_slots_in_use is not None
        if target_env_indices is None:
            target_env_indices = torch.arange(self.n_envs, dtype=torch.long, device=self.storage_device)
        if source_env_indices is None:
            source_env_indices = target_env_indices
        obs_slots = obs_slots.to(device=self.storage_device, dtype=torch.long)
        target_env_indices = target_env_indices.to(device=self.storage_device, dtype=torch.long)
        source_env_indices = source_env_indices.to(dtype=torch.long)

        checkpoint_slots = self._tensor_operations.allocate_temporal_state_slots(
            self._temporal_state_indices,
            self._temporal_state_slots_in_use,
            target_env_indices,
            obs_slots,
        )

        if self.temporal_states is None:
            self.temporal_states = self._new_temporal_state_storage(state)
        self._copy_temporal_state_tree_rows_(
            target=self.temporal_states,
            source=state,
            checkpoint_slots=checkpoint_slots,
            target_env_indices=target_env_indices,
            source_env_indices=source_env_indices,
        )
        self._temporal_state_available[target_env_indices, obs_slots] = True

    def _new_temporal_state_storage(self, state: Any) -> Any:
        if torch.is_tensor(state):
            if int(state.shape[0]) != self.n_envs:
                raise ValueError(f"Temporal state leading dimension must be {self.n_envs}, got {state.shape[0]}")
            dtype = self.temporal_state_storage_dtype if state.is_floating_point() else state.dtype
            return self._new_storage_tensor(
                (self.n_envs, self.temporal_state_capacity_per_env, *state.shape[1:]),
                dtype=dtype,
            )
        if isinstance(state, tuple):
            return tuple(self._new_temporal_state_storage(item) for item in state)
        if isinstance(state, list):
            return [self._new_temporal_state_storage(item) for item in state]
        if isinstance(state, Mapping):
            return type(state)((key, self._new_temporal_state_storage(value)) for key, value in state.items())
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _copy_temporal_state_tree_rows_(
            self,
            *,
            target: Any,
            source: Any,
            checkpoint_slots: torch.Tensor,
            target_env_indices: torch.Tensor,
            source_env_indices: torch.Tensor,
    ) -> None:
        if torch.is_tensor(target) and torch.is_tensor(source):
            source_indices = source_env_indices.to(device=source.device, dtype=torch.long)
            source_rows = source[source_indices].to(device=self.storage_device, dtype=target.dtype)
            target[target_env_indices, checkpoint_slots] = source_rows
            return
        if isinstance(target, tuple) and isinstance(source, tuple):
            for target_item, source_item in zip(target, source, strict=True):
                self._copy_temporal_state_tree_rows_(
                    target=target_item,
                    source=source_item,
                    checkpoint_slots=checkpoint_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        if isinstance(target, list) and isinstance(source, list):
            for target_item, source_item in zip(target, source, strict=True):
                self._copy_temporal_state_tree_rows_(
                    target=target_item,
                    source=source_item,
                    checkpoint_slots=checkpoint_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        if isinstance(target, Mapping) and isinstance(source, Mapping):
            if target.keys() != source.keys():
                raise ValueError("Temporal state mappings must have matching keys")
            for key in target:
                self._copy_temporal_state_tree_rows_(
                    target=target[key],
                    source=source[key],
                    checkpoint_slots=checkpoint_slots,
                    target_env_indices=target_env_indices,
                    source_env_indices=source_env_indices,
                )
            return
        raise TypeError(
            f"Temporal state structures do not match: {type(target).__name__} and {type(source).__name__}"
        )

    def _temporal_state_rows(
            self,
            *,
            env_indices: torch.Tensor,
            obs_slots: torch.Tensor,
    ) -> Any:
        assert self._temporal_state_indices is not None
        checkpoint_slots = self._temporal_state_indices[env_indices, obs_slots]
        if torch.any(checkpoint_slots < 0):
            raise RuntimeError("Requested observation slot has no temporal-state checkpoint.")
        return self._temporal_state_tree_rows(
            state=self.temporal_states,
            env_indices=env_indices,
            checkpoint_slots=checkpoint_slots,
        )

    def _temporal_state_tree_rows(
            self,
            *,
            state: Any,
            env_indices: torch.Tensor,
            checkpoint_slots: torch.Tensor,
    ) -> Any:
        if torch.is_tensor(state):
            return state[env_indices, checkpoint_slots]
        if isinstance(state, tuple):
            return tuple(
                self._temporal_state_tree_rows(
                    state=item,
                    env_indices=env_indices,
                    checkpoint_slots=checkpoint_slots,
                )
                for item in state
            )
        if isinstance(state, list):
            return [
                self._temporal_state_tree_rows(
                    state=item,
                    env_indices=env_indices,
                    checkpoint_slots=checkpoint_slots,
                )
                for item in state
            ]
        if isinstance(state, Mapping):
            return type(state)(
                (
                    key,
                    self._temporal_state_tree_rows(
                        state=value,
                        env_indices=env_indices,
                        checkpoint_slots=checkpoint_slots,
                    ),
                )
                for key, value in state.items()
            )
        raise TypeError(f"Unsupported temporal state item: {type(state).__name__}")

    def _advance_obs_slots(self, obs_slots: torch.Tensor) -> torch.Tensor:
        return self._tensor_operations.advance_obs_slots(
            obs_slots,
            self.observation_capacity_per_env,
        )

    def _logical_to_transition_slots(self, logical_transition_slots: torch.Tensor) -> torch.Tensor:
        return self._tensor_operations.logical_to_transition_slots(
            logical_transition_slots,
            self._logical_transition_slot_offset,
            self.capacity_per_env,
        )
