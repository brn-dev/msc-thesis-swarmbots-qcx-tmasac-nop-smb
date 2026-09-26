from dataclasses import dataclass
import math
from typing import Any

import torch

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch, PPORolloutBatchBuilder
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEIntervalResetMode,
    GSDEProbabilityResetMode,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.rollout_utils import append_episode_infos, extract_bootstrap_obs, initial_previous_actions, snapshot_obs
from swarmbots.learn.summary_statistics import compute_summary_statistics


@dataclass(slots=True)
class PPORolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    temporal_state: Any
    rollout_step_idx: int


@dataclass(slots=True)
class _RolloutTimers:
    reset_noise_timings: list[float]
    policy_forward_timings: list[float]
    env_step_timings: list[float]
    buffer_add_timings: list[float]
    reset_noise_timer: PerformanceTimer
    policy_forward_timer: PerformanceTimer
    env_step_timer: PerformanceTimer
    buffer_add_timer: PerformanceTimer


def _parse_gsde_reset_mode(
        policy: BasePPOPolicy[Any, Any],
        gsde_reset_mode: GSDEResetMode | None,
) -> tuple[bool, bool, int, float]:
    gsde_enabled = policy.gsde_enabled
    active_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=gsde_enabled,
        reset_mode=gsde_reset_mode,
    )
    is_gsde_interval_reset_mode = False
    gsde_reset_interval = -1
    gsde_reset_prob = -1.0
    if active_reset_mode is not None:
        if isinstance(active_reset_mode, GSDEIntervalResetMode):
            is_gsde_interval_reset_mode = True
            gsde_reset_interval = active_reset_mode.interval
        elif isinstance(active_reset_mode, GSDEProbabilityResetMode):
            gsde_reset_prob = active_reset_mode.probability
    return gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob


def _init_rollout_timers() -> _RolloutTimers:
    return _RolloutTimers(
        reset_noise_timings=[],
        policy_forward_timings=[],
        env_step_timings=[],
        buffer_add_timings=[],
        reset_noise_timer=PerformanceTimer(),
        policy_forward_timer=PerformanceTimer(),
        env_step_timer=PerformanceTimer(),
        buffer_add_timer=PerformanceTimer(),
    )


def _build_gsde_step_reset_mask(
        *,
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
        rollout_step_idx: int,
        batch_shape: tuple[int, ...],
        rollout_device: torch.device,
) -> torch.Tensor | None:
    if not gsde_enabled:
        return None
    if is_gsde_interval_reset_mode:
        if (rollout_step_idx % gsde_reset_interval) != 0:
            return None
        return torch.ones(batch_shape, device=rollout_device, dtype=torch.bool)
    return torch.empty(
        batch_shape,
        device=rollout_device,
        dtype=torch.bool,
    ).bernoulli_(gsde_reset_prob)


def _reset_temporal_correlations(
        *,
        policy: BasePPOPolicy[Any, Any],
        episode_start_mask: torch.Tensor | None = None,
        step_reset_mask: torch.Tensor | None = None,
        batch_shape: tuple[int, ...] | None = None,
) -> None:
    if episode_start_mask is not None:
        policy.action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
    if step_reset_mask is not None or batch_shape is not None:
        policy.action_dist.reset_temporal_correlations_on_step(mask=step_reset_mask, batch_shape=batch_shape)


def _evaluate_values(
        *,
        policy: BasePPOPolicy[Any, Any],
        obs: dict[str, torch.Tensor],
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        episode_start_mask: torch.Tensor | None = None,
        terminated_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values, _ = policy.predict_values_with_temporal_state(
        local_obs=obs["local_obs"],
        global_obs=obs["global_obs"],
        hidden_local_vars=obs["hidden_local_vars"],
        hidden_global_vars=obs["hidden_global_vars"],
        agent_mask=obs.get("agent_mask", None),
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        episode_start_mask=episode_start_mask,
    )

    if terminated_mask is None:
        return values
    return values.masked_fill(terminated_mask, 0.0)


def _reset_rollout_state(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        n_agent_actions: int,
) -> PPORolloutState:
    obs, _info = env.reset()
    local_obs = obs["local_obs"]
    episode_start_mask = torch.ones((local_obs.shape[0],), dtype=torch.bool, device=local_obs.device)
    previous_actions = initial_previous_actions(
        policy=policy,
        obs=obs,
        n_agent_actions=n_agent_actions,
    )
    temporal_state = policy.initial_temporal_state(
        batch_size=local_obs.shape[0],
        n_agents=local_obs.shape[1],
        device=local_obs.device,
        dtype=local_obs.dtype,
    )
    return PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=0,
    )


def _collect_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        obs: dict[str, torch.Tensor],
        episode_start_mask: torch.Tensor,
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        rollout_step_idx: int,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, Any, int]:
    obs_snapshot = snapshot_obs(obs)
    local_obs = obs_snapshot["local_obs"]
    batch_shape = tuple(local_obs.shape[:-1])
    step_reset_mask = _build_gsde_step_reset_mask(
        gsde_enabled=gsde_enabled,
        is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
        gsde_reset_interval=gsde_reset_interval,
        gsde_reset_prob=gsde_reset_prob,
        rollout_step_idx=rollout_step_idx,
        batch_shape=batch_shape,
        rollout_device=buffer.rollout_device,
    )

    with timers.reset_noise_timer:
        _reset_temporal_correlations(
            policy=policy,
            episode_start_mask=episode_start_mask,
            step_reset_mask=step_reset_mask,
        )
    timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

    with timers.policy_forward_timer:
        actions, log_probs, values, next_temporal_state = policy.forward_with_temporal_state(
            local_obs=obs_snapshot["local_obs"],
            global_obs=obs_snapshot["global_obs"],
            hidden_local_vars=obs_snapshot["hidden_local_vars"],
            hidden_global_vars=obs_snapshot["hidden_global_vars"],
            agent_mask=obs_snapshot.get("agent_mask", None),
            previous_actions=previous_actions,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())

    with timers.env_step_timer:
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
    timers.env_step_timings.append(timers.env_step_timer.get_duration())

    dones = torch.logical_or(terminations, truncations)
    append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)

    bootstrap_obs = extract_bootstrap_obs(
        env=env,
        next_obs=next_obs,
        infos=infos,
        dones=dones,
    )
    bootstrap_previous_actions = None if previous_actions is None else actions.detach()
    bootstrap_values = _evaluate_values(
        policy=policy,
        obs=bootstrap_obs,
        previous_actions=bootstrap_previous_actions,
        temporal_state=next_temporal_state,
        terminated_mask=terminations,
    )

    _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

    with timers.buffer_add_timer:
        buffer.add(
            local_obs=obs_snapshot["local_obs"],
            global_obs=obs_snapshot["global_obs"],
            hidden_local_vars=obs_snapshot["hidden_local_vars"],
            hidden_global_vars=obs_snapshot["hidden_global_vars"],
            agent_mask=obs_snapshot.get("agent_mask", None),
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            previous_actions=previous_actions,
            episode_start_mask=episode_start_mask,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            bootstrap_obs=bootstrap_obs,
            bootstrap_values=bootstrap_values,
            dones=dones,
        )
    timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return next_obs, dones, next_previous_actions, next_temporal_state, rollout_step_idx + 1


def _warmup_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        obs: dict[str, torch.Tensor],
        episode_start_mask: torch.Tensor,
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        rollout_step_idx: int,
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, Any, int]:
    local_obs = obs["local_obs"]
    batch_shape = tuple(local_obs.shape[:-1])
    step_reset_mask = _build_gsde_step_reset_mask(
        gsde_enabled=gsde_enabled,
        is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
        gsde_reset_interval=gsde_reset_interval,
        gsde_reset_prob=gsde_reset_prob,
        rollout_step_idx=rollout_step_idx,
        batch_shape=batch_shape,
        rollout_device=local_obs.device,
    )

    _reset_temporal_correlations(
        policy=policy,
        episode_start_mask=episode_start_mask,
        step_reset_mask=step_reset_mask,
    )

    actions, _log_probs, _values, next_temporal_state = policy.forward_with_temporal_state(
        local_obs=local_obs,
        global_obs=obs["global_obs"],
        hidden_local_vars=obs["hidden_local_vars"],
        hidden_global_vars=obs["hidden_global_vars"],
        agent_mask=obs.get("agent_mask", None),
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        episode_start_mask=episode_start_mask,
    )

    next_obs, _rewards, terminations, truncations, _infos = env.step(actions)
    dones = torch.logical_or(terminations, truncations)

    _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return next_obs, dones, next_previous_actions, next_temporal_state, rollout_step_idx + 1


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        buffer_get_whole_episodes_time: float,
        episode_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "env_reset_time": env_reset_time,
        "to_rollout_device_time": to_rollout_device_time,
        "reset_noise_time": compute_summary_statistics(timers.reset_noise_timings),
        "total_reset_noise_time": sum(timers.reset_noise_timings),
        "policy_forward_time": compute_summary_statistics(timers.policy_forward_timings),
        "total_policy_forward_time": sum(timers.policy_forward_timings),
        "env_step_time": compute_summary_statistics(timers.env_step_timings),
        "total_env_step_time": sum(timers.env_step_timings),
        "buffer_add_time": compute_summary_statistics(timers.buffer_add_timings),
        "total_buffer_add_time": sum(timers.buffer_add_timings),
        "buffer_get_whole_episodes_time": buffer_get_whole_episodes_time,
    }
    success_values = [float(ep_info["success"]) for ep_info in episode_infos if "success" in ep_info]
    if success_values:
        metrics["ep_success_rate"] = 100.0 * (sum(success_values) / len(success_values))
    return metrics


@torch.no_grad()
def warmup_rollout_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> PPORolloutState:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    if rollout_state is None:
        rollout_state = _reset_rollout_state(
            env=env,
            policy=policy,
            n_agent_actions=env.action_space.total_agent_action_dim,
        )

    obs = rollout_state.obs
    episode_start_mask = rollout_state.episode_start_mask
    previous_actions = rollout_state.previous_actions
    temporal_state = rollout_state.temporal_state
    rollout_step_idx = rollout_state.rollout_step_idx

    policy.to(obs["local_obs"].device)
    policy.eval()

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += obs["local_obs"].shape[0]
        obs, episode_start_mask, previous_actions, temporal_state, rollout_step_idx = _warmup_rollout_step(
            env=env,
            policy=policy,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    return PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
    )


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_episodes: int,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisodeSegment], list[dict[str, Any]], dict[str, Any]]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()
    with PerformanceTimer() as env_reset_timer:
        obs, _info = env.reset()
    episode_start_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    previous_actions = initial_previous_actions(
        policy=policy,
        obs=obs,
        n_agent_actions=buffer.n_agent_actions,
    )
    temporal_state = policy.initial_temporal_state(
        batch_size=buffer.n_envs,
        n_agents=buffer.n_agents,
        device=buffer.rollout_device,
        dtype=buffer.rollout_dtype,
    )

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    episode_infos: list[dict[str, Any]] = []
    rollout_step_idx = 0
    timers = _init_rollout_timers()

    while len(buffer.episodes) < n_episodes:
        obs, episode_start_mask, previous_actions, temporal_state, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        episodes = buffer.get_whole_episodes()

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_timer.get_duration(),
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        buffer_get_whole_episodes_time=buffer_get_whole_episodes_timer.get_duration(),
        episode_infos=episode_infos,
    )
    return episodes, episode_infos, metrics


@torch.no_grad()
def collect_step_rollout_batch(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[PPORolloutBatch, list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    buffer.reset()

    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            obs, _info = env.reset()
        env_reset_time = env_reset_timer.get_duration()
        episode_start_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        previous_actions = initial_previous_actions(
            policy=policy,
            obs=obs,
            n_agent_actions=buffer.n_agent_actions,
        )
        temporal_state = policy.initial_temporal_state(
            batch_size=buffer.n_envs,
            n_agents=buffer.n_agents,
            device=buffer.rollout_device,
            dtype=buffer.rollout_dtype,
        )
        rollout_step_idx = 0
    else:
        obs = rollout_state.obs
        episode_start_mask = rollout_state.episode_start_mask
        previous_actions = rollout_state.previous_actions
        temporal_state = rollout_state.temporal_state
        rollout_step_idx = rollout_state.rollout_step_idx

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    vector_steps = math.ceil(n_steps / buffer.n_envs)
    with PerformanceTimer() as batch_builder_init_timer:
        batch_builder = PPORolloutBatchBuilder(
            n_envs=buffer.n_envs,
            n_steps=vector_steps,
            n_agents=buffer.n_agents,
            agent_obs_shape=buffer.agent_obs_shape,
            global_obs_shape=buffer.global_obs_shape,
            hidden_local_vars_shape=buffer.hidden_local_vars_shape,
            hidden_global_vars_shape=buffer.hidden_global_vars_shape,
            n_agent_actions=buffer.n_agent_actions,
            has_agent_mask=buffer.has_agent_mask,
            has_previous_actions=policy.requires_previous_actions(),
            gamma=buffer.gamma,
            gae_lambda=buffer.gae_lambda,
            storage_device=buffer.rollout_device,
            storage_dtype=buffer.rollout_dtype,
        )

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    for batch_step_idx in range(vector_steps):
        obs_snapshot = snapshot_obs(obs)
        local_obs = obs_snapshot["local_obs"]
        batch_shape = tuple(local_obs.shape[:-1])
        step_reset_mask = _build_gsde_step_reset_mask(
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
            rollout_step_idx=rollout_step_idx,
            batch_shape=batch_shape,
            rollout_device=buffer.rollout_device,
        )

        with timers.reset_noise_timer:
            _reset_temporal_correlations(
                policy=policy,
                episode_start_mask=episode_start_mask,
                step_reset_mask=step_reset_mask,
            )
        timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

        with timers.policy_forward_timer:
            actions, log_probs, values, next_temporal_state = policy.forward_with_temporal_state(
                local_obs=obs_snapshot["local_obs"],
                global_obs=obs_snapshot["global_obs"],
                hidden_local_vars=obs_snapshot["hidden_local_vars"],
                hidden_global_vars=obs_snapshot["hidden_global_vars"],
                agent_mask=obs_snapshot.get("agent_mask", None),
                previous_actions=previous_actions,
                temporal_state=temporal_state,
                episode_start_mask=episode_start_mask,
            )
        timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())

        with timers.env_step_timer:
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
        timers.env_step_timings.append(timers.env_step_timer.get_duration())

        dones = torch.logical_or(terminations, truncations)
        append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)

        bootstrap_obs = extract_bootstrap_obs(
            env=env,
            next_obs=next_obs,
            infos=infos,
            dones=dones,
        )
        bootstrap_previous_actions = None if previous_actions is None else actions.detach()
        bootstrap_values = _evaluate_values(
            policy=policy,
            obs=bootstrap_obs,
            previous_actions=bootstrap_previous_actions,
            temporal_state=next_temporal_state,
            terminated_mask=terminations,
        )

        _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

        with timers.buffer_add_timer:
            batch_builder.add(
                step_idx=batch_step_idx,
                local_obs=obs_snapshot["local_obs"],
                global_obs=obs_snapshot["global_obs"],
                hidden_local_vars=obs_snapshot["hidden_local_vars"],
                hidden_global_vars=obs_snapshot["hidden_global_vars"],
                agent_mask=obs_snapshot.get("agent_mask", None),
                previous_actions=previous_actions,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                bootstrap_obs=bootstrap_obs,
                bootstrap_values=bootstrap_values,
                terminations=terminations,
                truncations=truncations,
                dones=dones,
                episode_start_mask=episode_start_mask,
            )
        timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

        next_previous_actions: torch.Tensor | None = None
        if previous_actions is not None:
            next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

        obs = next_obs
        episode_start_mask = dones
        previous_actions = next_previous_actions
        temporal_state = next_temporal_state
        rollout_step_idx += 1

    with PerformanceTimer() as batch_finalize_timer:
        rollout_batch = batch_builder.build().to(device=buffer.train_device, dtype=buffer.train_dtype)

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        buffer_get_whole_episodes_time=batch_finalize_timer.get_duration(),
        episode_infos=episode_infos,
    )
    metrics["step_rollout_batch_path"] = True
    metrics["rollout_batch_builder_init_time"] = batch_builder_init_timer.get_duration()
    metrics["rollout_batch_finalize_time"] = batch_finalize_timer.get_duration()
    new_state = PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
    )
    return rollout_batch, episode_infos, metrics, new_state


@torch.no_grad()
def collect_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisodeSegment], list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()

    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            obs, _info = env.reset()
        env_reset_time = env_reset_timer.get_duration()
        episode_start_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        previous_actions = initial_previous_actions(
            policy=policy,
            obs=obs,
            n_agent_actions=buffer.n_agent_actions,
        )
        temporal_state = policy.initial_temporal_state(
            batch_size=buffer.n_envs,
            n_agents=buffer.n_agents,
            device=buffer.rollout_device,
            dtype=buffer.rollout_dtype,
        )
        rollout_step_idx = 0
    else:
        obs = rollout_state.obs
        episode_start_mask = rollout_state.episode_start_mask
        previous_actions = rollout_state.previous_actions
        temporal_state = rollout_state.temporal_state
        rollout_step_idx = rollout_state.rollout_step_idx

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += buffer.n_envs
        obs, episode_start_mask, previous_actions, temporal_state, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    final_values = _evaluate_values(
        policy=policy,
        obs=obs,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        episode_start_mask=episode_start_mask,
    )

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        completed_episodes = buffer.get_whole_episodes()
    buffer_get_whole_episodes_time = buffer_get_whole_episodes_timer.get_duration()
    with PerformanceTimer() as buffer_dump_partial_episodes_timer:
        partial_episodes = buffer.dump_partial_episodes(
            final_obs=obs,
            final_values=final_values,
            clone_tensors=False,
        )
    buffer_get_whole_episodes_time += buffer_dump_partial_episodes_timer.get_duration()

    episodes = completed_episodes + partial_episodes

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        buffer_get_whole_episodes_time=buffer_get_whole_episodes_time,
        episode_infos=episode_infos,
    )
    new_state = PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
    )
    return episodes, episode_infos, metrics, new_state
