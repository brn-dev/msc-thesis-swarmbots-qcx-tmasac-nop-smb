from dataclasses import dataclass
from typing import Any

import torch

from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBuffer
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEIntervalResetMode,
    GSDEProbabilityResetMode,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.rollout_utils import (
    append_episode_infos,
    extract_terminal_obs,
    initial_previous_actions,
    sample_random_actions,
    snapshot_obs,
)
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.temporal_state import move_temporal_state
from swarmbots.learn.torch_device import as_device


@dataclass(slots=True)
class OffPolicyRolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    temporal_state: Any
    rollout_step_idx: int
    gsde_noise_state: Any = None


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


def _reset_rollout_state(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePolicy | None,
        track_previous_actions: bool,
        n_agent_actions: int,
) -> OffPolicyRolloutState:
    obs, _info = env.reset()
    local_obs = obs["local_obs"]
    episode_start_mask = torch.ones((local_obs.shape[0],), dtype=torch.bool, device=local_obs.device)
    previous_actions = None
    if track_previous_actions:
        previous_actions = initial_previous_actions(
            obs=obs,
            n_agent_actions=n_agent_actions,
        )
    temporal_state = None
    if policy is not None:
        temporal_state = policy.initial_temporal_state(
            batch_size=local_obs.shape[0],
            n_agents=local_obs.shape[1],
            device=local_obs.device,
            dtype=local_obs.dtype,
        )
    return OffPolicyRolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=0,
    )


def _move_obs_to_device(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in obs.items()}


def _move_rollout_state_to_device(
        *,
        rollout_state: OffPolicyRolloutState,
        device: torch.device,
) -> OffPolicyRolloutState:
    return OffPolicyRolloutState(
        obs=_move_obs_to_device(rollout_state.obs, device=device),
        episode_start_mask=rollout_state.episode_start_mask.to(device=device),
        previous_actions=(
            None
            if rollout_state.previous_actions is None
            else rollout_state.previous_actions.to(device=device)
        ),
        temporal_state=move_temporal_state(rollout_state.temporal_state, device=device),
        rollout_step_idx=rollout_state.rollout_step_idx,
        gsde_noise_state=move_temporal_state(rollout_state.gsde_noise_state, device=device),
    )


def _reset_policy_action_noise(
        *,
        policy: BasePolicy,
        local_obs: torch.Tensor,
        episode_start_mask: torch.Tensor,
        rollout_step_idx: int,
        gsde_reset_mode: GSDEResetMode,
        force_step_reset: bool,
) -> None:
    action_dist = getattr(policy, "action_dist")
    batch_shape = tuple(local_obs.shape[:-1])
    if force_step_reset:
        action_dist.reset_temporal_correlations_on_step(batch_shape=batch_shape)
    action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
    if isinstance(gsde_reset_mode, GSDEIntervalResetMode):
        if not force_step_reset and (rollout_step_idx % gsde_reset_mode.interval) == 0:
            action_dist.reset_temporal_correlations_on_step(batch_shape=batch_shape)
        return
    if isinstance(gsde_reset_mode, GSDEProbabilityResetMode):
        if force_step_reset:
            return
        mask = torch.empty(batch_shape, device=local_obs.device, dtype=torch.bool).bernoulli_(gsde_reset_mode.probability)
        action_dist.reset_temporal_correlations_on_step(mask=mask)
        return
    raise TypeError(f"Unknown gsde_reset_mode type: {type(gsde_reset_mode)}")


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
        transitions_collected: int,
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
        "transitions_collected": transitions_collected,
    }
    success_values = [float(ep_info["success"]) for ep_info in episode_infos if "success" in ep_info]
    if success_values:
        metrics["ep_success_rate"] = 100.0 * (sum(success_values) / len(success_values))
    return metrics


@torch.no_grad()
def collect_off_policy_steps(
        env: BaseLearnEnvWrapper,
        replay_buffer: OffPolicyReplayBuffer,
        n_steps: int,
        *,
        policy: BasePolicy | None = None,
        rollout_state: OffPolicyRolloutState | None = None,
        random_actions: bool = False,
        deterministic: bool = False,
        gsde_reset_mode: GSDEResetMode | None = None,
        rollout_device: torch.device | str | None = None,
        _store_transitions: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], OffPolicyRolloutState]:
    """
    Collect complete vector-env steps into replay.

    n_steps counts individual transitions, so it must be a multiple of the vector env lane count.
    rollout_device controls where the env wrapper emits tensors and where policy inference runs. If omitted, the
    current env/rollout-state device is used.
    _store_transitions=False is only for startup warmup before replay has any stored observations.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")
    if n_steps % replay_buffer.n_envs != 0:
        raise ValueError(
            f"n_steps must be a multiple of replay_buffer.n_envs ({replay_buffer.n_envs}) because off-policy "
            "rollout stores full vector-env steps."
        )
    if policy is None and not random_actions:
        raise ValueError("Either pass a policy or set random_actions=True.")
    if not _store_transitions and len(replay_buffer) > 0:
        raise ValueError("Discarded off-policy rollout warmup requires an empty replay buffer.")
    if rollout_state is None and len(replay_buffer) > 0:
        raise ValueError("A non-empty replay buffer requires a rollout_state to preserve observation-slot continuity.")

    policy_requires_previous_actions = policy is not None and policy.requires_previous_actions()
    if policy_requires_previous_actions and not replay_buffer.store_previous_actions:
        raise ValueError(
            "Policies that require previous actions need replay_buffer.store_previous_actions=True so sampled "
            "batches contain the same conditioning inputs used during rollout."
        )
    track_previous_actions = replay_buffer.store_previous_actions or policy_requires_previous_actions
    env_reset_time = 0.0
    current_obs_is_stored = _store_transitions and rollout_state is not None and replay_buffer.has_current_obs

    resolved_rollout_device: torch.device | None = None
    to_rollout_device_time = 0.0
    with PerformanceTimer() as to_rollout_device_timer:
        if rollout_device is not None:
            resolved_rollout_device = as_device(rollout_device)
            env.set_device(resolved_rollout_device)
            if rollout_state is not None:
                rollout_state = _move_rollout_state_to_device(
                    rollout_state=rollout_state,
                    device=resolved_rollout_device,
                )
    to_rollout_device_time += to_rollout_device_timer.get_duration()

    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            rollout_state = _reset_rollout_state(
                env=env,
                policy=policy,
                track_previous_actions=track_previous_actions,
                n_agent_actions=replay_buffer.n_agent_actions,
            )
        env_reset_time = env_reset_timer.get_duration()

    obs = rollout_state.obs
    episode_start_mask = rollout_state.episode_start_mask
    previous_actions = rollout_state.previous_actions
    temporal_state = rollout_state.temporal_state
    rollout_step_idx = rollout_state.rollout_step_idx
    gsde_noise_state = rollout_state.gsde_noise_state
    gsde_noise_initialized = gsde_noise_state is not None
    if policy is not None and temporal_state is None:
        initial_temporal_state = policy.initial_temporal_state(
            batch_size=obs["local_obs"].shape[0],
            n_agents=obs["local_obs"].shape[1],
            device=obs["local_obs"].device,
            dtype=obs["local_obs"].dtype,
        )
        if initial_temporal_state is not None and not bool(episode_start_mask.all()):
            raise ValueError(
                "Cannot introduce a recurrent policy after random rollout has entered an episode. "
                "Pass the policy during random collection so its temporal state is advanced."
            )
        temporal_state = initial_temporal_state

    advance_policy_state_with_random_actions = (
        random_actions
        and policy is not None
        and temporal_state is not None
    )

    with PerformanceTimer() as policy_to_device_timer:
        if policy is not None:
            policy.to(resolved_rollout_device if resolved_rollout_device is not None else obs["local_obs"].device)
            policy.eval()
    to_rollout_device_time += policy_to_device_timer.get_duration()
    active_gsde_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=(
            policy is not None
            and (not random_actions or advance_policy_state_with_random_actions)
            and not deterministic
            and policy.gsde_enabled
        ),
        reset_mode=gsde_reset_mode,
    )
    if policy is not None and active_gsde_reset_mode is not None and gsde_noise_state is not None:
        policy.action_dist.set_temporal_correlation_state(gsde_noise_state)

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    transitions_collected = 0
    rollout_action_metrics_batches: list[torch.Tensor] = []
    while transitions_collected < n_steps:
        current_obs_needs_copy = not current_obs_is_stored
        obs_for_step = snapshot_obs(obs) if current_obs_needs_copy else obs

        with timers.reset_noise_timer:
            if policy is not None and active_gsde_reset_mode is not None:
                force_gsde_step_reset = not gsde_noise_initialized
                _reset_policy_action_noise(
                    policy=policy,
                    local_obs=obs_for_step["local_obs"],
                    episode_start_mask=episode_start_mask,
                    rollout_step_idx=rollout_step_idx,
                    gsde_reset_mode=active_gsde_reset_mode,
                    force_step_reset=force_gsde_step_reset,
                )
                gsde_noise_initialized = True
        timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

        with timers.policy_forward_timer:
            if random_actions:
                actions = sample_random_actions(
                    env.action_space,
                    device=obs_for_step["local_obs"].device,
                    dtype=obs_for_step["local_obs"].dtype,
                )
                if advance_policy_state_with_random_actions:
                    assert policy is not None
                    _unused_policy_actions, next_temporal_state = policy.act_with_temporal_state(
                        local_obs=obs_for_step["local_obs"],
                        global_obs=obs_for_step["global_obs"],
                        hidden_local_vars=obs_for_step["hidden_local_vars"],
                        hidden_global_vars=obs_for_step["hidden_global_vars"],
                        agent_mask=obs_for_step.get("agent_mask", None),
                        **(
                            {} if "scenario_id" not in obs_for_step else {"scenario_ids": obs_for_step["scenario_id"]}
                        ),
                        previous_actions=previous_actions if policy_requires_previous_actions else None,
                        deterministic=deterministic,
                        temporal_state=temporal_state,
                        episode_start_mask=episode_start_mask,
                    )
                else:
                    next_temporal_state = temporal_state
            else:
                assert policy is not None
                actions, next_temporal_state = policy.act_with_temporal_state(
                    local_obs=obs_for_step["local_obs"],
                    global_obs=obs_for_step["global_obs"],
                    hidden_local_vars=obs_for_step["hidden_local_vars"],
                    hidden_global_vars=obs_for_step["hidden_global_vars"],
                    agent_mask=obs_for_step.get("agent_mask", None),
                    **(
                        {} if "scenario_id" not in obs_for_step else {"scenario_ids": obs_for_step["scenario_id"]}
                    ),
                    previous_actions=previous_actions if policy_requires_previous_actions else None,
                    deterministic=deterministic,
                    temporal_state=temporal_state,
                    episode_start_mask=episode_start_mask,
                )
        timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())
        actions_for_metrics = actions.detach()
        agent_mask_for_metrics = obs_for_step.get("agent_mask", None)
        if agent_mask_for_metrics is None:
            rollout_action_metrics_batches.append(actions_for_metrics.reshape(-1, actions_for_metrics.shape[-1]))
        else:
            rollout_action_metrics_batches.append(actions_for_metrics[agent_mask_for_metrics])

        with timers.env_step_timer:
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
        timers.env_step_timings.append(timers.env_step_timer.get_duration())

        dones = torch.logical_or(terminations, truncations)
        append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)
        terminal_obs = extract_terminal_obs(
            env=env,
            next_obs=next_obs,
            infos=infos,
            dones=dones,
        )

        rollout_next_previous_actions: torch.Tensor | None = None
        if previous_actions is not None:
            rollout_next_previous_actions = actions.detach().masked_fill(
                dones.unsqueeze(-1).unsqueeze(-1),
                0.0,
            )

        if _store_transitions:
            with timers.buffer_add_timer:
                replay_buffer.add(
                    obs=obs_for_step,
                    actions=actions.detach(),
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    next_obs=next_obs,
                    terminal_obs=terminal_obs,
                    previous_actions=previous_actions,
                    episode_start_mask=episode_start_mask,
                    temporal_state=temporal_state,
                    next_temporal_state=next_temporal_state,
                    copy_current_obs=current_obs_needs_copy,
                )
            timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

        obs = next_obs
        episode_start_mask = dones
        previous_actions = rollout_next_previous_actions
        temporal_state = next_temporal_state
        current_obs_is_stored = _store_transitions
        rollout_step_idx += 1
        transitions_collected += replay_buffer.n_envs

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_time,
        timers=timers,
        episode_infos=episode_infos,
        transitions_collected=transitions_collected,
    )
    if rollout_action_metrics_batches:
        metrics["_rollout_actions"] = torch.cat(rollout_action_metrics_batches, dim=0)
    if policy is not None and active_gsde_reset_mode is not None:
        gsde_noise_state = policy.action_dist.get_temporal_correlation_state()
    new_state = OffPolicyRolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
        gsde_noise_state=gsde_noise_state,
    )
    return episode_infos, metrics, new_state


@torch.no_grad()
def warmup_off_policy_steps(
        env: BaseLearnEnvWrapper,
        replay_buffer: OffPolicyReplayBuffer,
        n_steps: int,
        *,
        policy: BasePolicy | None = None,
        rollout_state: OffPolicyRolloutState | None = None,
        random_actions: bool = False,
        deterministic: bool = False,
        gsde_reset_mode: GSDEResetMode | None = None,
        rollout_device: torch.device | str | None = None,
) -> OffPolicyRolloutState:
    _episode_infos, _metrics, new_state = collect_off_policy_steps(
        env=env,
        replay_buffer=replay_buffer,
        n_steps=n_steps,
        policy=policy,
        rollout_state=rollout_state,
        random_actions=random_actions,
        deterministic=deterministic,
        gsde_reset_mode=gsde_reset_mode,
        rollout_device=rollout_device,
        _store_transitions=False,
    )
    return new_state
