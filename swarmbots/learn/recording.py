import os
from typing import Any

import moviepy.video.io.ImageSequenceClip
import numpy as np
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEIntervalResetMode,
    GSDEProbabilityResetMode,
    resolve_gsde_reset_mode,
)
from swarmbots.utils.recording_overlay import draw_accumulated_reward
from swarmbots.learn.tensor_conversion import to_numpy_array


def _get_episode_stat(
    infos: dict[str, Any],
    *,
    env_idx: int,
    key: str,
) -> float | None:
    episode_stats = infos.get("episode")
    if not isinstance(episode_stats, dict):
        return None

    episode_mask = infos.get("_episode")
    if episode_mask is not None:
        done_mask = to_numpy_array(episode_mask, dtype=bool).reshape(-1)
        if env_idx >= done_mask.shape[0] or not bool(done_mask[env_idx]):
            return None

    values = episode_stats.get(key)
    if values is None:
        return None

    values_array = to_numpy_array(values)
    if values_array.shape == ():
        return float(values_array)

    flattened_values = values_array.reshape(-1)
    if env_idx >= flattened_values.shape[0]:
        return None
    return float(flattened_values[env_idx])


def _maybe_reset_gsde_noise(
    *,
    policy: BasePolicy,
    local_obs: torch.Tensor,
    deterministic: bool,
    rollout_step_idx: int,
    gsde_reset_mode: GSDEResetMode | None,
) -> None:
    active_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=not deterministic and policy.gsde_enabled,
        reset_mode=gsde_reset_mode,
    )
    if active_reset_mode is None:
        return

    action_dist = getattr(policy, "action_dist", None)
    if action_dist is None or not hasattr(action_dist, "reset_temporal_correlations_on_step"):
        raise RuntimeError(
            "Policy reports gsde_enabled=True but has no action_dist.reset_temporal_correlations_on_step()."
        )

    batch_shape = tuple(local_obs.shape[:-1])
    if isinstance(active_reset_mode, GSDEIntervalResetMode):
        if (rollout_step_idx % active_reset_mode.interval) == 0:
            action_dist.reset_temporal_correlations_on_step(batch_shape=batch_shape)
        return

    if isinstance(active_reset_mode, GSDEProbabilityResetMode):
        mask = torch.empty(batch_shape, device=local_obs.device, dtype=torch.bool).bernoulli_(
            active_reset_mode.probability
        )
        action_dist.reset_temporal_correlations_on_step(mask=mask)
        return


def _extract_env_reward(reward: Any, *, env_idx: int = 0) -> float:
    reward_array = to_numpy_array(reward)
    if reward_array.shape == ():
        return float(reward_array)
    flattened = reward_array.reshape(-1)
    if env_idx >= flattened.shape[0]:
        raise IndexError(f"Reward does not contain env_idx={env_idx}.")
    return float(flattened[env_idx])


def _extract_reward_terms(infos: dict[str, Any], *, env_idx: int) -> dict[str, float]:
    reward_terms = infos.get("reward_terms")
    if not isinstance(reward_terms, dict):
        return {}

    extracted: dict[str, float] = {}
    for label, values in reward_terms.items():
        if str(label).startswith("_"):
            continue
        extracted[str(label)] = _extract_env_reward(values, env_idx=env_idx)
    return extracted


def _find_normalize_reward_wrapper(env: Any) -> TorchNormalizeRewardWrapper | None:
    current_env = env
    while hasattr(current_env, "env"):
        if isinstance(current_env, TorchNormalizeRewardWrapper):
            return current_env
        current_env = current_env.env
    return None


def _extract_raw_env_reward(
    reward: Any,
    *,
    env_idx: int,
    normalize_reward_wrapper: TorchNormalizeRewardWrapper | None,
) -> float:
    reward_value = _extract_env_reward(reward, env_idx=env_idx)
    if normalize_reward_wrapper is None:
        return reward_value

    denominator = float(
        torch.sqrt(
            normalize_reward_wrapper.return_rms.var.to(device="cpu", dtype=torch.float64)
            + normalize_reward_wrapper.epsilon
        ).item()
    )
    return reward_value * denominator


def _extract_render_frame(frame: Any) -> np.ndarray | None:
    if isinstance(frame, (list, tuple)):
        if len(frame) == 0:
            return None
        return _extract_render_frame(frame[0])
    if isinstance(frame, np.ndarray):
        if frame.ndim == 4:
            return frame[0]
        return frame
    return None


def record_policy(
    env: BaseLearnEnvWrapper,
    policy: BasePolicy,
    video_folder: str,
    video_name_prefix: str,
    num_episodes: int = 5,
    deterministic: bool = False,
    gsde_reset_mode: GSDEResetMode | None = None,
    fps: int = 30,
    device: torch.device = torch.device("cpu"),
):
    """
    Record episodes of a policy interacting with an environment.
    
    Args:
        env: The environment wrapper (BaseLearnEnvWrapper)
        policy: The policy to evaluate
        video_folder: Directory to save videos
        video_name_prefix: Prefix for video filenames
        num_episodes: Number of episodes to record
        deterministic: Whether to use deterministic actions
        gsde_reset_mode: Controls gSDE noise resampling strategy when deterministic=False.
        fps: Frames per second for the output video
        device: Torch device
    """
    os.makedirs(video_folder, exist_ok=True)
    
    policy.eval()
    policy.to(device)
    normalize_reward_wrapper = _find_normalize_reward_wrapper(env)

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        previous_actions: torch.Tensor | None = None
        if policy.requires_previous_actions():
            previous_actions = torch.zeros(
                (env.num_envs, env.n_agents, env.action_space.total_agent_action_dim),
                dtype=obs["local_obs"].dtype,
                device=obs["local_obs"].device,
            )
        temporal_state = policy.initial_temporal_state(
            batch_size=env.num_envs,
            n_agents=env.n_agents,
            device=obs["local_obs"].device,
            dtype=obs["local_obs"].dtype,
        )
        episode_start_mask = torch.ones((env.num_envs,), device=obs["local_obs"].device, dtype=torch.bool)
        action_dist = getattr(policy, "action_dist", None)
        frames = []
        ep_rew = None
        ep_progress_reward = None
        ep_guidance_reward = None
        accumulated_reward = 0.0
        accumulated_reward_terms: dict[str, float] = {}
                    
        try:
            first_frame = _extract_render_frame(env.render())
        except Exception as e:
            print(f"Warning: Could not render environment. Error: {e}")
            first_frame = None

        if first_frame is None:
            print("Environment render returned None. Make sure render_mode='rgb_array' is set.")
            return
        frames.append(
            draw_accumulated_reward(
                first_frame,
                accumulated_reward,
                accumulated_reward_terms=accumulated_reward_terms,
            )
        )

        done = False
        step_cnt = 0
        
        while not done:
            with torch.no_grad():
                local_obs = obs['local_obs']
                global_obs = obs['global_obs']
                hidden_local_vars = obs["hidden_local_vars"]
                hidden_global_vars = obs["hidden_global_vars"]
                agent_mask = obs.get("agent_mask", None)

                if action_dist is not None and hasattr(action_dist, "reset_temporal_correlations_on_ep_start"):
                    action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
                _maybe_reset_gsde_noise(
                    policy=policy,
                    local_obs=local_obs,
                    deterministic=deterministic,
                    rollout_step_idx=step_cnt,
                    gsde_reset_mode=gsde_reset_mode,
                )
                actions, temporal_state = policy.act_with_temporal_state(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    hidden_local_vars=hidden_local_vars,
                    hidden_global_vars=hidden_global_vars,
                    agent_mask=agent_mask,
                    **(
                        {}
                        if "scenario_id" not in obs
                        else {"scenario_ids": obs["scenario_id"]}
                    ),
                    previous_actions=previous_actions,
                    deterministic=deterministic,
                    temporal_state=temporal_state,
                    episode_start_mask=episode_start_mask,
                )
                # print(format_summary_statistics(compute_summary_statistics(actions[:, :, :8], make_histogram=True), SummaryStatisticsFormat(histogram=True)))
            
            obs, reward, term, trunc, infos = env.step(actions)
            episode_start_mask = torch.logical_or(term, trunc).to(device=obs["local_obs"].device, dtype=torch.bool)
            accumulated_reward += _extract_raw_env_reward(
                reward,
                env_idx=0,
                normalize_reward_wrapper=normalize_reward_wrapper,
            )
            for label, value in _extract_reward_terms(infos, env_idx=0).items():
                accumulated_reward_terms[label] = accumulated_reward_terms.get(label, 0.0) + value

            current_frame = _extract_render_frame(env.render())
            if current_frame is not None:
                frames.append(
                    draw_accumulated_reward(
                        current_frame,
                        accumulated_reward,
                        accumulated_reward_terms=accumulated_reward_terms,
                    )
                )
            
            if term[0] or trunc[0]:
                ep_rew = _get_episode_stat(infos, env_idx=0, key="r")
                ep_progress_reward = _get_episode_stat(infos, env_idx=0, key="progress_reward")
                ep_guidance_reward = _get_episode_stat(infos, env_idx=0, key="guidance_reward")
                done = True
            if previous_actions is not None:
                done_mask = torch.logical_or(term, trunc).unsqueeze(-1).unsqueeze(-1)
                previous_actions = actions.detach().masked_fill(done_mask, 0.0)
            
            step_cnt += 1

        ep_rew_str = f"{ep_rew:.4f}" if ep_rew is not None else "n/a"
        ep_progress_str = f"{ep_progress_reward:.4f}" if ep_progress_reward is not None else "n/a"
        ep_guidance_str = f"{ep_guidance_reward:.4f}" if ep_guidance_reward is not None else "n/a"
        print(
            f"Episode {episode_idx} finished after {step_cnt} steps. "
            f"ep_rew={ep_rew_str}, progress_reward={ep_progress_str}, guidance_reward={ep_guidance_str}"
        )

        if frames:
            video_path = os.path.join(
                video_folder,
                f"{video_name_prefix}_ep_{episode_idx}-ep_rew={ep_rew_str}.mp4",
            )
            try:
                clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
                clip.write_videofile(video_path, logger=None)
                print(f"Saved video to {video_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")
        else:
            print("No frames collected.")

