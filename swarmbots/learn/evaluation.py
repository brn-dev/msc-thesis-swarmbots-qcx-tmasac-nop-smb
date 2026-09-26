from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from swarmbots.learn.checkpointing import apply_env_state, capture_env_state, freeze_env_normalization
from swarmbots.learn.metrics_logger import MetricsLogger
from swarmbots.learn.rollout_utils import initial_previous_actions, to_python_episode_stat
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.temporal_state import clone_detach_temporal_state
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH


DEFAULT_EVALUATION_MILESTONES: tuple[float, ...] = (25, 50, 75, 90, 95, 100)
EVALUATION_ACTION_MODES: tuple[tuple[str, bool], ...] = (
    ("stochastic", False),
    ("deterministic", True),
)


@dataclass(frozen=True, slots=True)
class EvaluationRecordingConfig:
    num_episodes: int = 0
    frame_stride: int = 1
    fps: int = 30
    fps_mode: str = "compensate_stride"
    width: int = DEFAULT_RECORDING_WIDTH
    height: int = DEFAULT_RECORDING_HEIGHT
    camera: int | str = -1


class FrozenEvaluationRunner:
    def __init__(
        self,
        *,
        make_env: Callable[[], Any],
        training_env: Any,
        policy: Any,
        episodes_per_env: int = 1,
        seed: int = 1_000_000,
        recording_config: EvaluationRecordingConfig | None = None,
        video_folder: Path | None = None,
    ) -> None:
        if episodes_per_env <= 0:
            raise ValueError(f"episodes_per_env must be positive, got {episodes_per_env}")
        if recording_config is not None and recording_config.num_episodes < 0:
            raise ValueError(
                f"evaluation recording episode count must be non-negative, got {recording_config.num_episodes}"
            )

        self._make_env = make_env
        self._training_env = training_env
        self._policy = policy
        self.episodes_per_env = int(episodes_per_env)
        self.seed = int(seed)
        self.recording_config = recording_config
        self.video_folder = video_folder
        self._env: Any | None = None

    @property
    def env(self) -> Any:
        if self._env is None:
            self._env = self._make_env()
        return self._env

    def evaluate(self, *, timesteps: int, milestone_percentage: float) -> dict[str, Any]:
        env = self.env
        apply_env_state(env, capture_env_state(self._training_env))
        freeze_env_normalization(env)

        action_dist = getattr(self._policy, "action_dist", None)
        action_dist_state = (
            clone_detach_temporal_state(action_dist.get_temporal_correlation_state())
            if action_dist is not None and hasattr(action_dist, "get_temporal_correlation_state")
            else None
        )
        module_training_modes = {
            module: module.training
            for module in self._policy.modules()
        }
        cuda_devices = self._cuda_rng_devices()
        started_at = time.perf_counter()
        metrics: dict[str, Any] = {}

        try:
            with torch.random.fork_rng(devices=cuda_devices):
                for action_mode, deterministic in EVALUATION_ACTION_MODES:
                    mode_started_at = time.perf_counter()
                    self._seed_rng(cuda_devices)
                    if action_dist_state is not None:
                        action_dist.set_temporal_correlation_state(
                            clone_detach_temporal_state(action_dist_state)
                        )
                    obs, _info = env.reset(
                        seed=self.seed,
                        options={"force_settled": True},
                    )
                    if deterministic:
                        self._start_recording(
                            env=env,
                            timesteps=timesteps,
                            milestone_percentage=milestone_percentage,
                            action_mode=action_mode,
                        )
                    episode_metrics = self._collect_episodes(
                        env=env,
                        initial_obs=obs,
                        deterministic=deterministic,
                    )
                    metrics.update(self._summarize_metrics(episode_metrics, action_mode=action_mode))
                    metrics[f"eval_{action_mode}_duration"] = time.perf_counter() - mode_started_at
        finally:
            if action_dist is not None and hasattr(action_dist, "set_temporal_correlation_state"):
                action_dist.set_temporal_correlation_state(action_dist_state)
            for module, was_training in module_training_modes.items():
                module.training = was_training

        duration = time.perf_counter() - started_at
        episodes_per_mode = env.num_envs * self.episodes_per_env
        metrics.update(
            {
                "eval_duration": duration,
                "eval_episodes": episodes_per_mode * len(EVALUATION_ACTION_MODES),
                "eval_episodes_per_mode": episodes_per_mode,
                "eval_episodes_per_env": self.episodes_per_env,
                "eval_milestone_pct": float(milestone_percentage),
                "timesteps": int(timesteps),
            }
        )
        return metrics

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _collect_episodes(
        self,
        *,
        env: Any,
        initial_obs: dict[str, torch.Tensor],
        deterministic: bool,
    ) -> dict[str, list[float]]:
        self._policy.eval()
        obs = initial_obs
        previous_actions = initial_previous_actions(
            policy=self._policy,
            obs=obs,
            n_agent_actions=env.action_space.total_agent_action_dim,
        )
        temporal_state = self._policy.initial_temporal_state(
            batch_size=env.num_envs,
            n_agents=env.n_agents,
            device=obs["local_obs"].device,
            dtype=obs["local_obs"].dtype,
        )
        episode_start_mask = torch.ones(
            (env.num_envs,),
            device=obs["local_obs"].device,
            dtype=torch.bool,
        )
        completed_per_env = torch.zeros_like(episode_start_mask, dtype=torch.int64)
        episode_metrics: dict[str, list[float]] = {}
        rollout_step_idx = 0
        action_dist = getattr(self._policy, "action_dist", None)

        with torch.no_grad():
            while bool(torch.any(completed_per_env < self.episodes_per_env)):
                if action_dist is not None and hasattr(action_dist, "reset_temporal_correlations_on_ep_start"):
                    action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
                if (
                    rollout_step_idx == 0
                    and not deterministic
                    and bool(getattr(self._policy, "gsde_enabled", False))
                    and action_dist is not None
                    and hasattr(action_dist, "reset_temporal_correlations_on_step")
                ):
                    action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(obs["local_obs"].shape[:-1]))

                actions, temporal_state = self._policy.act_with_temporal_state(
                    local_obs=obs["local_obs"],
                    global_obs=obs["global_obs"],
                    hidden_local_vars=obs["hidden_local_vars"],
                    hidden_global_vars=obs["hidden_global_vars"],
                    agent_mask=obs.get("agent_mask"),
                    **({} if "scenario_id" not in obs else {"scenario_ids": obs["scenario_id"]}),
                    previous_actions=previous_actions,
                    deterministic=deterministic,
                    temporal_state=temporal_state,
                    episode_start_mask=episode_start_mask,
                )
                obs, _rewards, terminations, truncations, infos = env.step(actions)
                dones = torch.logical_or(terminations, truncations)
                accepted_dones = dones & (completed_per_env < self.episodes_per_env)
                self._append_episode_metrics(
                    episode_metrics=episode_metrics,
                    infos=infos,
                    accepted_dones=accepted_dones,
                    dones=dones,
                )
                completed_per_env += accepted_dones.to(dtype=completed_per_env.dtype)
                episode_start_mask = dones.to(device=obs["local_obs"].device, dtype=torch.bool)
                if previous_actions is not None:
                    previous_actions = actions.detach().masked_fill(dones[:, None, None], 0.0)
                rollout_step_idx += 1

        return episode_metrics

    @staticmethod
    def _append_episode_metrics(
        *,
        episode_metrics: dict[str, list[float]],
        infos: dict[str, Any],
        accepted_dones: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        if not torch.any(accepted_dones):
            return

        info_source = infos
        if "episode" not in info_source:
            final_info = infos.get("final_info")
            if isinstance(final_info, dict) and "episode" in final_info:
                info_source = final_info
            else:
                raise ValueError("Evaluation requires episode statistics in infos['episode'].")

        episode_stats = info_source["episode"]
        if not isinstance(episode_stats, Mapping):
            raise ValueError(f"Expected infos['episode'] to be a mapping, got {type(episode_stats)}")
        episode_mask = torch.as_tensor(
            info_source.get("_episode", dones),
            device=dones.device,
            dtype=torch.bool,
        ).reshape(-1)
        if not torch.equal(episode_mask, dones):
            raise ValueError("Expected infos['_episode'] to match evaluation dones.")

        for env_idx in torch.nonzero(accepted_dones, as_tuple=False).flatten().tolist():
            for key, values in episode_stats.items():
                if key.startswith("_"):
                    continue
                value_mask = episode_stats.get(f"_{key}")
                if value_mask is not None and not bool(to_python_episode_stat(value_mask[env_idx])):
                    continue
                value = to_python_episode_stat(values[env_idx])
                if isinstance(value, (bool, int, float)):
                    episode_metrics.setdefault(key, []).append(float(value))

    @staticmethod
    def _summarize_metrics(
        episode_metrics: dict[str, list[float]],
        *,
        action_mode: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        summary_keys = {
            "r": "ep_rew",
            "l": "ep_len",
            "progress_reward": "progress_reward",
            "guidance_reward": "guidance_reward",
            "successful_connections_per_unit_mean": "successful_connections_per_unit_mean",
            "successful_connections_per_unit_std": "successful_connections_per_unit_std",
            "successful_connections_per_unit_min": "successful_connections_per_unit_min",
            "successful_connections_per_unit_max": "successful_connections_per_unit_max",
        }
        for source_key, metric_suffix in summary_keys.items():
            values = episode_metrics.get(source_key)
            if values:
                result[f"eval_{action_mode}_{metric_suffix}"] = compute_summary_statistics(
                    values,
                    find_min=True,
                    find_max=True,
                )
        successes = episode_metrics.get("success")
        if successes:
            result[f"eval_{action_mode}_success_rate"] = 100.0 * sum(successes) / len(successes)
        return result

    def _start_recording(
        self,
        *,
        env: Any,
        timesteps: int,
        milestone_percentage: float,
        action_mode: str,
    ) -> None:
        config = self.recording_config
        if config is None or config.num_episodes == 0 or self.video_folder is None:
            return
        start_recording = getattr(env.unwrapped, "start_video_recording", None)
        if not callable(start_recording):
            logger.warning("Evaluation recording requested, but the evaluation environment cannot record video.")
            return

        episode_count = min(config.num_episodes, env.num_envs * self.episodes_per_env)
        start_recording(
            video_folder=str(self.video_folder),
            video_name_prefix=(
                f"eval_{action_mode}_{_format_percentage(milestone_percentage)}pct_{timesteps}_steps"
            ),
            num_episodes=episode_count,
            max_parallel_episodes=min(episode_count, env.num_envs),
            fps=config.fps,
            fps_mode=config.fps_mode,
            frame_stride=config.frame_stride,
            width=config.width,
            height=config.height,
            camera=config.camera,
        )

    def _seed_rng(self, cuda_devices: list[int]) -> None:
        torch.random.default_generator.manual_seed(self.seed)
        for cuda_device in cuda_devices:
            with torch.cuda.device(cuda_device):
                torch.cuda.manual_seed(self.seed)

    def _cuda_rng_devices(self) -> list[int]:
        try:
            device = next(self._policy.parameters()).device
        except StopIteration:
            return []
        if device.type != "cuda":
            return []
        return [torch.cuda.current_device() if device.index is None else device.index]


class ScheduledEvaluationHook:
    def __init__(
        self,
        *,
        algorithm: Any,
        total_timesteps: int,
        start_timesteps: int = 0,
        milestones: Sequence[float],
        runner: FrozenEvaluationRunner,
        metrics_logger: MetricsLogger,
    ) -> None:
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
        if start_timesteps < 0 or start_timesteps >= total_timesteps:
            raise ValueError(
                "start_timesteps must be in [0, total_timesteps), got "
                f"{start_timesteps=} {total_timesteps=}"
            )
        self.runner = runner
        self.metrics_logger = metrics_logger
        current_timesteps = int(algorithm.n_total_timesteps)
        seen_targets: set[int] = set()
        self._pending: list[tuple[float, int]] = []
        for percentage in sorted(milestones):
            if percentage < 0 or percentage > 100:
                raise ValueError(f"Evaluation milestone must be in [0, 100], got {percentage}")
            target_timesteps = max(
                start_timesteps + 1,
                start_timesteps
                + int((total_timesteps - start_timesteps) * percentage / 100),
            )
            if target_timesteps in seen_targets or target_timesteps <= current_timesteps:
                continue
            seen_targets.add(target_timesteps)
            self._pending.append((float(percentage), target_timesteps))

    def __call__(self, algorithm: Any, metrics: dict[str, Any], rollout_steps: int) -> None:
        _ = metrics, rollout_steps
        if not self._pending or algorithm.n_total_timesteps < self._pending[0][1]:
            return

        percentage, target_timesteps = self._pending.pop(0)
        while self._pending and algorithm.n_total_timesteps >= self._pending[0][1]:
            percentage, target_timesteps = self._pending.pop(0)

        logger.warning(
            f"Running frozen evaluation for {percentage:g}% milestone "
            f"(target={target_timesteps}, actual={algorithm.n_total_timesteps})."
        )
        evaluation_metrics = self.runner.evaluate(
            timesteps=int(algorithm.n_total_timesteps),
            milestone_percentage=percentage,
        )
        self.metrics_logger.log(evaluation_metrics)
        self.metrics_logger.flush()

    def close(self) -> None:
        try:
            self.runner.close()
        finally:
            self.metrics_logger.close()


def _format_percentage(percentage: float) -> str:
    if percentage.is_integer():
        return f"{int(percentage):03d}"
    return str(percentage).rstrip("0").rstrip(".").replace(".", "p")
