import abc
import json
import os
import pathlib
import queue
import threading
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Collection, Callable, Mapping, Self

import torch
from loguru import logger

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys, load_checkpoint, \
    extract_policy_state_dict, extract_optimizer_state_dict, apply_env_state, extract_env_state, \
    freeze_env_normalization, capture_env_state, move_env_to_device
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage, HybridEMA
from swarmbots.learn.metrics_logger import MetricsLogger, mean_std, rate, summed
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.recording import record_policy
from swarmbots.utils.machine_specs import collect_machine_specs
from swarmbots.utils.recording_resolution import DEFAULT_RECORDING_HEIGHT, DEFAULT_RECORDING_WIDTH




MIN_ITERATIONS_FOR_EMA = 10
MIN_ITERATIONS_FOR_BEST = 100

LearningRate = float | list[float] | dict[str, float]
LearnIterationHook = Callable[["BaseAlgorithm", dict[str, Any], int], None]
PolicyStateDictTransform = Callable[
    [dict[str, Any], Mapping[str, Any]],
    dict[str, Any],
]

MEAN_STD_LOG_METRIC_KEYS = frozenset({
    "rollout_time",
    "env_reset_time",
    "to_rollout_device_time",
    "total_reset_noise_time",
    "total_policy_forward_time",
    "total_env_step_time",
    "total_buffer_add_time",
    "buffer_get_whole_episodes_time",
    "rollout_batch_builder_init_time",
    "rollout_batch_finalize_time",
    "to_train_device_time",
    "sampler_init_time",
    "total_sampling_time",
    "total_update_time",
    "metrics_time",
    "train_time",
    "total_compute_grad_norms_time",
    "expl_var",
    "grad_clip_frac",
})
SUM_LOG_METRIC_KEYS = frozenset({"updates", "transitions_collected"})


def _apply_log_metric_reductions(metrics: dict[str, Any]) -> dict[str, Any]:
    logging_metrics = metrics.copy()
    for key in MEAN_STD_LOG_METRIC_KEYS:
        if key in logging_metrics and logging_metrics[key] is not None:
            logging_metrics[key] = mean_std(logging_metrics[key])
    for key in SUM_LOG_METRIC_KEYS:
        if key in logging_metrics and logging_metrics[key] is not None:
            logging_metrics[key] = summed(logging_metrics[key])
    return logging_metrics


class BaseAlgorithm(abc.ABC):

    def __init__(
            self,
            policy: BasePolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: LearningRate,
    ):
        self.policy = policy
        self.env = env

        self.learning_rate = learning_rate

        self.n_total_iterations = 0
        self.n_total_updates = 0
        self.n_total_timesteps = 0
        self._command_queue: queue.Queue[str] = queue.Queue()
        self._command_prompt_started: bool = False
        self._active_run_dir: Path | None = None
        self._active_extra_run_metadata: dict[str, Any] | None = None
        self._active_save_optimizer: bool = True
        self._last_return_ema: float | None = None
        self._final_return_ema: float | None = None
        self._best_return_ema: float | None = None
        self._latest_hp_update: str | None = None
        self._stop_requested = False
        self._stop_should_save = True
        self._stop_save_optimizer: bool | None = None
        self._make_record_env: Callable[[], BaseLearnEnvWrapper] | None = None
        self._command_log_path: Path | None = None
        self._pause_until_monotonic: float | None = None
        self._pause_notice_logged: bool = False
        self._active_max_total_timesteps: int | None = None
        self._active_learn_started_monotonic: float | None = None
        self._active_learn_started_timesteps: int | None = None

    @abc.abstractmethod
    def get_hyper_parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_learning_rate(self, lr: LearningRate):
        raise NotImplementedError()

    @abc.abstractmethod
    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        """
        :return: metrics, rollout_steps
        """
        raise NotImplementedError()

    def _before_learn_loop(self) -> None:
        pass

    def learn(
            self,
            max_total_timesteps: int | None = None,
            additional_timesteps: int | None = None,
            run_dir: Optional[str | pathlib.Path] = None,
            log_interval: int = 1,
            save_interval: Optional[int] = None,
            save_optimizer: bool = True,
            extra_run_metadata: dict[str, Any] | None = None,
            episode_return_ema_alpha: float = 0.01,
            best_rotation_n: int = 1,
            wandb_project: str | None = None,
            wandb_entity: str | None = None,
            wandb_run_name: str | None = None,
            wandb_group: str | None = None,
            wandb_tags: list[str] | None = None,
            wandb_mode: str | None = None,
            wandb_kwargs: dict[str, Any] | None = None,
            logging_ignore_keys_for_persistence: list[str] | None = None,
            logging_console_keys: Collection[str] | Collection[tuple[str, str | None]] | None = None,
            logging_buffer_size: int = 1,
            compress_metrics_log_on_exit: bool = False,
            enable_command_prompt: bool = True,
            make_record_env: Callable[[], BaseLearnEnvWrapper] | None = None,
            post_iteration_hooks: Collection[LearnIterationHook] | None = None,
    ) -> Self:
        assert (
                (max_total_timesteps is not None and max_total_timesteps > 0 and additional_timesteps is None)
                or
                (additional_timesteps is not None and additional_timesteps > 0 and max_total_timesteps is None)
        )
        assert best_rotation_n >= 1
        if log_interval is not None and log_interval < 1:
            raise ValueError(f"log_interval must be >= 1 or None, got {log_interval}")
        if logging_buffer_size < 1:
            raise ValueError(f"logging_buffer_size must be >= 1, got {logging_buffer_size}")

        if max_total_timesteps is None:
            max_total_timesteps = self.n_total_timesteps + additional_timesteps

        if run_dir is not None:
            run_dir = pathlib.Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            self._write_run_metadata(run_dir, extra_run_metadata)

        best_models_dir: pathlib.Path | None = None
        learn_started_at = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        learn_started_iterations = self.n_total_iterations
        self._latest_hp_update = learn_started_at
        if run_dir is not None:
            best_models_dir = run_dir / "models" / "best" / learn_started_at

        wandb_config: dict[str, Any] | None = None
        if wandb_project is not None:
            wandb_config = {
                "hyper_parameters": self.get_hyper_parameters(),
                "machine_specs": collect_machine_specs(),
            }
            if extra_run_metadata:
                wandb_config["extra_run_metadata"] = json.loads(json.dumps(extra_run_metadata, default=str))
            if run_dir is not None:
                wandb_config["run_dir"] = str(run_dir)

            if wandb_run_name is None and run_dir is not None:
                wandb_run_name = run_dir.name

        metric_logger = MetricsLogger(
            log_dir=run_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_run_name=wandb_run_name,
            wandb_group=wandb_group,
            wandb_tags=wandb_tags,
            wandb_config=wandb_config,
            wandb_mode=wandb_mode,
            wandb_kwargs=wandb_kwargs,
            wandb_step_key="timesteps",
            ignore_keys_for_persistence=logging_ignore_keys_for_persistence,
            console_keys=logging_console_keys,
            buffer_size=logging_buffer_size,
        )
        episode_return_ema = HybridEMA(alpha=episode_return_ema_alpha)
        episode_success_rate_ema = HybridEMA(alpha=episode_return_ema_alpha)
        best_return_ema: float | None = self._best_return_ema
        best_save_counter = 0

        self._active_run_dir = run_dir
        self._active_extra_run_metadata = extra_run_metadata
        self._active_save_optimizer = save_optimizer
        self._active_max_total_timesteps = max_total_timesteps
        self._active_learn_started_monotonic = time.monotonic()
        self._active_learn_started_timesteps = self.n_total_timesteps
        self._stop_requested = False
        self._stop_should_save = True
        self._stop_save_optimizer = None
        self._last_return_ema = None
        self._final_return_ema = None
        self._make_record_env = make_record_env
        self._command_log_path = None if run_dir is None else (run_dir / "command_log.jsonl")
        should_compress_metrics_log = False

        try:
            if enable_command_prompt:
                self._maybe_start_command_prompt()

            self._before_learn_loop()

            while self.n_total_timesteps < max_total_timesteps:
                if enable_command_prompt:
                    self._poll_and_execute_commands(run_dir=run_dir, extra_run_metadata=extra_run_metadata)
                    if self._stop_requested:
                        break

                iter_timer = PerformanceTimer().start()
                metrics, rollout_steps = self.perform_iteration(
                    episode_return_ema,
                    episode_success_rate_ema,
                    update_ema=self.n_total_iterations >= MIN_ITERATIONS_FOR_EMA
                )
                iter_duration = iter_timer.stop().get_duration()

                if post_iteration_hooks is not None:
                    for hook in post_iteration_hooks:
                        hook(self, metrics, rollout_steps)

                current_return_ema = episode_return_ema.get()
                current_success_rate_ema = episode_success_rate_ema.get()
                current_success_rate_ema_percent = (
                    None if current_success_rate_ema is None else 100.0 * current_success_rate_ema
                )
                self._last_return_ema = current_return_ema
                if (current_return_ema is not None and
                        self.n_total_iterations - learn_started_iterations >= MIN_ITERATIONS_FOR_BEST):
                    best_return_ema, best_save_counter = self._maybe_save_best_ema_model(
                        best_models_dir=best_models_dir,
                        best_rotation_n=best_rotation_n,
                        save_optimizer=save_optimizer,
                        current_episode_return_ema=current_return_ema,
                        best_episode_return_ema=best_return_ema,
                        best_save_counter=best_save_counter,
                    )
                    self._best_return_ema = best_return_ema

                if log_interval is not None and self.n_total_iterations % log_interval == 0:
                    metric_logger.log({
                        'learn_start': learn_started_at,
                        'latest_hp_update': self._latest_hp_update,
                        'iteration': self.n_total_iterations,
                        'timesteps': self.n_total_timesteps,
                        'learning_rate': self.learning_rate,
                        **_apply_log_metric_reductions(metrics),
                        'ep_rew_ema': current_return_ema,
                        'ep_success_rate_ema': current_success_rate_ema_percent,
                        'best_ep_rew_ema': best_return_ema,
                        'fps': rate(rollout_steps, iter_duration),
                    })

                if save_interval is not None and run_dir is not None and self.n_total_iterations % save_interval == 0:
                    save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps.pt"
                    self.save(
                        save_path,
                        optimizer_state_dict=self._get_optimizer_state_dict() if save_optimizer else None,
                        return_ema=current_return_ema
                    )
                    logger.log("SAVE", f"Saved model to {save_path.as_posix()}")

            if run_dir is not None:
                if self._stop_requested and not self._stop_should_save:
                    logger.warning("Stop requested: exiting without saving final model.")
                else:
                    suffix = "stopped" if self._stop_requested else "final"
                    save_path = run_dir / f"models/model_{self.n_total_timesteps}_steps_{suffix}.pt"
                    should_save_optimizer = save_optimizer
                    if self._stop_requested and self._stop_save_optimizer is not None:
                        should_save_optimizer = self._stop_save_optimizer
                    self.save(
                        save_path,
                        optimizer_state_dict=self._get_optimizer_state_dict() if should_save_optimizer else None,
                        return_ema=episode_return_ema.get()
                    )
                    logger.log("SAVE", f"Saved {suffix} model to {save_path.as_posix()}")

            should_compress_metrics_log = compress_metrics_log_on_exit

        finally:
            self._final_return_ema = self._last_return_ema
            try:
                try:
                    metric_logger.close()
                finally:
                    if should_compress_metrics_log:
                        metric_logger.compress_persisted_log()
            finally:
                self._active_run_dir = None
                self._active_extra_run_metadata = None
                self._active_save_optimizer = True
                self._active_max_total_timesteps = None
                self._active_learn_started_monotonic = None
                self._active_learn_started_timesteps = None
                self._stop_requested = False
                self._stop_should_save = True
                self._stop_save_optimizer = None
                self._last_return_ema = None
                self._latest_hp_update = None
                self._make_record_env = None
                self._command_log_path = None

        return self

    def set_learning_rate(self, lr: LearningRate) -> None:
        self.learning_rate = lr
        self._apply_learning_rate(lr)

    def _write_run_metadata(self, run_dir: Path, extra_run_metadata: dict[str, Any] | None) -> None:
        metadata: dict[str, Any] = {
            "algorithm": type(self).__name__,
            "hyper_parameters": self.get_hyper_parameters(),
            "policy_hyper_parameters": self.policy.get_hyper_parameters(),
            "policy_repr": str(self.policy),
            "env_repr": str(self.env),
        }
        if extra_run_metadata:
            metadata.update(extra_run_metadata)
        metadata["machine_specs"] = collect_machine_specs()
        metadata["iterations"] = self.n_total_iterations
        metadata["updates"] = self.n_total_updates
        metadata["timesteps"] = self.n_total_timesteps
        metadata_json: dict[str, Any] = json.loads(json.dumps(metadata, default=str))

        existing_metadata_files = list(run_dir.glob("run_metadata*.json"))
        if existing_metadata_files:
            def _sort_key(path: Path) -> tuple[int, float, str]:
                step = _parse_step_from_metadata_filename(path)
                step_key = step if step is not None else -1
                return step_key, path.stat().st_mtime, path.name

            latest_path = max(existing_metadata_files, key=_sort_key)
            latest_metadata = _read_metadata_json(latest_path)
            if _normalize_run_metadata_for_comparison(latest_metadata) == _normalize_run_metadata_for_comparison(metadata_json):
                return

        step = int(self.n_total_timesteps)
        base_path = run_dir / f"run_metadata_{step}.json"
        if base_path.exists():
            existing = _read_metadata_json(base_path)
            if _normalize_run_metadata_for_comparison(existing) == _normalize_run_metadata_for_comparison(metadata_json):
                return

            suffix_idx = 1
            while (run_dir / f"run_metadata_{step}_{suffix_idx}.json").exists():
                suffix_idx += 1
            metadata_path = run_dir / f"run_metadata_{step}_{suffix_idx}.json"
        else:
            metadata_path = base_path

        metadata_path.write_text(json.dumps(metadata_json, indent=2, sort_keys=True), encoding="utf-8")

    def _maybe_save_best_ema_model(
            self,
            best_models_dir: pathlib.Path | None,
            best_rotation_n: int,
            save_optimizer: bool,
            current_episode_return_ema: float,
            best_episode_return_ema: float | None,
            best_save_counter: int,
    ) -> tuple[float | None, int]:
        if best_episode_return_ema is not None and current_episode_return_ema <= best_episode_return_ema:
            return best_episode_return_ema, best_save_counter

        best_episode_return_ema = current_episode_return_ema
        self._best_return_ema = best_episode_return_ema
        if best_models_dir is None:
            return best_episode_return_ema, best_save_counter

        if best_rotation_n == 1:
            best_save_path = best_models_dir / "model_best.pt"
        else:
            best_save_idx = best_save_counter % best_rotation_n
            best_save_path = best_models_dir / f"model_best_{best_save_idx}.pt"
            best_save_counter += 1

        self.save(
            best_save_path,
            optimizer_state_dict=self._get_optimizer_state_dict() if save_optimizer else None,
            return_ema=current_episode_return_ema
        )
        logger.log(
            "SAVE",
            f"Saved best-EMA model to {best_save_path.as_posix()} (ep_rew_ema={best_episode_return_ema:.4f})",
        )
        return best_episode_return_ema, best_save_counter

    def save(
            self,
            path: str | Path,
            optimizer_state_dict: Optional[dict[str, Any]] = None,
            return_ema: Optional[float] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        env_state = capture_env_state(self.env)

        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'env_state': env_state,
            'n_total_iterations': self.n_total_iterations,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
            'best_return_ema': self._best_return_ema,
        }

        if optimizer_state_dict is not None:
            save_dict['optimizer_state_dict'] = optimizer_state_dict

        torch.save(save_dict, path)

        metadata = {
            'n_total_iterations': self.n_total_iterations,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
            'best_return_ema': self._best_return_ema,
        }
        metadata_path = path.with_name(f"{path.name}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def load(
            self,
            path: str | Path,
            *,
            map_location: Any | None = "cpu",
            recover_best_return_ema: bool = True,
            strict_load_state_dict: bool = True,
            policy_state_dict_transform: PolicyStateDictTransform | None = None,
            restore_optimizer_state: bool = True,
            restore_env_state: bool = True,
    ) -> None:
        checkpoint = load_checkpoint(path, map_location=map_location)
        target_state_dict = self.policy.state_dict()
        policy_state_dict = align_torch_compile_state_dict_keys(
            extract_policy_state_dict(checkpoint),
            target_keys=target_state_dict.keys(),
        )
        if policy_state_dict_transform is not None:
            policy_state_dict = policy_state_dict_transform(
                policy_state_dict,
                target_state_dict,
            )
        missing_keys, unexpected_keys = self.policy.load_state_dict(
            policy_state_dict,
            strict=strict_load_state_dict
        )
        if missing_keys or unexpected_keys:
            logger.warning(f'{missing_keys = },  {unexpected_keys = }')

        optimizer_state_dict = (
            extract_optimizer_state_dict(checkpoint)
            if restore_optimizer_state
            else None
        )
        if optimizer_state_dict is not None:
            if missing_keys or unexpected_keys:
                logger.warning(f'Skipping optimizer state dict load, as we have missing/unexpected keys')
                self._apply_learning_rate(self.learning_rate)
            else:
                self._apply_optimizer_state_dict(
                    optimizer_state_dict,
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                )
        else:
            self._apply_learning_rate(self.learning_rate)

        if isinstance(checkpoint, dict):
            self.n_total_iterations = checkpoint.get("n_total_iterations", 0)
            self.n_total_updates = checkpoint.get("n_total_updates", 0)
            self.n_total_timesteps = checkpoint.get("n_total_timesteps", 0)
            if recover_best_return_ema:
                self._best_return_ema = checkpoint.get("best_return_ema", checkpoint.get("return_ema", None))

        if restore_env_state:
            apply_env_state(self.env, extract_env_state(checkpoint))

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> bool:
        """
        :return: True if the command was executed successfully and updated the hyper parameters, False otherwise
        """
        if cmd == 'show_hps':
            logger.info(self.get_hyper_parameters())
            return False
        elif cmd in {"show_run_path", "show_id"}:
            self._cmd_show_run_path(params)
            return False
        elif cmd in {"pid", "show_pid"}:
            self._cmd_show_pid(params)
            return False
        elif cmd in {"show_reward_weights", "show_rw"}:
            self._cmd_show_reward_weights(params)
            return False
        elif cmd in {"show_eta", "eta"}:
            self._cmd_show_eta(params)
            return False
        elif cmd == 'set_lr':
            lr = json.loads(params)
            logger.warning(f'Setting learning rate to {lr}')
            self.set_learning_rate(lr)
            return True
        elif cmd in {'set_reward_weights', 'set_rw'}:
            self._cmd_set_reward_weights(params, extra_run_metadata)
            return True
        elif cmd == 'save':
            self._cmd_save(params)
            return False
        elif cmd == 'stop' or cmd == 'quit' or cmd == 'exit':
            self._cmd_stop(params)
            return False
        elif cmd == 'record':
            self._cmd_record(params)
            return False
        elif cmd in {'record_status', 'record_progress', 'recording_status', 'recording_progress'}:
            self._cmd_record_status(params)
            return False
        elif cmd == 'pause':
            self._cmd_pause(params)
            return False
        elif cmd in {'unpause', 'resume'}:
            self._cmd_unpause(params)
            return False
        else:
            logger.error(f'Unknown command "{cmd}"')
            return False

    def execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> bool:
        """
        :return: True if the command was executed successfully and updated the hyper parameters, False otherwise
        """
        try:
            return self._execute_command(cmd.strip().lower(), params.strip(), extra_run_metadata)
        except Exception:
            logger.exception('Executing command failed')
            return False

    def _maybe_start_command_prompt(self) -> None:
        if self._command_prompt_started:
            return
        self._command_prompt_started = _start_command_prompt(self._command_queue)

    def _poll_and_execute_commands(
            self,
            run_dir: Path | None,
            extra_run_metadata: dict[str, Any] | None
    ) -> None:
        while True:
            hps_updated: bool = False
            while True:
                try:
                    raw = self._command_queue.get_nowait()
                except queue.Empty:
                    break
                updated, updated_commands = self._execute_command_line(raw, extra_run_metadata)
                if updated and updated_commands:
                    self._append_command_log(raw=raw, updated_commands=updated_commands)
                hps_updated |= updated

            if hps_updated:
                self._latest_hp_update = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if hps_updated and run_dir is not None:
                self._write_run_metadata(run_dir, extra_run_metadata)

            if self._stop_requested:
                return

            if self._pause_until_monotonic is None:
                return

            remaining = self._pause_until_monotonic - time.monotonic()
            if remaining <= 0:
                self._pause_until_monotonic = None
                self._pause_notice_logged = False
                logger.warning("Pause finished. Resuming training.")
                return

            if not self._pause_notice_logged:
                logger.warning(f"Training paused for {_format_duration_seconds(remaining)}. Commands still accepted.")
                self._pause_notice_logged = True

            time.sleep(min(remaining, 0.25))

    def _execute_command_line(
            self,
            raw: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> tuple[bool, list[dict[str, Any]]]:
        raw = raw.strip()
        if not raw:
            return False, []

        hps_updated: bool = False
        updated_commands: list[dict[str, Any]] = []
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" in chunk:
                cmd, params = chunk.split(":", 1)
            else:
                cmd, params = chunk, ""
            cmd = cmd.strip()
            if not cmd:
                continue
            updated = self.execute_command(cmd, params, extra_run_metadata)
            hps_updated |= updated
            if updated:
                updated_commands.append({"cmd": cmd, "params": params.strip()})
        return hps_updated, updated_commands

    def _append_command_log(self, raw: str, updated_commands: list[dict[str, Any]]) -> None:
        if self._command_log_path is None:
            return

        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "timesteps": int(self.n_total_timesteps),
            "iterations": int(self.n_total_iterations),
            "raw": raw.strip(),
            "updated_commands": updated_commands,
            "hyper_parameters": self.get_hyper_parameters(),
        }

        try:
            self._command_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._command_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            logger.exception(f"Failed to append command log to {self._command_log_path.as_posix()}")

    def _cmd_show_run_path(self, params: str) -> None:
        _ = params
        if self._active_run_dir is None:
            logger.info({"run_path": None, "run_id": None})
            return

        run_dir = self._active_run_dir.resolve()
        logger.info({"run_path": run_dir.as_posix(), "run_id": run_dir.name})

    def _cmd_show_pid(self, params: str) -> None:
        _ = params
        logger.info({"pid": os.getpid()})

    def _cmd_save(self, params: str) -> None:
        config = _parse_params_maybe_json(params)
        if isinstance(config, dict):
            path = config.get("path", None)
            name = config.get("name", None)
            include_optimizer = config.get("optimizer", None)
        else:
            path = params.strip() if params.strip() else None
            name = None
            include_optimizer = None

        if include_optimizer is None:
            include_optimizer = self._active_save_optimizer
        else:
            include_optimizer = _parse_bool(include_optimizer)

        save_path: Path
        if path:
            save_path = Path(path)
            if not save_path.is_absolute() and self._active_run_dir is not None:
                save_path = self._active_run_dir / "models" / save_path
        elif name:
            if self._active_run_dir is None:
                raise ValueError("save:name requires an active run_dir (or pass save:path)")
            save_path = self._active_run_dir / "models" / str(name)
        else:
            if self._active_run_dir is None:
                raise ValueError("save requires a path when learn() has no run_dir")
            save_path = self._active_run_dir / "models" / f"model_{self.n_total_timesteps}_steps_cmd.pt"

        self.save(
            save_path,
            optimizer_state_dict=self._get_optimizer_state_dict() if include_optimizer else None,
            return_ema=self._last_return_ema,
        )
        logger.log("SAVE", f"Saved model to {save_path.as_posix()}")

    def _cmd_stop(self, params: str) -> None:
        config = _parse_params_maybe_json(params)
        if isinstance(config, dict):
            save = _parse_bool(config.get("save", True))
            optimizer = config.get("optimizer", None)
            self._stop_save_optimizer = None if optimizer is None else _parse_bool(optimizer)
        else:
            save = True
            if params.strip():
                save = _parse_bool(params.strip())
        self._stop_requested = True
        self._stop_should_save = save
        logger.warning(f"Stop requested (save={save}). Will stop before next iteration.")

    def _cmd_pause(self, params: str) -> None:
        seconds = _parse_duration_to_seconds(params)
        if seconds <= 0:
            raise ValueError("pause expects a positive duration, e.g. pause:90s or pause:1h30m10")

        self._pause_until_monotonic = time.monotonic() + seconds
        self._pause_notice_logged = False
        logger.warning(f"Pausing training for {_format_duration_seconds(seconds)}")

    def _cmd_unpause(self, params: str) -> None:
        _ = params
        if self._pause_until_monotonic is None:
            logger.warning("Training is not paused.")
            return

        self._pause_until_monotonic = None
        self._pause_notice_logged = False
        logger.warning("Pause canceled. Resuming training.")

    def _cmd_record(self, params: str) -> None:
        config = _parse_params_maybe_json(params)
        if isinstance(config, dict):
            num_episodes = int(config.get("episodes", 3))
            deterministic = _parse_bool(config.get("deterministic", False))
            fps = int(config.get("fps", 30))
            fps_mode = str(config.get("fps_mode", "compensate_stride"))
            prefix = str(config.get("prefix", f"record_{self.n_total_timesteps}"))
            video_folder = config.get("folder", None)
            max_parallel_episodes = int(config.get("parallel", config.get("max_parallel", min(num_episodes, 4))))
            frame_stride = int(config.get("frame_stride", 1))
            width = int(config.get("width", DEFAULT_RECORDING_WIDTH))
            height = int(config.get("height", DEFAULT_RECORDING_HEIGHT))
            camera = config.get("camera", -1)
            live = _parse_bool(config.get("live", True))
        else:
            num_episodes = 3
            if params.strip():
                num_episodes = int(params)
            deterministic = False
            fps = 30
            fps_mode = "compensate_stride"
            prefix = f"record_{self.n_total_timesteps}"
            video_folder = None
            max_parallel_episodes = min(num_episodes, 10)
            frame_stride = 1
            width = DEFAULT_RECORDING_WIDTH
            height = DEFAULT_RECORDING_HEIGHT
            camera = -1
            live = True

        if video_folder is None:
            if self._active_run_dir is not None:
                folder = self._active_run_dir / "videos"
            else:
                folder = Path("videos")
        else:
            folder = Path(str(video_folder))

        live_record_fn = getattr(self.env.unwrapped, "start_video_recording", None)
        if live and callable(live_record_fn):
            if deterministic:
                logger.warning("Ignoring deterministic=... for live MJW recording; episodes come from the current training rollout.")
            live_record_fn(
                video_folder=str(folder),
                video_name_prefix=prefix,
                num_episodes=num_episodes,
                max_parallel_episodes=max_parallel_episodes,
                fps=fps,
                fps_mode=fps_mode,
                frame_stride=frame_stride,
                width=width,
                height=height,
                camera=camera,
            )
            logger.warning(
                f"Started live recording of {num_episodes} episode(s) to {folder.as_posix()} "
                f"(parallel={max_parallel_episodes}, frame_stride={frame_stride}, fps={fps}, "
                f"fps_mode={fps_mode}, resolution={width}x{height})"
            )
            return

        if self._make_record_env is None:
            raise ValueError(
                "record is unavailable: env has no live recording support and make_record_env is None"
            )

        requested_device = getattr(self, "record_device", getattr(self, "rollout_device", torch.device("cpu")))
        policy_device = _infer_module_device(self.policy)
        if policy_device is not None and requested_device != policy_device and _policy_uses_compiled_modules(self.policy):
            logger.warning(
                "Overriding record_device from "
                f"{requested_device} to {policy_device} because compiled policy recording across devices "
                "would trigger recompiles or backend compile failures."
            )
            device = policy_device
        else:
            device = requested_device
        gsde_reset_mode = getattr(self, "gsde_reset_mode", None)
        record_env: BaseLearnEnvWrapper | None = None

        try:
            record_env = self._make_record_env()
            _set_record_env_render_resolution(record_env, width=width, height=height)
            move_env_to_device(record_env, device)
            apply_env_state(record_env, capture_env_state(self.env))
            freeze_env_normalization(record_env)
            first_frame = record_env.render()
            if first_frame is None:
                raise RuntimeError(
                    "Recording env render() returned None. Ensure make_record_env creates SwarmBotsEnv with "
                    "render_mode='rgb_array'."
                )

            logger.warning(
                f"Recording {num_episodes} episode(s) to {folder.as_posix()} "
                f"(deterministic={deterministic}, fps={fps}, resolution={width}x{height})"
            )
            record_policy(
                env=record_env,
                policy=self.policy,
                video_folder=str(folder),
                video_name_prefix=prefix,
                num_episodes=num_episodes,
                deterministic=deterministic,
                gsde_reset_mode=gsde_reset_mode,
                fps=fps,
                device=device,
            )
        finally:
            if record_env is not None:
                record_env.close()

    def _cmd_record_status(self, params: str) -> None:
        _ = params
        get_status_fn = getattr(self.env.unwrapped, "get_video_recording_status", None)
        if callable(get_status_fn):
            logger.info({"recording": get_status_fn(), "recording_mode": "live_env"})
            return

        if self._make_record_env is not None:
            logger.info(
                {
                    "recording": {
                        "active": False,
                        "message": "This env uses the legacy separate record-env flow. No live recording session exists.",
                    },
                    "recording_mode": "separate_record_env",
                }
            )
            return

        logger.info(
            {
                "recording": {
                    "active": False,
                    "message": "Recording is not configured for this run.",
                },
                "recording_mode": "unavailable",
            }
        )

    def _cmd_show_reward_weights(self, params: str) -> None:
        _ = params
        results = self.env.unwrapped.call("get_reward_weights")

        if not isinstance(results, list):
            logger.info({"reward_weights": results})
            return

        first = results[0] if results else None
        logger.info(first)

    def _cmd_show_eta(self, params: str) -> None:
        _ = params
        eta = self._build_run_eta_estimate()
        if eta is None:
            logger.info({
                "estimated_finish_at": None,
                "remaining_duration": None,
                "remaining_timesteps": None,
                "observed_throughput_tps": None,
                "reason": "Need at least one completed iteration in the current learn() call.",
            })
            return

        logger.info(eta)

    def _cmd_set_reward_weights(self, params: str, extra_run_metadata: dict[str, Any] | None) -> None:
        config = _parse_params_maybe_json(params)
        if not isinstance(config, dict):
            raise ValueError("set_reward_weights expects a JSON object, e.g. set_reward_weights:{\"progress_reward_weight\":1.0}")

        reward_weights = config.get("reward_weights", config)
        if not isinstance(reward_weights, dict):
            raise ValueError(
                "set_reward_weights expects either a JSON object of weights or "
                "{\"reward_weights\": {...}}"
            )

        results = self.env.unwrapped.call("update_reward_weights", reward_weights)

        if not isinstance(results, tuple):
            logger.warning(f"Updated reward weights with unexpected results: {results} for {reward_weights}")
            return

        failed: list[tuple[int, dict[str, Any]]] = []
        for i, res in enumerate(results):
            if isinstance(res, dict) and res.get("ok", True) is False:
                failed.append((i, res))

        if not failed:
            logger.warning(f"Updated reward weights in {len(results)} env(s): {reward_weights}")
            if extra_run_metadata is not None:
                extra_run_metadata['updated_reward_weights'] = results[0]
            return

        logger.error(f"Reward weights update failed in {len(failed)}/{len(results)} env(s): {reward_weights}")
        for i, res in failed:
            logger.error(f"env[{i}] update failed: {res}")

    def _build_run_eta_estimate(
            self,
            current_time: datetime | None = None,
            elapsed_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        max_total_timesteps = self._active_max_total_timesteps
        if max_total_timesteps is None:
            return None

        learn_started_timesteps = self._active_learn_started_timesteps
        learn_started_monotonic = self._active_learn_started_monotonic
        if learn_started_timesteps is None or learn_started_monotonic is None:
            return None

        completed_timesteps = self.n_total_timesteps - learn_started_timesteps
        if completed_timesteps <= 0:
            return None

        elapsed_seconds = time.monotonic() - learn_started_monotonic if elapsed_seconds is None else elapsed_seconds
        if elapsed_seconds <= 0.0:
            return None

        current_time = datetime.now().astimezone() if current_time is None else current_time
        remaining_timesteps = max(0, max_total_timesteps - self.n_total_timesteps)
        observed_throughput_tps = completed_timesteps / elapsed_seconds
        remaining_duration_seconds = remaining_timesteps / observed_throughput_tps
        estimated_finish_at = current_time + timedelta(seconds=remaining_duration_seconds)
        return {
            "estimated_finish_at": estimated_finish_at.isoformat(timespec="seconds"),
            "remaining_duration": _format_duration_seconds(remaining_duration_seconds),
            "remaining_timesteps": remaining_timesteps,
            "observed_throughput_tps": round(observed_throughput_tps, 3),
        }


def _parse_step_from_metadata_filename(path: Path) -> int | None:
    if path.name == "run_metadata.json":
        return None
    if not (path.name.startswith("run_metadata_") and path.name.endswith(".json")):
        return None

    suffix = path.name.removeprefix("run_metadata_").removesuffix(".json")
    step_str = suffix.split("_", 1)[0]
    if not step_str.isdigit():
        return None
    return int(step_str)


def _infer_module_device(module: torch.nn.Module) -> torch.device | None:
    try:
        first_parameter = next(module.parameters())
    except StopIteration:
        return None
    return first_parameter.device


def _policy_uses_compiled_modules(policy: BasePolicy) -> bool:
    config = getattr(policy, "config", None)
    if config is not None and bool(getattr(config, "compile_modules", False)):
        return True

    inner_policy = getattr(policy, "policy", None)
    if inner_policy is not None and _policy_uses_compiled_modules(inner_policy):
        return True

    return bool(getattr(policy, "_wm_compile_modules", False))


def _set_record_env_render_resolution(record_env: BaseLearnEnvWrapper, *, width: int, height: int) -> None:
    set_resolution = getattr(record_env, "call", None)
    if callable(set_resolution):
        set_resolution("set_render_resolution", width=int(width), height=int(height))


def _read_metadata_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


_RUN_METADATA_VOLATILE_KEYS: frozenset[str] = frozenset({
    "iterations",
    "updates",
    "timesteps",
    "load_path",
    "script",
})


def _normalize_run_metadata_for_comparison(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    return {k: v for k, v in metadata.items() if k not in _RUN_METADATA_VOLATILE_KEYS}


_DURATION_PART_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)(?P<unit>[dhms])", re.IGNORECASE)


def _parse_duration_to_seconds(params: str) -> float:
    s = params.strip().lower().replace(" ", "")
    if not s:
        raise ValueError("Missing duration, e.g. pause:90s or pause:1h30m10")

    total_seconds = 0.0
    pos = 0
    for match in _DURATION_PART_RE.finditer(s):
        if match.start() != pos:
            break
        value = float(match.group("value"))
        unit = match.group("unit")
        factor = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        total_seconds += value * factor
        pos = match.end()

    trailing = s[pos:]
    if trailing:
        if total_seconds == 0.0 or not trailing.replace(".", "", 1).isdigit():
            raise ValueError(
                f"Invalid duration {params!r}. Expected formats like 90s, 10m, 1h30m, or 1h30m10"
            )
        total_seconds += float(trailing)

    if total_seconds <= 0.0:
        raise ValueError("Duration must be > 0")
    return total_seconds


def _format_duration_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return "".join(parts)


def _start_command_prompt(cmd_q: queue.Queue[str]) -> bool:
    use_prompt_toolkit = True
    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.patch_stdout import patch_stdout as pt_patch_stdout
    except ImportError:
        logger.warning(
            "Command prompt disabled (prompt_toolkit not installed). "
            "Install it with `pip install prompt_toolkit`."
        )
        use_prompt_toolkit = False

    if not sys.stdin or not sys.stdin.isatty():
        use_prompt_toolkit = False
        logger.warning("Command prompt may not work (stdin is not a TTY).")

    def _run() -> None:
        while True:
            try:
                if use_prompt_toolkit:
                    with pt_patch_stdout():
                        cmd = pt_prompt("> ")
                else:
                    cmd = input("> ")
            except (EOFError, KeyboardInterrupt):
                logger.debug("EOF: Exiting command prompt")
                return
            cmd_q.put(cmd.strip())

    threading.Thread(target=_run, daemon=True).start()
    return True

def _parse_params_maybe_json(params: str) -> Any:
    s = params.strip()
    if not s:
        return None
    if not (s.startswith("{") or s.startswith("[")):
        return None
    return json.loads(s)


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if v is None:
        raise ValueError("Expected boolean value, got None")
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from {v!r}")
