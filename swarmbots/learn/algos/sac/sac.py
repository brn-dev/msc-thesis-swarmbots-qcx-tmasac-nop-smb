from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.action_sampling import (
    ActionSampleStrategy,
    expand_action_samples,
    validate_action_sampling,
)
from swarmbots.learn.algos.base_algorithm import (
    BaseAlgorithm,
    LearningRate,
    PolicyStateDictTransform,
)
from swarmbots.learn.algos.off_policy import (
    OffPolicyReplayBuffer,
    OffPolicyRolloutState,
    collect_off_policy_steps,
    warmup_off_policy_steps,
)
from swarmbots.learn.algos.off_policy.replay_buffer import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
from swarmbots.learn.algos.sac.sac_tensor_ops import (
    build_optimizer_step,
    build_sac_tensor_operations,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEIntervalResetMode,
    GSDEProbabilityResetMode,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.metrics_logger import SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.torch_device import as_device


DEFAULT_TOTAL_REPLAY_CAPACITY = 1_000_000
ENTROPY_AGENT_REDUCTION = "mean"
TrainMetric = float | torch.Tensor
TrainStepResult = tuple[dict[str, TrainMetric], TrainMetric, TrainMetric]


class SAC(BaseAlgorithm):
    """SAC with independently configurable actor and Bellman action sample counts.

    Stratification applies to bounded quantile distributions; other distributions
    keep their iid samplers. Counts of one preserve the ordinary training path.
    """

    policy: BaseSACPolicy
    learning_rate: float
    supports_recurrent_training = False

    def __init__(
            self,
            policy: BaseSACPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: float = 3e-4,
            learning_rate_warmup_updates: int = 500,
            learning_rate_warmup_start_factor: float = 0.01,
            buffer_capacity_per_env: int | None = None,
            learning_starts: int = 10_000,
            batch_size: int = 256,
            rollout_steps_per_iteration: int | None = None,
            rollout_warmup_steps_per_env: int = 0,
            gradient_steps: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            ent_coef: float | str = "auto",
            ent_coef_learning_rate: float | None = None,
            target_entropy: float | str = "auto",
            target_update_interval: int = 1,
            max_grad_norm: float | None = 2.0,
            nop_batch_size: int | None = None,
            independent_nop_sampling: bool = False,
            gsde_reset_mode: GSDEResetMode | None = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            record_device: str | torch.device | None = None,
            replay_storage_device: str | torch.device = "cuda",
            replay_storage_pin_memory: bool = False,
            replay_compile_tensor_operations: bool | None = None,
            sac_compile_tensor_operations: bool | None = None,
            sac_compile_optimizer_steps: bool | None = None,
            sac_compile_mode: str = "default",
            metrics_action_splitters: ActionMetricsSplitterInput = None,
            actor_action_samples: int = 1,
            target_action_samples: int = 1,
            action_sample_strategy: ActionSampleStrategy = "iid",
    ) -> None:
        if not isinstance(learning_rate, float):
            raise TypeError(f"learning_rate must be a float, got {type(learning_rate).__name__}")
        if policy.requires_recurrent_training() and not self.supports_recurrent_training:
            raise TypeError(
                f"{type(policy).__name__} requires a recurrent SAC algorithm; "
                f"plain {type(self).__name__} samples isolated replay transitions."
            )
        super().__init__(policy, env, learning_rate)

        self.learning_rate = learning_rate
        self.learning_rate_warmup_updates = int(learning_rate_warmup_updates)
        self.learning_rate_warmup_start_factor = float(learning_rate_warmup_start_factor)
        self.buffer_capacity_per_env = (
            max(1, DEFAULT_TOTAL_REPLAY_CAPACITY // env.action_space.n_envs)
            if buffer_capacity_per_env is None
            else int(buffer_capacity_per_env)
        )
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.rollout_steps_per_iteration = (
            env.action_space.n_envs if rollout_steps_per_iteration is None else int(rollout_steps_per_iteration)
        )
        self.rollout_warmup_steps_per_env = int(rollout_warmup_steps_per_env)
        self._rollout_warmup_done = self.rollout_warmup_steps_per_env == 0
        self.gradient_steps = int(gradient_steps)
        self.actor_action_samples = actor_action_samples
        self.target_action_samples = target_action_samples
        self.action_sample_strategy = action_sample_strategy
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.ent_coef = ent_coef
        self.ent_coef_learning_rate = (
            None if ent_coef_learning_rate is None else float(ent_coef_learning_rate)
        )
        self.target_entropy = target_entropy
        self.target_update_interval = int(target_update_interval)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.nop_batch_size = self.batch_size if nop_batch_size is None else int(nop_batch_size)
        self.independent_nop_sampling = bool(independent_nop_sampling)
        self.gsde_reset_mode = gsde_reset_mode
        self.train_device = as_device(train_device)
        self.rollout_device = as_device(rollout_device)
        self.record_device = self.rollout_device if record_device is None else as_device(record_device)
        self.replay_storage_device = as_device(replay_storage_device)
        self.replay_storage_pin_memory = bool(replay_storage_pin_memory)
        self.replay_compile_tensor_operations = replay_compile_tensor_operations
        self.sac_compile_tensor_operations = (
            self.train_device.type == "cuda"
            if sac_compile_tensor_operations is None
            else bool(sac_compile_tensor_operations)
        )
        self.sac_compile_optimizer_steps = (
            self.train_device.type == "cuda"
            if sac_compile_optimizer_steps is None
            else bool(sac_compile_optimizer_steps)
        )
        self.sac_compile_mode = sac_compile_mode
        self._tensor_operations = build_sac_tensor_operations(
            compile_operations=self.sac_compile_tensor_operations,
            compile_mode=self.sac_compile_mode,
        )
        self.metrics_action_splitters = metrics_action_splitters
        self.agent_action_dim = int(env.action_space.total_agent_action_dim)
        self._rollout_state: OffPolicyRolloutState | None = None

        self._validate_hyper_parameters()
        self.replay_buffer = self._build_replay_buffer()

        self.policy.to(self.train_device)
        initial_actor_critic_lr = self._actor_critic_learning_rate_for_update(self.n_total_updates)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=initial_actor_critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(), lr=initial_actor_critic_lr)
        self.log_ent_coef: torch.Tensor | None = None
        self.ent_coef_tensor: torch.Tensor | None = None
        self.ent_coef_optimizer: torch.optim.Optimizer | None = None
        self._setup_entropy_coefficient()
        self._rebuild_optimizer_steps()
        self._policy_num_params = self.policy.num_parameters(learnable_only=False)
        self._policy_num_trainable_params = self.policy.num_parameters()

    def _build_replay_buffer(self) -> OffPolicyReplayBuffer:
        return OffPolicyReplayBuffer(
            capacity_per_env=self.buffer_capacity_per_env,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            store_previous_actions=self.policy.requires_previous_actions(),
            storage_device=self.replay_storage_device,
            storage_dtype=torch.float32,
            storage_pin_memory=self.replay_storage_pin_memory,
            train_device=self.train_device,
            train_dtype=torch.float32,
            compile_tensor_operations=self.replay_compile_tensor_operations,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "learning_rate_warmup_updates": self.learning_rate_warmup_updates,
            "learning_rate_warmup_start_factor": self.learning_rate_warmup_start_factor,
            "buffer_capacity_per_env": self.buffer_capacity_per_env,
            "learning_starts": self.learning_starts,
            "replay_fill_target": self.replay_fill_target,
            "batch_size": self.batch_size,
            "rollout_steps_per_iteration": self.rollout_steps_per_iteration,
            "rollout_warmup_steps_per_env": self.rollout_warmup_steps_per_env,
            "gradient_steps": self.gradient_steps,
            "actor_action_samples": self.actor_action_samples,
            "target_action_samples": self.target_action_samples,
            "action_sample_strategy": self.action_sample_strategy,
            "gamma": self.gamma,
            "tau": self.tau,
            "ent_coef": self.ent_coef,
            "ent_coef_learning_rate": self.ent_coef_learning_rate,
            "resolved_ent_coef_learning_rate": self._resolved_ent_coef_learning_rate(),
            "target_entropy": self.target_entropy,
            "entropy_agent_reduction": ENTROPY_AGENT_REDUCTION,
            "target_update_interval": self.target_update_interval,
            "max_grad_norm": self.max_grad_norm,
            "nop_batch_size": self.nop_batch_size,
            "independent_nop_sampling": self.independent_nop_sampling,
            "gsde_reset_mode": self._serialize_gsde_reset_mode(self.gsde_reset_mode),
            "train_device": str(self.train_device),
            "rollout_device": str(self.rollout_device),
            "record_device": str(self.record_device),
            "replay_storage_device": str(self.replay_storage_device),
            "replay_storage_pin_memory": self.replay_storage_pin_memory,
            "replay_compile_tensor_operations": self.replay_buffer.compile_tensor_operations,
            "sac_compile_tensor_operations": self.sac_compile_tensor_operations,
            "sac_compile_optimizer_steps": self.sac_compile_optimizer_steps,
            "sac_compile_mode": self.sac_compile_mode,
            "policy_num_params": self._policy_num_params,
            "policy_num_trainable_params": self._policy_num_trainable_params,
        }

    def _before_learn_loop(self) -> None:
        if self._rollout_warmup_done:
            return
        if (
                self.n_total_timesteps > 0
                or self.n_total_iterations > 0
                or self.n_total_updates > 0
                or self._rollout_state is not None
                or len(self.replay_buffer) > 0
        ):
            self._rollout_warmup_done = True
            return

        warmup_transitions = self.rollout_warmup_steps_per_env * self.replay_buffer.n_envs
        logger.info(
            f"Running SAC rollout warmup for {self.rollout_warmup_steps_per_env} vector steps "
            f"({warmup_transitions} transitions) without training or replay writes."
        )
        random_actions = self.n_total_timesteps < self.learning_starts
        with PerformanceTimer() as warmup_timer:
            self._rollout_state = warmup_off_policy_steps(
                env=self.env,
                replay_buffer=self.replay_buffer,
                n_steps=warmup_transitions,
                policy=self.policy,
                rollout_state=None,
                random_actions=random_actions,
                deterministic=False,
                gsde_reset_mode=self.gsde_reset_mode,
                rollout_device=self.rollout_device,
            )
        self._rollout_warmup_done = True
        logger.info(f"Finished SAC rollout warmup in {warmup_timer.get_duration():.2f}s.")

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        random_actions = self.n_total_timesteps < self.learning_starts
        with PerformanceTimer() as rollout_timer:
            episode_infos, rollout_metrics, self._rollout_state = collect_off_policy_steps(
                env=self.env,
                replay_buffer=self.replay_buffer,
                n_steps=self.rollout_steps_per_iteration,
                policy=self.policy,
                rollout_state=self._rollout_state,
                random_actions=random_actions,
                deterministic=False,
                gsde_reset_mode=self.gsde_reset_mode,
                rollout_device=self.rollout_device,
            )

        rollout_steps = int(rollout_metrics.get("transitions_collected", self.rollout_steps_per_iteration))
        rollout_actions = rollout_metrics.pop("_rollout_actions", None)
        self.n_total_timesteps += rollout_steps
        self.n_total_iterations += 1

        if update_ema:
            for ep_info in episode_infos:
                if "r" in ep_info:
                    episode_return_ema.update(ep_info["r"])
                if "success" in ep_info:
                    episode_success_rate_ema.update(float(ep_info["success"]))

        train_metrics: dict[str, Any]
        if self._should_train():
            train_metrics = self.train(gradient_steps=self._resolved_gradient_steps(rollout_steps))
        else:
            train_metrics = {
                "updates": 0,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "random_actions": random_actions,
                "training_skipped": True,
                SUPPRESS_MISSING_CONSOLE_KEY_WARNINGS: True,
            }

        metrics = {
            **train_metrics,
            **rollout_metrics,
            **self._episode_metrics(episode_infos),
            "rollout_time": rollout_timer.get_duration(),
            "random_actions": random_actions,
            "replay_size": len(self.replay_buffer),
        }
        if rollout_actions is not None:
            metrics.update(self._compute_action_metrics(rollout_actions, prefix="rollout"))
        return metrics, rollout_steps

    def train(self, *, gradient_steps: int) -> dict[str, Any]:
        if gradient_steps <= 0:
            return {"updates": 0, "total_updates": self.n_total_updates}

        with PerformanceTimer() as to_train_device_timer:
            self.policy.train()
            self.policy.to(self.train_device)
            self._move_entropy_tensors_to_train_device()

        loss_metrics = MetricsLists[float]()
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        pending_step_results: list[TrainStepResult] = []
        update_timings: list[float] = []
        update_cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        use_cuda_update_timing = self.train_device.type == "cuda"
        sample_timings: list[float] = []
        sample_timer = PerformanceTimer()
        update_timer = PerformanceTimer()
        train_timer = PerformanceTimer().start()

        n_updates = 0
        for _ in range(gradient_steps):
            with sample_timer:
                batch, nop_batch, reuse_critic_nop_latents = self._sample_training_batches()
            sample_timings.append(sample_timer.get_duration())

            global_update_idx = self.n_total_updates + n_updates
            if use_cuda_update_timing:
                update_start_event = torch.cuda.Event(enable_timing=True)
                update_end_event = torch.cuda.Event(enable_timing=True)
                update_start_event.record(torch.cuda.current_stream(self.train_device))
            with update_timer:
                step_metrics, actor_grad_norm, critic_grad_norm = self._train_step(
                    batch,
                    nop_batch=nop_batch,
                    reuse_critic_nop_latents=reuse_critic_nop_latents,
                    global_update_idx=global_update_idx,
                    materialize_metrics=False,
                )
            if use_cuda_update_timing:
                update_end_event.record(torch.cuda.current_stream(self.train_device))
                update_cuda_events.append((update_start_event, update_end_event))
            else:
                update_timings.append(update_timer.get_duration())
            pending_step_results.append((step_metrics, actor_grad_norm, critic_grad_norm))
            n_updates += 1

        if update_cuda_events:
            update_cuda_events[-1][1].synchronize()
            update_timings.extend(
                start_event.elapsed_time(end_event) / 1_000.0
                for start_event, end_event in update_cuda_events
            )
        train_timer.stop()
        self.n_total_updates += n_updates

        with PerformanceTimer() as metrics_timer:
            for step_metrics, actor_grad_norm, critic_grad_norm in self._materialize_train_step_results(
                pending_step_results
            ):
                loss_metrics.add(step_metrics)
                actor_grad_norms.append(actor_grad_norm)
                critic_grad_norms.append(critic_grad_norm)
            metrics: dict[str, Any] = {
                **loss_metrics.compute_summary_statistics(),
                "updates": n_updates,
                "total_updates": self.n_total_updates,
                "replay_size": len(self.replay_buffer),
                "actor_grad_norm": compute_summary_statistics(actor_grad_norms, find_max=True, find_min=True),
                "critic_grad_norm": compute_summary_statistics(critic_grad_norms, find_max=True, find_min=True),
            }
            sampled_actions = batch.actions
            if batch.agent_mask is not None:
                sampled_actions = sampled_actions[batch.agent_mask]
            metrics.update(self._compute_action_metrics(sampled_actions, prefix="replay"))

        return {
            **metrics,
            "to_train_device_time": to_train_device_timer.get_duration(),
            "sampling_time": compute_summary_statistics(sample_timings),
            "total_sampling_time": sum(sample_timings),
            "update_time": compute_summary_statistics(update_timings),
            "total_update_time": sum(update_timings),
            "metrics_time": metrics_timer.get_duration(),
            "train_time": train_timer.get_duration(),
        }

    def _compute_action_metrics(self, actions: torch.Tensor, *, prefix: str) -> dict[str, Any]:
        return {
            f"{prefix}_{name}": value
            for name, value in self.policy.action_dist.get_metrics(
                actions,
                action_splitter=self.metrics_action_splitters,
            ).items()
        }

    def _train_step(
            self,
            batch: OffPolicyReplayBatch,
            *,
            nop_batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | None = None,
            reuse_critic_nop_latents: bool = False,
            global_update_idx: int,
            materialize_metrics: bool = True,
    ) -> TrainStepResult:
        self._mark_cuda_graph_train_step_begin()
        actor_critic_lr = self._apply_actor_critic_learning_rate_for_update(global_update_idx)
        nop_loss_batch = batch if nop_batch is None else nop_batch
        skip_multi_step_nop_loss = self._uses_multi_step_nop() and nop_batch is None
        self._reset_train_gsde_noise(batch.local_obs)
        actions_pi, log_prob_pi = self.policy.action_log_prob(
            num_action_samples=self.actor_action_samples,
            action_sample_strategy=self.action_sample_strategy,
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            previous_actions=batch.previous_actions,
            deterministic=False,
        )
        actor_action_dist_losses, actor_action_dist_metrics = self._compute_actor_action_dist_extra_losses(
            agent_mask=batch.agent_mask,
        )
        reduced_actor_action_dist_losses = self._reduce_actor_action_dist_extra_losses(
            batch=batch,
            extra_losses=actor_action_dist_losses,
        )
        log_prob_pi_mean = self._mean_agent_log_probs(log_prob_pi, batch.agent_mask)
        ent_coef, ent_coef_loss = self._update_entropy_coefficient(
            log_prob_mean=self._mean_action_samples(log_prob_pi_mean, self.actor_action_samples),
            batch=batch,
        )
        target_entropy = self._target_entropy(
            batch=batch,
            dtype=log_prob_pi_mean.dtype,
            device=log_prob_pi_mean.device,
        )

        with torch.no_grad():
            (
                bootstrap_next_local_obs,
                bootstrap_next_global_obs,
                bootstrap_next_hidden_local_vars,
                bootstrap_next_hidden_global_vars,
                bootstrap_next_agent_mask,
            ) = self._tensor_operations.mask_terminal_observations(
                batch.terminal_mask,
                batch.next_local_obs,
                batch.next_global_obs,
                batch.next_hidden_local_vars,
                batch.next_hidden_global_vars,
                batch.next_agent_mask,
            )
            self._reset_train_gsde_noise(bootstrap_next_local_obs)
            target_q = self._target_forward_phase(
                batch,
                bootstrap_next_local_obs,
                bootstrap_next_global_obs,
                bootstrap_next_hidden_local_vars,
                bootstrap_next_hidden_global_vars,
                bootstrap_next_agent_mask,
                ent_coef,
            )

        current_q1, current_q2, critic_nop_latents, critic_loss = self._critic_forward_phase(
            batch,
            target_q,
        )
        if skip_multi_step_nop_loss:
            critic_nop_loss, critic_nop_metrics = None, {"nop_loss_skipped": 1.0}
        elif reuse_critic_nop_latents and critic_nop_latents is not None:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss(
                nop_loss_batch,
                source_latents=critic_nop_latents,
            )
        else:
            critic_nop_loss, critic_nop_metrics = self.policy.compute_critic_nop_loss(nop_loss_batch)
        critic_total_loss = critic_loss if critic_nop_loss is None else critic_loss + critic_nop_loss

        self.critic_optimizer.zero_grad()
        critic_total_loss.backward()
        critic_grad_norm = self._clip_grad_norm(self.policy.critic_parameters())
        self._step_actor_or_critic_optimizer(
            optimizer=self.critic_optimizer,
            compiled_step=self._critic_optimizer_step,
            global_update_idx=global_update_idx,
        )

        critic_parameters = self.policy.critic_parameters()
        self._set_requires_grad(critic_parameters, False)
        try:
            q1_pi, q2_pi, actor_loss = self._actor_forward_phase(
                batch,
                actions_pi,
                log_prob_pi_mean,
                ent_coef,
            )
        finally:
            self._set_requires_grad(critic_parameters, True)

        if skip_multi_step_nop_loss:
            actor_nop_loss, actor_nop_metrics = None, {}
        else:
            actor_nop_loss, actor_nop_metrics = self.policy.compute_actor_nop_loss(nop_loss_batch)
        actor_total_loss = actor_loss if actor_nop_loss is None else actor_loss + actor_nop_loss
        if reduced_actor_action_dist_losses:
            actor_total_loss = actor_total_loss + torch.stack(
                tuple(reduced_actor_action_dist_losses.values())
            ).sum()

        self.actor_optimizer.zero_grad()
        actor_total_loss.backward()
        actor_grad_norm = self._clip_grad_norm(self.policy.actor_parameters())
        self._step_actor_or_critic_optimizer(
            optimizer=self.actor_optimizer,
            compiled_step=self._actor_optimizer_step,
            global_update_idx=global_update_idx,
        )

        if global_update_idx % self.target_update_interval == 0:
            self.policy.polyak_update_targets(self.tau)
        self.policy.after_optimizer_step()

        metrics = {
            "critic_loss": critic_loss.detach(),
            "critic_total_loss": critic_total_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "actor_total_loss": actor_total_loss.detach(),
            "target_q": target_q.mean().detach(),
            "current_q1": current_q1.mean().detach(),
            "current_q2": current_q2.mean().detach(),
            "q_pi": torch.minimum(q1_pi, q2_pi).mean().detach(),
            "log_prob": log_prob_pi_mean.mean().detach(),
            "entropy": (-log_prob_pi_mean).mean().detach(),
            "target_entropy": target_entropy.mean().detach(),
            "ent_coef": ent_coef.detach(),
            "actor_critic_learning_rate": actor_critic_lr,
        }
        if ent_coef_loss is not None:
            metrics["ent_coef_loss"] = ent_coef_loss.detach()
            metrics["ent_coef_learning_rate"] = self._resolved_ent_coef_learning_rate()
        metrics.update({
            f"actor_action_dist_{name}_loss_scaled": value.detach()
            for name, value in reduced_actor_action_dist_losses.items()
        })
        metrics.update({
            f"actor_action_dist_{name}": value
            for name, value in actor_action_dist_metrics.items()
        })
        metrics.update(actor_nop_metrics)
        metrics.update(critic_nop_metrics)
        result = (metrics, actor_grad_norm, critic_grad_norm)
        if not materialize_metrics:
            return self._preserve_train_step_result(result)
        return self._materialize_train_step_results([result])[0]

    def _mark_cuda_graph_train_step_begin(self) -> None:
        if self.train_device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()

    @staticmethod
    def _preserve_train_step_result(result: TrainStepResult) -> TrainStepResult:
        metrics, actor_grad_norm, critic_grad_norm = result

        def preserve(value: TrainMetric) -> TrainMetric:
            if not torch.is_tensor(value):
                return value
            return value.detach().clone()

        return (
            {name: preserve(value) for name, value in metrics.items()},
            preserve(actor_grad_norm),
            preserve(critic_grad_norm),
        )

    @staticmethod
    def _materialize_train_step_results(results: list[TrainStepResult]) -> list[TrainStepResult]:
        tensor_locations: list[tuple[int, str | None]] = []
        tensor_values: list[torch.Tensor] = []
        for result_idx, (metrics, actor_grad_norm, critic_grad_norm) in enumerate(results):
            for name, value in metrics.items():
                if torch.is_tensor(value):
                    tensor_locations.append((result_idx, name))
                    tensor_values.append(value.detach().reshape(()))
            for name, value in (("__actor_grad_norm", actor_grad_norm), ("__critic_grad_norm", critic_grad_norm)):
                if torch.is_tensor(value):
                    tensor_locations.append((result_idx, name))
                    tensor_values.append(value.detach().reshape(()))
        if not tensor_values:
            return results

        materialized_values = torch.stack(tensor_values).cpu().tolist()
        mutable_results = [
            [dict(metrics), actor_grad_norm, critic_grad_norm]
            for metrics, actor_grad_norm, critic_grad_norm in results
        ]
        for (result_idx, name), value in zip(tensor_locations, materialized_values, strict=True):
            if name == "__actor_grad_norm":
                mutable_results[result_idx][1] = value
            elif name == "__critic_grad_norm":
                mutable_results[result_idx][2] = value
            else:
                assert name is not None
                mutable_results[result_idx][0][name] = value
        return [
            (metrics, actor_grad_norm, critic_grad_norm)
            for metrics, actor_grad_norm, critic_grad_norm in mutable_results
        ]

    def _target_forward_phase(
            self,
            batch: OffPolicyReplayBatch,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            next_hidden_local_vars: torch.Tensor,
            next_hidden_global_vars: torch.Tensor,
            next_agent_mask: torch.Tensor | None,
            ent_coef: torch.Tensor,
    ) -> torch.Tensor:
        next_actions, next_log_probs = self.policy.action_log_prob(
            num_action_samples=self.target_action_samples,
            action_sample_strategy=self.action_sample_strategy,
            local_obs=next_local_obs,
            global_obs=next_global_obs,
            hidden_local_vars=next_hidden_local_vars,
            hidden_global_vars=next_hidden_global_vars,
            agent_mask=next_agent_mask,
            **(
                {}
                if batch.next_scenario_ids is None
                else {"scenario_ids": batch.next_scenario_ids}
            ),
            previous_actions=batch.actions,
            deterministic=False,
            use_rsample=False,
        )
        next_log_prob_mean = self._mean_agent_log_probs(next_log_probs, next_agent_mask)
        target_q1, target_q2 = self.policy.q_values_samples(
            num_action_samples=self.target_action_samples,
            target=True,
            local_obs=next_local_obs,
            global_obs=next_global_obs,
            hidden_local_vars=next_hidden_local_vars,
            hidden_global_vars=next_hidden_global_vars,
            agent_mask=next_agent_mask,
            **(
                {}
                if batch.next_scenario_ids is None
                else {"scenario_ids": batch.next_scenario_ids}
            ),
            actions=next_actions,
        )
        target_q = self._tensor_operations.bellman_target(
            batch.rewards,
            batch.terminal_mask,
            target_q1,
            target_q2,
            next_log_prob_mean,
            ent_coef,
            self.gamma,
        )
        return self._mean_action_samples(target_q, self.target_action_samples)

    def _critic_forward_phase(
            self,
            batch: OffPolicyReplayBatch,
            target_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        current_q1, current_q2, critic_nop_latents = self.policy.q_values_with_nop_latents(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            actions=batch.actions,
        )
        critic_loss = self._tensor_operations.critic_loss(current_q1, current_q2, target_q)
        return current_q1, current_q2, critic_nop_latents, critic_loss

    def _actor_forward_phase(
            self,
            batch: OffPolicyReplayBatch,
            actions: torch.Tensor,
            log_prob_mean: torch.Tensor,
            ent_coef: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1, q2 = self.policy.q_values_samples(
            num_action_samples=self.actor_action_samples,
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            **({} if batch.scenario_ids is None else {"scenario_ids": batch.scenario_ids}),
            actions=actions,
        )
        actor_loss = self._tensor_operations.actor_loss(q1, q2, log_prob_mean, ent_coef)
        return q1, q2, actor_loss

    @staticmethod
    def _mean_action_samples(value: torch.Tensor, count: int) -> torch.Tensor:
        return value.mean(dim=0) if count > 1 and value.ndim > 0 else value

    def _setup_entropy_coefficient(self) -> None:
        if isinstance(self.ent_coef, str):
            init_value = self._parse_auto_value(
                self.ent_coef,
                parameter_name="ent_coef",
                suffix_name="initial_value",
                default=1.0,
            )
            if init_value <= 0:
                raise ValueError(f"Initial entropy coefficient must be > 0, got {init_value}")
            self.log_ent_coef = torch.log(
                torch.ones((), device=self.train_device, dtype=torch.float32) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef],
                lr=self._resolved_ent_coef_learning_rate(),
            )
            return

        ent_coef = float(self.ent_coef)
        if ent_coef < 0:
            raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
        self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)

    def _update_entropy_coefficient(
            self,
            *,
            log_prob_mean: torch.Tensor,
            batch: OffPolicyReplayBatch,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.log_ent_coef is None:
            assert self.ent_coef_tensor is not None
            return self.ent_coef_tensor, None

        target_entropy = self._target_entropy(batch=batch, dtype=log_prob_mean.dtype, device=log_prob_mean.device)
        ent_coef_loss = self._tensor_operations.entropy_coefficient_loss(
            self.log_ent_coef,
            log_prob_mean,
            target_entropy,
        )
        assert self.ent_coef_optimizer is not None
        self.ent_coef_optimizer.zero_grad()
        ent_coef_loss.backward()
        assert self._ent_coef_optimizer_step is not None
        self._ent_coef_optimizer_step()
        return self.log_ent_coef.detach().exp(), ent_coef_loss.detach()

    def _target_entropy(
            self,
            *,
            batch: OffPolicyReplayBatch,
            dtype: torch.dtype,
            device: torch.device,
    ) -> torch.Tensor:
        if isinstance(self.target_entropy, str):
            auto_scale = self._target_entropy_auto_scale()
            target_entropy_per_agent = -auto_scale * float(self.agent_action_dim)
            return torch.full(batch.rewards.shape, target_entropy_per_agent, dtype=dtype, device=device)

        return torch.full(batch.rewards.shape, float(self.target_entropy), dtype=dtype, device=device)

    def _target_entropy_auto_scale(self) -> float:
        assert isinstance(self.target_entropy, str)
        scale = self._parse_auto_value(
            self.target_entropy,
            parameter_name="target_entropy",
            suffix_name="scale",
            default=1.0,
        )
        if scale <= 0.0:
            raise ValueError(f"target_entropy auto scale must be > 0, got {scale}")
        return scale

    @staticmethod
    def _parse_auto_value(
            value: str,
            *,
            parameter_name: str,
            suffix_name: str,
            default: float,
    ) -> float:
        normalized_value = value.lower()
        if normalized_value == "auto":
            return default
        for separator in ("_", "*"):
            prefix = f"auto{separator}"
            if normalized_value.startswith(prefix):
                return float(normalized_value.removeprefix(prefix))
        raise ValueError(
            f"{parameter_name} string must be 'auto', 'auto_{suffix_name}', or 'auto*{suffix_name}'"
        )

    def _mean_agent_log_probs(
            self,
            log_probs: torch.Tensor,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._tensor_operations.mean_agent_log_probs(log_probs, agent_mask)

    def _compute_actor_action_dist_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        action_dist = getattr(self.policy, "action_dist", None)
        if self.actor_action_samples > 1:
            agent_mask = expand_action_samples(agent_mask, self.actor_action_samples)
        compute_extra_losses_without_metrics = getattr(action_dist, "compute_extra_losses_without_metrics", None)
        if callable(compute_extra_losses_without_metrics):
            losses, metrics = compute_extra_losses_without_metrics(agent_mask=agent_mask), {}
        else:
            compute_extra_losses = getattr(action_dist, "compute_extra_losses", None)
            if not callable(compute_extra_losses):
                return {}, {}
            losses, metrics = compute_extra_losses(agent_mask=agent_mask)
        return {
            name: self._mean_action_samples(value, self.actor_action_samples)
            for name, value in losses.items()
        }, metrics

    def _reduce_actor_action_dist_extra_losses(
            self,
            *,
            batch: OffPolicyReplayBatch,
            extra_losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            name: self._reduce_actor_action_dist_extra_loss(batch=batch, value=value)
            for name, value in extra_losses.items()
        }

    def _reduce_actor_action_dist_extra_loss(
            self,
            *,
            batch: OffPolicyReplayBatch,
            value: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim == 0:
            return value
        if value.shape == batch.rewards.shape:
            return value.mean()

        expected_agent_shape = tuple(batch.local_obs.shape[:2])
        if tuple(value.shape) != expected_agent_shape:
            raise ValueError(
                f"Expected SAC actor action-dist extra loss shape {expected_agent_shape} or "
                f"{tuple(batch.rewards.shape)}, got {tuple(value.shape)}"
            )
        if batch.agent_mask is None:
            return value.sum(dim=1).mean()
        return (value * batch.agent_mask.to(dtype=value.dtype)).sum(dim=1).mean()

    def _clip_grad_norm(self, parameters: list[torch.nn.Parameter]) -> TrainMetric:
        if self.max_grad_norm is None:
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if not gradients:
                return 0.0
            return torch.nn.utils.get_total_norm(gradients, norm_type=2.0)
        return torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)

    def _apply_actor_critic_learning_rate_for_update(self, update_idx: int) -> float:
        lr = self._actor_critic_learning_rate_for_update(update_idx)

        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        return lr

    def _step_actor_or_critic_optimizer(
            self,
            *,
            optimizer: torch.optim.Optimizer,
            compiled_step: Callable[[], None],
            global_update_idx: int,
    ) -> None:
        if (
                self.sac_compile_optimizer_steps
                and global_update_idx >= self.learning_rate_warmup_updates
        ):
            compiled_step()
            return
        optimizer.step()

    def _rebuild_optimizer_steps(self) -> None:
        self._actor_optimizer_step = build_optimizer_step(
            self.actor_optimizer,
            compile_step=self.sac_compile_optimizer_steps,
            compile_mode=self.sac_compile_mode,
        )
        self._critic_optimizer_step = build_optimizer_step(
            self.critic_optimizer,
            compile_step=self.sac_compile_optimizer_steps,
            compile_mode=self.sac_compile_mode,
        )
        self._ent_coef_optimizer_step = (
            None
            if self.ent_coef_optimizer is None
            else build_optimizer_step(
                self.ent_coef_optimizer,
                compile_step=self.sac_compile_optimizer_steps,
                compile_mode=self.sac_compile_mode,
            )
        )

    def _actor_critic_learning_rate_for_update(self, update_idx: int) -> float:
        if self.learning_rate_warmup_updates <= 0:
            return self.learning_rate
        if update_idx >= self.learning_rate_warmup_updates:
            return self.learning_rate

        progress = max(0.0, float(update_idx) / float(self.learning_rate_warmup_updates))
        factor = self.learning_rate_warmup_start_factor + (1.0 - self.learning_rate_warmup_start_factor) * progress

        lr = self.learning_rate * factor
        logger.warning(f'[Warmup] LR {lr:.3e}')
        return lr

    def _should_train(self) -> bool:
        return len(self.replay_buffer) >= self.replay_fill_target

    @property
    def replay_fill_target(self) -> int:
        return max(self.learning_starts, self.batch_size)

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
        super().load(
            path,
            map_location=map_location,
            recover_best_return_ema=recover_best_return_ema,
            strict_load_state_dict=strict_load_state_dict,
            policy_state_dict_transform=policy_state_dict_transform,
            restore_optimizer_state=restore_optimizer_state,
            restore_env_state=restore_env_state,
        )
        self.replay_buffer.reset()
        self._rollout_state = None
        self._rollout_warmup_done = True
        logger.info(
            "Loaded SAC without replay state; collecting "
            f"{self.replay_fill_target} fresh transitions before training resumes."
        )

    def _sample_training_batches(
            self,
    ) -> tuple[
        OffPolicyReplayBatch,
        OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | None,
        bool,
    ]:
        if not self.policy.has_nop_loss():
            return self.replay_buffer.sample(self.batch_size), None, False
        if self.independent_nop_sampling:
            return (
                self.replay_buffer.sample(self.batch_size),
                self._sample_independent_nop_batch(),
                False,
            )
        nop_batch = self.replay_buffer.sample_episode_windows(
            self.batch_size,
            num_next_steps=self.policy.get_nop_num_next_steps(),
        )
        return nop_batch.origin_batch, nop_batch, True

    def _sample_independent_nop_batch(
            self,
    ) -> OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | None:
        if not self._uses_multi_step_nop():
            return self.replay_buffer.sample(self.nop_batch_size)
        try:
            return self.replay_buffer.sample_episode_segments(
                self.nop_batch_size,
                segment_length=self.policy.get_nop_num_next_steps(),
                require_initial_temporal_state=False,
            )
        except NoEpisodeSegmentCandidatesError:
            return None

    def _uses_multi_step_nop(self) -> bool:
        return self.policy.has_nop_loss() and self.policy.get_nop_num_next_steps() > 1

    def _resolved_gradient_steps(self, rollout_steps: int) -> int:
        if self.gradient_steps == -1:
            return rollout_steps
        return self.gradient_steps

    def _validate_hyper_parameters(self) -> None:
        validate_action_sampling(
            actor_action_samples=self.actor_action_samples,
            target_action_samples=self.target_action_samples,
            action_sample_strategy=self.action_sample_strategy,
        )
        if max(self.actor_action_samples, self.target_action_samples) > 1:
            if self.policy.gsde_enabled:
                raise ValueError("Multi-sample SAC does not support temporally correlated gSDE noise.")
        if self.buffer_capacity_per_env <= 0:
            raise ValueError(f"buffer_capacity_per_env must be > 0, got {self.buffer_capacity_per_env}")
        if self.learning_rate_warmup_updates < 0:
            raise ValueError(
                f"learning_rate_warmup_updates must be >= 0, got {self.learning_rate_warmup_updates}"
            )
        if not (0.0 < self.learning_rate_warmup_start_factor <= 1.0):
            raise ValueError(
                "learning_rate_warmup_start_factor must be in (0, 1], got "
                f"{self.learning_rate_warmup_start_factor}"
            )
        if self.ent_coef_learning_rate is not None and self.ent_coef_learning_rate <= 0.0:
            raise ValueError(
                f"ent_coef_learning_rate must be > 0 when set, got {self.ent_coef_learning_rate}"
            )
        if self.learning_starts < 0:
            raise ValueError(f"learning_starts must be >= 0, got {self.learning_starts}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        total_replay_capacity = self.buffer_capacity_per_env * self.env.action_space.n_envs
        if self.replay_fill_target > total_replay_capacity:
            raise ValueError(
                f"Replay fill target {self.replay_fill_target} exceeds total replay capacity "
                f"{total_replay_capacity}. Increase buffer_capacity_per_env or reduce learning_starts/batch_size."
            )
        if self.rollout_steps_per_iteration <= 0:
            raise ValueError(
                f"rollout_steps_per_iteration must be > 0, got {self.rollout_steps_per_iteration}"
            )
        if self.rollout_steps_per_iteration % self.env.action_space.n_envs != 0:
            raise ValueError(
                f"rollout_steps_per_iteration must be a multiple of n_envs ({self.env.action_space.n_envs}), "
                f"got {self.rollout_steps_per_iteration}"
            )
        if self.rollout_warmup_steps_per_env < 0:
            raise ValueError(
                f"rollout_warmup_steps_per_env must be >= 0, got {self.rollout_warmup_steps_per_env}"
            )
        if self.gradient_steps == 0 or self.gradient_steps < -1:
            raise ValueError(f"gradient_steps must be -1 or > 0, got {self.gradient_steps}")
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.target_update_interval <= 0:
            raise ValueError(f"target_update_interval must be > 0, got {self.target_update_interval}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be > 0 when set, got {self.max_grad_norm}")
        if self.nop_batch_size <= 0:
            raise ValueError(f"nop_batch_size must be > 0, got {self.nop_batch_size}")
        if (
                self.policy.has_nop_loss()
                and not self.independent_nop_sampling
                and self.nop_batch_size != self.batch_size
        ):
            raise ValueError(
                "nop_batch_size must equal batch_size when independent_nop_sampling=False; "
                "shared-origin NOP uses the Bellman batch origins."
            )
        resolve_gsde_reset_mode(
            gsde_enabled=self.policy.gsde_enabled,
            reset_mode=self.gsde_reset_mode,
        )

    @staticmethod
    def _set_requires_grad(parameters: list[torch.nn.Parameter], value: bool) -> None:
        for parameter in parameters:
            parameter.requires_grad_(value)

    def _reset_train_gsde_noise(self, local_obs: torch.Tensor) -> None:
        if not self.policy.gsde_enabled:
            return
        self.policy.action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))

    @staticmethod
    def _episode_metrics(episode_infos: list[dict[str, Any]]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        key_map = {
            "r": "ep_rew",
            "l": "ep_len",
            "t": "ep_time",
            "progress_reward": "ep_progress_rew",
            "guidance_reward": "ep_guidance_rew",
            "successful_connections_per_unit_mean": "ep_successful_connections_per_unit_mean",
            "successful_connections_per_unit_std": "ep_successful_connections_per_unit_std",
            "successful_connections_per_unit_min": "ep_successful_connections_per_unit_min",
            "successful_connections_per_unit_max": "ep_successful_connections_per_unit_max",
        }

        def add_episode_group_metrics(
                group_infos: list[dict[str, Any]],
                *,
                prefix: str,
                include_empty_statistics: bool,
                make_histograms: bool,
        ) -> None:
            metric_prefix = "" if not prefix else f"{prefix}/"
            ranged_info_keys = {
                "r",
                "l",
                "progress_reward",
                "guidance_reward",
                "successful_connections_per_unit_mean",
                "successful_connections_per_unit_std",
                "successful_connections_per_unit_min",
                "successful_connections_per_unit_max",
            }
            for info_key, metric_key in key_map.items():
                values = [ep_info[info_key] for ep_info in group_infos if info_key in ep_info]
                if not values and not include_empty_statistics:
                    continue
                metrics[f"{metric_prefix}{metric_key}"] = compute_summary_statistics(
                    values,
                    find_min=info_key in ranged_info_keys,
                    find_max=info_key in ranged_info_keys,
                    make_histogram=15 if make_histograms and info_key in ranged_info_keys else False,
                )
            success_values = [
                float(ep_info["success"])
                for ep_info in group_infos
                if "success" in ep_info
            ]
            if success_values:
                metrics[f"{metric_prefix}ep_success_rate"] = (
                    100.0 * sum(success_values) / len(success_values)
                )

        add_episode_group_metrics(
            episode_infos,
            prefix="",
            include_empty_statistics=True,
            make_histograms=True,
        )
        scenario_episode_infos: dict[str, list[dict[str, Any]]] = {}
        for ep_info in episode_infos:
            scenario_name = ep_info.get("scenario_name", None)
            scenario_id = ep_info.get("scenario_id", None)
            if scenario_name is None and scenario_id is None:
                continue
            scenario_key = str(scenario_name) if scenario_name is not None else f"id_{scenario_id}"
            scenario_episode_infos.setdefault(scenario_key, []).append(ep_info)
        for scenario_key, scenario_infos in scenario_episode_infos.items():
            prefix = f"scenario/{scenario_key}"
            add_episode_group_metrics(
                scenario_infos,
                prefix=prefix,
                include_empty_statistics=False,
                make_histograms=False,
            )
            metrics[f"{prefix}/episodes"] = len(scenario_infos)
        return metrics

    def _move_entropy_tensors_to_train_device(self) -> None:
        optimizer_replaced = False
        if self.log_ent_coef is not None and self.log_ent_coef.device != self.train_device:
            self.log_ent_coef = self.log_ent_coef.detach().to(self.train_device).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef],
                lr=self._resolved_ent_coef_learning_rate(),
            )
            optimizer_replaced = True
        if self.ent_coef_tensor is not None and self.ent_coef_tensor.device != self.train_device:
            self.ent_coef_tensor = self.ent_coef_tensor.to(self.train_device)
        if optimizer_replaced:
            self._rebuild_optimizer_steps()

    @staticmethod
    def _serialize_gsde_reset_mode(mode: GSDEResetMode | None) -> dict[str, Any] | None:
        if mode is None:
            return None
        if isinstance(mode, GSDEIntervalResetMode):
            return {"type": "interval", "interval": mode.interval}
        if isinstance(mode, GSDEProbabilityResetMode):
            return {"type": "probability", "probability": mode.probability}
        return {"type": type(mode).__name__}

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return {
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "entropy_agent_reduction": ENTROPY_AGENT_REDUCTION,
            "ent_coef_optimizer": None if self.ent_coef_optimizer is None else self.ent_coef_optimizer.state_dict(),
            "log_ent_coef": None if self.log_ent_coef is None else self.log_ent_coef.detach(),
            "ent_coef_tensor": None if self.ent_coef_tensor is None else self.ent_coef_tensor.detach(),
        }

    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        if missing_keys:
            raise NotImplementedError("Cannot safely load SAC optimizer state when policy keys are missing.")
        if unexpected_keys:
            logger.warning(f"Loading SAC optimizer state with unexpected policy keys: {unexpected_keys}")
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        critic_optimizer_state = state_dict["critic_optimizer"]
        saved_critic_parameter_count = sum(
            len(param_group["params"])
            for param_group in critic_optimizer_state["param_groups"]
        )
        current_critic_parameter_count = sum(
            len(param_group["params"])
            for param_group in self.critic_optimizer.param_groups
        )
        if saved_critic_parameter_count == current_critic_parameter_count:
            self.critic_optimizer.load_state_dict(critic_optimizer_state)
        else:
            logger.warning(
                "Critic optimizer parameter count changed "
                f"({saved_critic_parameter_count} -> {current_critic_parameter_count}); "
                "reinitializing critic optimizer state."
            )
            self.critic_optimizer = torch.optim.Adam(
                self.policy.critic_parameters(),
                lr=self._actor_critic_learning_rate_for_update(self.n_total_updates),
            )
        saved_entropy_agent_reduction = state_dict.get("entropy_agent_reduction")
        self.log_ent_coef = None
        self.ent_coef_optimizer = None
        self.ent_coef_tensor = None
        if saved_entropy_agent_reduction != ENTROPY_AGENT_REDUCTION:
            logger.warning(
                "Loaded a SAC checkpoint without compatible mean-agent entropy semantics; "
                "resetting the entropy coefficient from the current configuration."
            )
            self._setup_entropy_coefficient()
        else:
            log_ent_coef = state_dict.get("log_ent_coef", None)
            ent_coef_tensor = state_dict.get("ent_coef_tensor", None)
            if log_ent_coef is not None and ent_coef_tensor is not None:
                raise ValueError("Invalid SAC optimizer state: both log_ent_coef and ent_coef_tensor are set.")
            if log_ent_coef is not None:
                self.log_ent_coef = log_ent_coef.to(self.train_device).detach().requires_grad_(True)
                self.ent_coef_optimizer = torch.optim.Adam(
                    [self.log_ent_coef],
                    lr=self._resolved_ent_coef_learning_rate(),
                )
                ent_state = state_dict.get("ent_coef_optimizer", None)
                if ent_state is not None:
                    self.ent_coef_optimizer.load_state_dict(ent_state)
                self._apply_entropy_coefficient_learning_rate()
            if ent_coef_tensor is not None:
                self.ent_coef_tensor = ent_coef_tensor.to(self.train_device).detach()
        self._move_optimizer_state_to_device(self.actor_optimizer, self.train_device)
        self._move_optimizer_state_to_device(self.critic_optimizer, self.train_device)
        if self.ent_coef_optimizer is not None:
            self._move_optimizer_state_to_device(self.ent_coef_optimizer, self.train_device)
        self._rebuild_optimizer_steps()

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        actor_critic_lr = self._actor_critic_learning_rate_for_update(self.n_total_updates)
        for optimizer in (self.actor_optimizer, self.critic_optimizer):
            for param_group in optimizer.param_groups:
                param_group["lr"] = actor_critic_lr
        self._apply_entropy_coefficient_learning_rate()

    def _apply_entropy_coefficient_learning_rate(self) -> None:
        if self.ent_coef_optimizer is None:
            return
        learning_rate = self._resolved_ent_coef_learning_rate()
        for param_group in self.ent_coef_optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _resolved_ent_coef_learning_rate(self) -> float:
        if self.ent_coef_learning_rate is None:
            return self.learning_rate
        return self.ent_coef_learning_rate

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and value.device != device:
                    state[key] = value.to(device)

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None,
    ) -> bool:
        if cmd in {"set_batch_size", "batch_size"}:
            batch_size = int(params)
            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {batch_size}")
            total_replay_capacity = self.buffer_capacity_per_env * self.env.action_space.n_envs
            if max(self.learning_starts, batch_size) > total_replay_capacity:
                raise ValueError(
                    f"batch_size={batch_size} would make the replay fill target exceed total replay capacity "
                    f"{total_replay_capacity}."
                )
            self.batch_size = batch_size
            logger.warning(f"Setting batch_size to {self.batch_size}")
            return True
        if cmd == "set_gamma":
            self.gamma = float(params)
            if not (0.0 <= self.gamma <= 1.0):
                raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
            logger.warning(f"Setting gamma to {self.gamma}")
            return True
        if cmd == "set_tau":
            self.tau = float(params)
            if not (0.0 < self.tau <= 1.0):
                raise ValueError(f"tau must be in (0, 1], got {self.tau}")
            logger.warning(f"Setting tau to {self.tau}")
            return True
        if cmd in {"set_ent_coef", "ent_coef"}:
            ent_coef = float(params)
            if ent_coef < 0:
                raise ValueError(f"ent_coef must be >= 0, got {ent_coef}")
            self.ent_coef = ent_coef
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self._ent_coef_optimizer_step = None
            self.ent_coef_tensor = torch.tensor(ent_coef, device=self.train_device, dtype=torch.float32)
            logger.warning(f"Setting fixed ent_coef to {ent_coef}")
            return True
        if cmd in {"set_nop_loss_coef", "set_wm_loss_coef", "wm_loss_coef"}:
            coef = float(params)
            logger.warning(f"Setting NOP loss coefficient to {coef}")
            self.policy.update_loss_weights(nop_loss_coef=coef)
            return True
        return super()._execute_command(cmd, params, extra_run_metadata)
