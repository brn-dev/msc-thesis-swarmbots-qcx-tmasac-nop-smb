import abc
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Optional, Any, Literal, TypeVar, Generic, Callable, Protocol, NotRequired, TypedDict, cast

import torch
import torch.nn as nn
from loguru import logger

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm, LearningRate, _parse_bool
from swarmbots.learn.algos.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import (
    PPORolloutState,
    collect_step_rollout_batch,
    collect_steps,
    collect_whole_episodes,
    warmup_rollout_steps,
)
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEIntervalResetMode,
    GSDEProbabilityResetMode,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.masking import masked_mean
from swarmbots.learn.metrics_list import MetricsLists
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_fn
from swarmbots.learn.scheduling.schedulers import SchedulerManager
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.temporal_state import index_temporal_state
from swarmbots.learn.torch_device import as_device

TARGET_KL_MARGIN = 1.5
COMPUTE_GRAD_NORMS_EVERY_N_UPDATES = 4
COMPUTE_DETAILED_GRAD_NORMS_EVERY_N_ITERATIONS = 25

AGENTS_DIM = 1

try:
    logger.level("SAVE")
except ValueError:
    logger.level("SAVE", no=21, color="<magenta>")

class AutomaticLearningRateUpdateResult(TypedDict):
    new_lr: Optional[float]
    msg: NotRequired[str]
    event: NotRequired[str]

class AutomaticLearningRateUpdater(Protocol):

    def __call__(
            self,
            old_lr: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        ...


@dataclass(slots=True)
class AutomaticLearningRate:
    initial_lr: float
    updater: AutomaticLearningRateUpdater

    max_lr: float = 1e-2

PPOLearningRate = float | AutomaticLearningRate

class PPORolloutMode(abc.ABC):
    pass

@dataclass(slots=True, frozen=True)
class WholeEpisodesRolloutMode(PPORolloutMode):
    n_episodes_per_rollout: int


@dataclass(slots=True, frozen=True)
class StepsRolloutMode(PPORolloutMode):
    n_steps_per_rollout: int


PPOSamplesType = TypeVar('PPOSamplesType', bound=PPOSamples)
PPOSamplerConfigType = TypeVar('PPOSamplerConfigType', bound=PPOSamplerConfig)

# based on https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
class PPO(BaseAlgorithm, Generic[PPOSamplesType, PPOSamplerConfigType]):

    policy: BasePPOPolicy[PPOSamplesType, PPOSamplerConfigType]
    learning_rate: float

    def __init__(
            self,
            policy: BasePPOPolicy[PPOSamplesType, PPOSamplerConfigType],
            env: BaseLearnEnvWrapper,
            learning_rate: PPOLearningRate = 3e-4,
            rollout_mode: PPORolloutMode = WholeEpisodesRolloutMode(6),
            max_episode_length: int = 1000,
            sampler_config: PPOSamplerConfigType | None = None,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
            normalize_advantage: bool = True,
            mc_ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            value_loss_fn: nn.Module | None = None,
            max_grad_norm: float = 2.0,
            target_kl: Optional[float] = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            agent_logprob_reduction: Optional[Literal["sum", "mean"]] = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            record_device: str | torch.device | None = None,
            metrics_action_splitters: list[Callable[[torch.Tensor], dict[str, torch.Tensor]] | None] | None = None,
            use_popart: bool = False,
            scheduler_manager: SchedulerManager | None = None,
            virtual_mini_batches: int = 1,
            parameter_lr_multipliers: Mapping[str, float] | None = None,
            rollout_warmup_steps_per_env: int = 0,
    ):
        self.automatic_lr: AutomaticLearningRate | None = None
        self._auto_lr_enabled = False
        self._auto_lr_state: dict[str, Any] = {}

        initial_lr: float
        if isinstance(learning_rate, AutomaticLearningRate):
            if target_kl is None:
                raise ValueError("AutomaticLearningRate requires target_kl (otherwise KL early stopping cannot happen).")
            self.automatic_lr = learning_rate
            self._auto_lr_enabled = True
            initial_lr = learning_rate.initial_lr
        else:
            if not isinstance(learning_rate, float):
                raise TypeError(f"{learning_rate=} must be a float (or AutomaticLearningRate for auto LR).")
            initial_lr = learning_rate

        super().__init__(policy, env, initial_lr)

        self.learning_rate = initial_lr
        self.rollout_mode = rollout_mode
        self._rollout_state: PPORolloutState | None = None
        self.rollout_warmup_steps_per_env = int(rollout_warmup_steps_per_env)
        if self.rollout_warmup_steps_per_env < 0:
            raise ValueError(
                f"rollout_warmup_steps_per_env must be >= 0, got {self.rollout_warmup_steps_per_env}"
            )
        if self.rollout_warmup_steps_per_env > 0 and not isinstance(self.rollout_mode, StepsRolloutMode):
            raise ValueError("rollout_warmup_steps_per_env is only supported with StepsRolloutMode.")
        self._rollout_warmup_done = self.rollout_warmup_steps_per_env == 0
        self.max_episode_length = max_episode_length
        if sampler_config is None:
            sampler_config = cast(PPOSamplerConfigType, PPOSamplerConfig(batch_size=64))
        self.sampler_config = sampler_config
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.virtual_mini_batches = self._validate_virtual_mini_batches(virtual_mini_batches)
        if self.sampler_config.batch_size % self.virtual_mini_batches != 0:
            raise ValueError(
                f"sampler_config.batch_size must be divisible by virtual_mini_batches, got "
                f"batch_size={self.sampler_config.batch_size} and "
                f"virtual_mini_batches={self.virtual_mini_batches}"
            )
        self.mc_ent_coef = mc_ent_coef
        self.vf_coef = vf_coef
        self.value_loss_fn = value_loss_fn if value_loss_fn is not None else nn.MSELoss(reduction="none")
        self._validate_value_loss_fn(self.value_loss_fn)
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_popart = bool(use_popart)
        if self.use_popart and not self.policy.has_popart:
            raise ValueError(
                "use_popart=True, but policy.has_popart=False. "
                "Enable PopArt in the policy/critic first."
            )

        self.gsde_reset_mode = gsde_reset_mode
        resolve_gsde_reset_mode(
            gsde_enabled=policy.gsde_enabled,
            reset_mode=self.gsde_reset_mode,
        )
        if agent_logprob_reduction not in (None, "sum", "mean"):
            raise ValueError(f"{agent_logprob_reduction=} must be 'sum', 'mean', or None")
        if agent_logprob_reduction is not None and not isinstance(policy, PPOPolicy):
            logger.warning('agent_logprob_reduction is only intended for single agent PPO')
        self.agent_logprob_reduction: Optional[Literal["sum", "mean"]] = agent_logprob_reduction

        self.metrics_action_splitters: list[Callable[[torch.Tensor], dict[str, torch.Tensor]] | None]
        if metrics_action_splitters is not None:
            self.metrics_action_splitters = metrics_action_splitters
        else:
            self.metrics_action_splitters = [None] * len(self.policy.action_dist.distributions)

        self.train_device = as_device(train_device)
        self.rollout_device = as_device(rollout_device)
        self.record_device = self.rollout_device if record_device is None else as_device(record_device)
        self.scheduler_manager = scheduler_manager
        self.parameter_lr_multipliers = self._normalize_parameter_lr_multipliers(parameter_lr_multipliers)

        self.rollout_buffer_max_episode_length = self._resolve_rollout_buffer_max_episode_length(
            rollout_mode=self.rollout_mode,
            max_episode_length=self.max_episode_length,
            n_envs=env.action_space.n_envs,
        )
        self.rollout_buffer = PPORolloutBuffer(
            max_episode_length=self.rollout_buffer_max_episode_length,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=gamma,
            gae_lambda=gae_lambda,
            rollout_device=self.rollout_device,
            rollout_dtype=torch.float32,
            train_device=self.train_device,
            train_dtype=torch.float32,
        )

        self._policy_num_params = self.policy.num_parameters(learnable_only=False)
        self._policy_num_trainable_params = self.policy.num_parameters()
        self._detailed_grad_norm_metric_keys: tuple[str, ...] = tuple(self.policy.get_grad_norms().keys())

        self.optimizer = torch.optim.Adam(
            self._make_optimizer_param_groups(self.learning_rate),
            lr=self.learning_rate,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        auto_lr: dict[str, Any] | None
        if self.automatic_lr is None:
            auto_lr = None
        else:
            auto_lr = {
                "enabled": self._auto_lr_enabled,
                "initial_lr": self.automatic_lr.initial_lr,
                "max_lr": self.automatic_lr.max_lr,
                "updater": serialize_fn(self.automatic_lr.updater),
            }

        return {
            'learning_rate': self.learning_rate,
            'parameter_lr_multipliers': self.parameter_lr_multipliers,
            'automatic_learning_rate': auto_lr,
            'rollout_mode': self._serialize_rollout_mode(self.rollout_mode),
            'rollout_warmup_steps_per_env': self.rollout_warmup_steps_per_env,
            'max_episode_length': self.max_episode_length,
            'rollout_buffer_max_episode_length': self.rollout_buffer_max_episode_length,
            'sampler_config': serialize_dataclass(self.sampler_config),
            'n_epochs': self.n_epochs,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_range': self.clip_range,
            'clip_range_vf': self.clip_range_vf,
            'normalize_advantage': self.normalize_advantage,
            'virtual_mini_batches': self.virtual_mini_batches,
            'mc_ent_coef': self.mc_ent_coef,
            'vf_coef': self.vf_coef,
            'value_loss_fn': str(self.value_loss_fn),
            'max_grad_norm': self.max_grad_norm,
            'target_kl': self.target_kl,
            'train_device': str(self.train_device),
            'rollout_device': str(self.rollout_device),
            'record_device': str(self.record_device),
            'use_popart': self.use_popart,
            'gsde_reset_mode': self._serialize_gsde_reset_mode(self.gsde_reset_mode),
            'agent_logprob_reduction': self.agent_logprob_reduction,
            'policy_num_params': self._policy_num_params,
            'policy_num_trainable_params': self._policy_num_trainable_params,
            "schedulers": None if self.scheduler_manager is None else self.scheduler_manager.serialize(),
        }

    def compute_ppo_loss(
            self,
            batch: PPOSamples,
            log_probs: torch.Tensor,
            values: torch.Tensor,
            *,
            normalize_advantage: bool = True,
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        log_prob, old_log_prob = self.reduce_agents(batch, log_probs)
        valid_mask = self._build_batch_valid_mask(batch, log_prob)

        advantages = batch.advantages
        if normalize_advantage and self.normalize_advantage and advantages.numel() > 1:
            advantages = self._normalize_advantages(batch, advantages)

        ratio = torch.exp(log_prob - old_log_prob)

        while ratio.ndim > advantages.ndim:
            advantages = advantages.unsqueeze(-1)

        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -masked_mean(torch.min(policy_loss_1, policy_loss_2), valid_mask)

        value_targets = batch.returns
        new_values = values

        if self.use_popart:
            value_targets = self.policy.normalize_values(value_targets)
            new_values = self.policy.normalize_values(new_values)

        if self.clip_range_vf is None:
            values_pred = new_values
        else:
            old_values = batch.values
            if self.use_popart:
                old_values = self.policy.normalize_values(old_values)
            values_pred = old_values + torch.clamp(
                new_values - old_values, -self.clip_range_vf, self.clip_range_vf
            )

        value_valid_mask = self._build_batch_valid_mask(batch, value_targets)
        value_loss = self._compute_masked_value_loss(
            values_pred=values_pred,
            value_targets=value_targets,
            valid_mask=value_valid_mask,
        )

        mc_entropy_loss = masked_mean(log_prob, valid_mask)

        value_loss_scaled = self.vf_coef * value_loss
        mc_entropy_loss_scaled = self.mc_ent_coef * mc_entropy_loss

        loss = policy_loss + mc_entropy_loss_scaled + value_loss_scaled

        with torch.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl_div = masked_mean((torch.exp(log_ratio) - 1) - log_ratio, valid_mask).item()

        clip_fraction = masked_mean((torch.abs(ratio - 1) > self.clip_range).float(), valid_mask).item()
        ratio_for_metrics = ratio if valid_mask is None else ratio[valid_mask]
        metrics = {
            'act_loss': policy_loss.item(),
            'mc_ent_loss': mc_entropy_loss.item(),
            'mc_ent_loss_scaled': mc_entropy_loss_scaled.item(),
            'val_loss': value_loss.item(),
            'val_loss_scaled': value_loss_scaled.item(),
            'approx_kl': approx_kl_div,
            'clip_frac': clip_fraction,
            'ratio': compute_summary_statistics(ratio_for_metrics, find_min=True, find_max=True),
        }

        return loss, approx_kl_div, metrics

    def _compute_masked_value_loss(
            self,
            *,
            values_pred: torch.Tensor,
            value_targets: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        value_loss = self.value_loss_fn(values_pred, value_targets)
        if value_loss.shape != value_targets.shape:
            raise ValueError(
                f"Expected value loss shape {tuple(value_targets.shape)}, got {tuple(value_loss.shape)}"
            )
        return masked_mean(value_loss, valid_mask)

    @staticmethod
    def _validate_value_loss_fn(value_loss_fn: nn.Module) -> None:
        reduction = getattr(value_loss_fn, "reduction", None)
        if reduction != "none":
            raise ValueError(
                "value_loss_fn must expose reduction='none' so PPO can apply masks; "
                "for example use nn.MSELoss(reduction='none')"
            )

    def _select_valid_value_items(
            self,
            batch: Any,
            values: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = self._build_batch_valid_mask(batch, values)
        if valid_mask is None:
            return values.flatten()
        return values[valid_mask]

    def _before_learn_loop(self) -> None:
        if self._rollout_warmup_done:
            return
        if self.n_total_timesteps > 0 or self.n_total_iterations > 0 or self.n_total_updates > 0:
            self._rollout_warmup_done = True
            return
        if self._rollout_state is not None:
            self._rollout_warmup_done = True
            return
        if not isinstance(self.rollout_mode, StepsRolloutMode):
            raise ValueError("rollout_warmup_steps_per_env is only supported with StepsRolloutMode.")

        warmup_transitions = self.rollout_warmup_steps_per_env * self.rollout_buffer.n_envs
        logger.info(
            f"Running rollout warmup for {self.rollout_warmup_steps_per_env} vector steps "
            f"({warmup_transitions} transitions) without training."
        )
        with PerformanceTimer() as warmup_timer:
            self._rollout_state = warmup_rollout_steps(
                env=self.env,
                policy=self.policy,
                n_steps=warmup_transitions,
                rollout_state=None,
                gsde_reset_mode=self.gsde_reset_mode,
            )
        self.rollout_buffer.reset()
        self._rollout_warmup_done = True
        logger.info(f"Finished rollout warmup in {warmup_timer.get_duration():.2f}s.")

    def compute_loss(
            self,
            batch: PPOSamplesType,
            *,
            normalize_advantage: bool = True,
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        log_probs, values, extra_losses, extra_loss_metrics = self.policy.evaluate_actions(
            batch=batch,
            action_splitter=self.metrics_action_splitters,
        )

        loss, approx_kl_div, metrics = self.compute_ppo_loss(
            batch=batch,
            log_probs=log_probs,
            values=values,
            normalize_advantage=normalize_advantage,
        )

        reduced_extra_losses = self._reduce_extra_losses(batch, extra_losses)
        if reduced_extra_losses:
            loss = loss + torch.stack(tuple(reduced_extra_losses.values())).sum()
            metrics.update({f"{name}_loss_scaled": value.item() for name, value in reduced_extra_losses.items()})
        metrics.update(extra_loss_metrics)

        return loss, approx_kl_div, metrics

    def perform_iteration(
            self,
            episode_return_ema: ExponentialMovingAverage,
            episode_success_rate_ema: ExponentialMovingAverage,
            update_ema: bool,
    ) -> tuple[dict[str, Any], int]:
        with PerformanceTimer() as rollout_timer:
            if isinstance(self.rollout_mode, WholeEpisodesRolloutMode):
                episodes, episode_infos, rollout_metrics = collect_whole_episodes(
                    env=self.env,
                    policy=self.policy,
                    buffer=self.rollout_buffer,
                    n_episodes=self.rollout_mode.n_episodes_per_rollout,
                    gsde_reset_mode=self.gsde_reset_mode,
                )
                training_data: list[PPOEpisodeSegment] | PPORolloutBatch = episodes
                total_steps_in_rollout = sum(len(ep.rewards) for ep in episodes)
                self._rollout_state = None
            elif isinstance(self.rollout_mode, StepsRolloutMode):
                if self.policy.supports_rollout_batch_sampler(self.sampler_config):
                    rollout_batch, episode_infos, rollout_metrics, self._rollout_state = collect_step_rollout_batch(
                        env=self.env,
                        policy=self.policy,
                        buffer=self.rollout_buffer,
                        n_steps=self.rollout_mode.n_steps_per_rollout,
                        rollout_state=self._rollout_state,
                        gsde_reset_mode=self.gsde_reset_mode,
                    )
                    training_data = rollout_batch
                    total_steps_in_rollout = rollout_batch.n_samples
                else:
                    episodes, episode_infos, rollout_metrics, self._rollout_state = collect_steps(
                        env=self.env,
                        policy=self.policy,
                        buffer=self.rollout_buffer,
                        n_steps=self.rollout_mode.n_steps_per_rollout,
                        rollout_state=self._rollout_state,
                        gsde_reset_mode=self.gsde_reset_mode,
                    )
                    training_data = episodes
                    total_steps_in_rollout = sum(len(ep.rewards) for ep in episodes)
            else:
                raise TypeError(f"Unknown rollout_mode type: {type(self.rollout_mode)}")
        self.n_total_timesteps += total_steps_in_rollout
        self.n_total_iterations += 1

        ep_rew = compute_summary_statistics(
            [ep['r'] for ep in episode_infos], find_min=True, find_max=True, make_histogram=20
        )
        ep_len = compute_summary_statistics(
            [ep['l'] for ep in episode_infos], find_min=True, find_max=True, make_histogram=10
        )
        ep_time = compute_summary_statistics([ep['t'] for ep in episode_infos])
        ep_progress_rew = compute_summary_statistics(
            [ep['progress_reward'] for ep in episode_infos if 'progress_reward' in ep],
            find_min=True,
            find_max=True,
            make_histogram=20,
        )
        ep_guidance_rew = compute_summary_statistics(
            [ep['guidance_reward'] for ep in episode_infos if 'guidance_reward' in ep],
            find_min=True,
            find_max=True,
            make_histogram=20,
        )
        connection_metrics = {
            f"ep_successful_connections_per_unit_{statistic_name}": compute_summary_statistics(
                [
                    ep[f"successful_connections_per_unit_{statistic_name}"]
                    for ep in episode_infos
                    if f"successful_connections_per_unit_{statistic_name}" in ep
                ],
                find_min=True,
                find_max=True,
            )
            for statistic_name in ("mean", "std", "min", "max")
        }

        if update_ema:
            for ep_info in episode_infos:
                episode_return_ema.update(ep_info['r'])
                if "success" in ep_info:
                    episode_success_rate_ema.update(float(ep_info["success"]))

        update_metrics = self.train(training_data)
        metrics = {
            **update_metrics,
            **rollout_metrics,
            'rollout_time': rollout_timer.get_duration(),
            'ep_rew': ep_rew,
            'ep_len': ep_len,
            'ep_time': ep_time,
            'ep_progress_rew': ep_progress_rew,
            'ep_guidance_rew': ep_guidance_rew,
            **connection_metrics,
        }
        return metrics, total_steps_in_rollout

    def train(self, training_data: list[PPOEpisodeSegment] | PPORolloutBatch) -> dict[str, Any]:
        with PerformanceTimer() as to_train_device_timer:
            self.policy.train()
            self.policy.to(self.train_device)
            self.value_loss_fn.to(self.train_device)

        with PerformanceTimer() as sampler_init_timer:
            if isinstance(training_data, PPORolloutBatch):
                sampler = self.policy.make_rollout_batch_sampler(
                    rollout_batch=training_data,
                    config=self.sampler_config,
                )
            else:
                sampler = self.policy.make_sampler(training_data, config=self.sampler_config)
                training_data.clear()
        self.rollout_buffer.episodes.clear()

        valid_returns = self._select_valid_value_items(sampler, sampler.returns)
        if self.use_popart and valid_returns.numel() > 0:
            self.policy.update_value_normalizer(valid_returns)

        y_pred = self._select_valid_value_items(sampler, sampler.values)
        y_true = valid_returns
        if y_true.numel() > 1:
            var_y = torch.var(y_true)
        else:
            var_y = y_true.new_tensor(float("nan"))
        if not torch.isnan(var_y) and var_y > 1e-8:
            explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
        else:
            explained_var = 0.0

        loss_metrics = MetricsLists[float]()

        continue_training = True
        early_stop_epoch: int | None = None
        early_stop_kl_div: float | None = None
        n_updates = 0
        grad_norms: list[float] = []
        n_grad_clipped = 0

        sampling_timings: list[float] = []
        sample_timer = PerformanceTimer()

        update_timings: list[float] = []
        update_timer = PerformanceTimer()

        detailed_grad_norms = MetricsLists[float]()
        compute_grad_norms_timings: list[float] = []
        compute_grad_norms_timer = PerformanceTimer()
        compute_detailed_grad_norms_this_iteration = (
            self.n_total_iterations % COMPUTE_DETAILED_GRAD_NORMS_EVERY_N_ITERATIONS == 0
        )

        train_timer = PerformanceTimer().start()
        for epoch in range(self.n_epochs):
            sample_timer.start()
            for i, batch in enumerate(sampler.sample()):
                sampling_timings.append(sample_timer.stop().get_duration())

                update_timer.start()

                if self.virtual_mini_batches == 1:
                    loss, approx_kl_div, metrics = self.compute_loss(batch)
                    loss_metrics.add(metrics)

                    if self.target_kl is not None and approx_kl_div > TARGET_KL_MARGIN * self.target_kl:
                        continue_training = False
                        early_stop_epoch = epoch
                        early_stop_kl_div = approx_kl_div
                        msg = f"Early stopping at epoch {epoch}, batch {i} due to reaching max kl: {approx_kl_div:.3f}"
                        if epoch == 0:
                            logger.warning(msg)
                        else:
                            logger.debug(msg)
                        break

                    self.optimizer.zero_grad()
                    loss.backward()
                else:
                    approx_kl_div = self._compute_virtual_batch_gradients(
                        batch=batch,
                        epoch=epoch,
                        batch_idx=i,
                        loss_metrics=loss_metrics,
                    )
                    if self.target_kl is not None and approx_kl_div > TARGET_KL_MARGIN * self.target_kl:
                        continue_training = False
                        early_stop_epoch = epoch
                        early_stop_kl_div = approx_kl_div
                        self.optimizer.zero_grad()
                        msg = (
                            f"Early stopping at epoch {epoch}, batch {i} due to reaching max kl "
                            f"after virtual mini-batches: {approx_kl_div:.3f}"
                        )
                        if epoch == 0:
                            logger.warning(msg)
                        else:
                            logger.debug(msg)
                        break

                should_compute_grad_norms = (
                    compute_detailed_grad_norms_this_iteration
                    and n_updates % COMPUTE_GRAD_NORMS_EVERY_N_UPDATES == 0
                )
                if should_compute_grad_norms:
                    with compute_grad_norms_timer:
                        detailed_grad_norms.add(self.policy.get_grad_norms())
                    compute_grad_norms_timings.append(compute_grad_norms_timer.get_duration())

                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                    error_if_nonfinite=True,
                )
                total_grad_norm_f = float(total_grad_norm)
                grad_norms.append(total_grad_norm_f)

                if total_grad_norm_f > self.max_grad_norm:
                    n_grad_clipped += 1

                self.optimizer.step()
                self._after_optimizer_step()

                n_updates += 1

                update_timings.append(update_timer.stop().get_duration())
                sample_timer.start()

            if not continue_training:
                break

        train_timer.stop()

        self.n_total_updates += n_updates

        detailed_grad_norm_metrics = detailed_grad_norms.compute_summary_statistics(prefix='grad_norm_',)
        if detailed_grad_norm_metrics:
            self._detailed_grad_norm_metric_keys = tuple(k.removeprefix('grad_norm_') for k in detailed_grad_norm_metrics)
        else:
            detailed_grad_norm_metrics = {
                f'grad_norm_{key}': compute_summary_statistics([])
                for key in self._detailed_grad_norm_metric_keys
            }

        metrics_timer = PerformanceTimer().start()
        with torch.no_grad():
            metrics: dict[str, Any] = {
                **loss_metrics.compute_summary_statistics(),
                'updates': n_updates,
                'virtual_mini_batches': self.virtual_mini_batches,
                'total_updates': self.n_total_updates,
                'expl_var': explained_var,
                'grad_norm': compute_summary_statistics(grad_norms, find_max=True, find_min=True),
                'grad_clip_frac': (n_grad_clipped / len(grad_norms)) if grad_norms else 0.0,
                **detailed_grad_norm_metrics,
                'total_compute_grad_norms_time': sum(compute_grad_norms_timings),
                'compute_grad_norms_time': compute_summary_statistics(compute_grad_norms_timings)
            }

            auto_lr_metrics = self._maybe_update_automatic_lr(
                early_stop_kl_div=early_stop_kl_div,
                early_stop_epoch=early_stop_epoch,
                metrics=metrics,
            )
            metrics.update(auto_lr_metrics)
            metrics.update(self.policy.get_value_normalizer_metrics())
            actions_for_metrics = self._get_action_metrics_actions(sampler)
            metrics.update(
                self.policy.action_dist.get_metrics(
                    actions=actions_for_metrics,
                    action_splitter=self.metrics_action_splitters,
                )
            )
            metrics.update(self._maybe_apply_schedulers(metrics))

        metrics_timer.stop()

        return {
            **metrics,
            'to_train_device_time': to_train_device_timer.get_duration(),
            'sampler_init_time': sampler_init_timer.get_duration(),
            'sampling_time': compute_summary_statistics(sampling_timings),
            'total_sampling_time': sum(sampling_timings),
            'update_time': compute_summary_statistics(update_timings),
            'total_update_time': sum(update_timings),
            'metrics_time': metrics_timer.get_duration(),
            'train_time': train_timer.get_duration(),
        }

    def _maybe_update_automatic_lr(
            self,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> dict[str, Any]:
        if self.automatic_lr is None:
            return {}

        if not self._auto_lr_enabled:
            return {"auto_lr_event": "disabled", "auto_lr": self.learning_rate}

        update_result = self.automatic_lr.updater(
            old_lr=self.learning_rate,
            state=self._auto_lr_state,
            n_iterations=self.n_total_iterations,
            n_model_updates=self.n_total_updates,
            n_timesteps=self.n_total_timesteps,
            early_stop_kl_div=early_stop_kl_div,
            early_stop_epoch=early_stop_epoch,
            metrics=metrics,
        )
        requested_lr = update_result.get('new_lr', None)
        update_msg = update_result.get('msg', None)
        update_event = update_result.get('event', None)

        if requested_lr is None:
            return {"auto_lr_event": update_event, "auto_lr": self.learning_rate}

        new_lr = min(requested_lr, self.automatic_lr.max_lr)
        ratio = new_lr / self.learning_rate
        if new_lr > self.learning_rate:
            if new_lr < requested_lr:
                msg = f', {update_msg}' if update_msg else ''
                logger.warning(
                    f"Auto LR capped at {new_lr:.2e} (requested {requested_lr:.2e}{msg}, {ratio=:.2f})"
                )
            else:
                msg = f': {update_msg}' if update_msg else ''
                logger.warning(
                    f"Increasing LR to {new_lr:.2e}{msg} ({ratio=:.2f})"
                )
            self.set_learning_rate(new_lr)
            event = update_event or ("lr_increase" if new_lr == requested_lr else "lr_increase_capped")
            return {"auto_lr_event": event, "auto_lr": new_lr}

        if new_lr < self.learning_rate:
            msg = f': {update_msg}' if update_msg else ''
            logger.warning(
                f"Decaying LR to {new_lr:.2e}{msg} ({ratio=:.2f})"
            )
            self.set_learning_rate(new_lr)
            return {"auto_lr_event": update_event or "lr_decay", "auto_lr": new_lr}

        return {"auto_lr_event": update_event, "auto_lr": self.learning_rate}

    def _maybe_apply_schedulers(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.scheduler_manager is None:
            return {}

        scheduler_metrics: dict[str, Any] = {}
        for result in self.scheduler_manager.step(
            n_iterations=self.n_total_iterations,
            n_model_updates=self.n_total_updates,
            n_timesteps=self.n_total_timesteps,
            metrics=metrics,
        ):
            metric_prefix = f"scheduler_{result.name}"
            scheduler_metrics[f"{metric_prefix}_value"] = result.new_value
            scheduler_metrics[f"{metric_prefix}_event"] = result.event
            scheduler_metrics[f"{metric_prefix}_updated"] = result.updated

            if not result.updated:
                continue

            if result.msg is None:
                logger.warning(
                    f"Scheduler '{result.name}' updated value {result.old_value:.6g} -> {result.new_value:.6g}"
                )
            else:
                logger.warning(
                    f"Scheduler '{result.name}' updated value {result.old_value:.6g} -> "
                    f"{result.new_value:.6g}: {result.msg}"
                )
        return scheduler_metrics

    def _after_optimizer_step(self) -> None:
        self.policy.after_optimizer_step()

    @property
    def batch_size(self) -> int:
        return self.sampler_config.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if value <= 0:
            raise ValueError(f"batch_size must be > 0, got {value}")
        if value % self.virtual_mini_batches != 0:
            raise ValueError(
                f"batch_size must be divisible by virtual_mini_batches, got "
                f"{value=} and virtual_mini_batches={self.virtual_mini_batches}"
            )
        self.sampler_config = replace(self.sampler_config, batch_size=value)

    @staticmethod
    def _validate_virtual_mini_batches(value: int) -> int:
        if value <= 0:
            raise ValueError(f"virtual_mini_batches must be > 0, got {value}")
        return value

    def _compute_virtual_batch_gradients(
            self,
            *,
            batch: PPOSamplesType,
            epoch: int,
            batch_idx: int,
            loss_metrics: MetricsLists[float],
    ) -> float:
        batch_size = self._get_batch_leading_dim(batch)
        if batch_size % self.virtual_mini_batches != 0:
            raise ValueError(
                f"Logical PPO batch size must be divisible by virtual_mini_batches, got "
                f"{batch_size=} and virtual_mini_batches={self.virtual_mini_batches} "
                f"at epoch {epoch}, batch {batch_idx}"
            )

        batch = self._normalize_batch_advantages_once(batch)
        virtual_batches = self._split_batch(batch, n_chunks=self.virtual_mini_batches)
        self.optimizer.zero_grad()

        approx_kl_values: list[float] = []
        for virtual_batch in virtual_batches:
            loss, approx_kl_div, metrics = self.compute_loss(
                virtual_batch,
                normalize_advantage=False,
            )
            loss_metrics.add(metrics)
            approx_kl_values.append(approx_kl_div)
            (loss / self.virtual_mini_batches).backward()

        return sum(approx_kl_values) / len(approx_kl_values)

    def _normalize_batch_advantages_once(
            self,
            batch: PPOSamplesType,
    ) -> PPOSamplesType:
        if not self.normalize_advantage or batch.advantages.numel() <= 1:
            return batch
        return replace(batch, advantages=self._normalize_advantages(batch, batch.advantages))

    @classmethod
    def _split_batch(
            cls,
            batch: PPOSamplesType,
            *,
            n_chunks: int,
    ) -> list[PPOSamplesType]:
        if not is_dataclass(batch):
            raise TypeError(f"Expected dataclass batch, got {type(batch)}")
        batch_size = cls._get_batch_leading_dim(batch)
        if batch_size % n_chunks != 0:
            raise ValueError(f"Expected batch size divisible by {n_chunks}, got {batch_size}")
        chunk_size = batch_size // n_chunks
        result: list[PPOSamplesType] = []
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            values = {
                field.name: cls._slice_batch_field(getattr(batch, field.name), start_idx, end_idx)
                for field in fields(batch)
            }
            result.append(type(batch)(**values))
        return result

    @staticmethod
    def _get_batch_leading_dim(batch: PPOSamples) -> int:
        return int(batch.actions.shape[0])

    @staticmethod
    def _slice_batch_field(value: Any, start_idx: int, end_idx: int) -> Any:
        if isinstance(value, torch.Tensor):
            return value[start_idx:end_idx]
        if value is None:
            return None
        if isinstance(value, (tuple, list, Mapping)):
            return index_temporal_state(value, slice(start_idx, end_idx))
        raise TypeError(f"Unsupported batch field type {type(value)}")

    def reduce_agents(
            self,
            batch: PPOSamples,
            log_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        agent_mask = batch.agent_mask
        if agent_mask is not None:
            if agent_mask.dtype != torch.bool:
                raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
            if agent_mask.shape != log_probs.shape:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(log_probs.shape)}, got {tuple(agent_mask.shape)}"
                )

        agent_dim = log_probs.ndim - 1
        if self.agent_logprob_reduction is None:
            log_prob = log_probs
            old_log_prob = batch.log_probs
        elif self.agent_logprob_reduction == "sum":
            if agent_mask is None:
                log_prob = log_probs.sum(dim=agent_dim)
                old_log_prob = batch.log_probs.sum(dim=agent_dim)
            else:
                mask_f = agent_mask.to(dtype=log_probs.dtype)
                log_prob = (log_probs * mask_f).sum(dim=agent_dim)
                old_log_prob = (batch.log_probs * mask_f).sum(dim=agent_dim)
        elif self.agent_logprob_reduction == "mean":
            if agent_mask is None:
                log_prob = log_probs.mean(dim=agent_dim)
                old_log_prob = batch.log_probs.mean(dim=agent_dim)
            else:
                mask_f = agent_mask.to(dtype=log_probs.dtype)
                denom = mask_f.sum(dim=agent_dim).clamp_min(1.0)
                log_prob = (log_probs * mask_f).sum(dim=agent_dim) / denom
                old_log_prob = (batch.log_probs * mask_f).sum(dim=agent_dim) / denom
        else:
            raise ValueError(f"Unhandled {self.agent_logprob_reduction=}")
        return log_prob, old_log_prob

    def _reduce_extra_losses(
            self,
            batch: PPOSamples,
            extra_losses: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        reduced_losses: dict[str, torch.Tensor] = {}
        for name, value in extra_losses.items():
            reduced_losses[name] = self._reduce_extra_loss_value(batch, value)
        return reduced_losses

    def _reduce_extra_loss_value(
            self,
            batch: PPOSamples,
            value: torch.Tensor,
    ) -> torch.Tensor:
        if value.ndim == 0:
            return value

        if value.ndim == batch.log_probs.ndim:
            if value.shape != batch.log_probs.shape:
                raise ValueError(
                    f"Expected extra loss shape {tuple(batch.log_probs.shape)}, got {tuple(value.shape)}"
                )
            if self.agent_logprob_reduction is None:
                reduced = value
            elif self.agent_logprob_reduction == "sum":
                if batch.agent_mask is None:
                    reduced = value.sum(dim=value.ndim - 1)
                else:
                    mask_f = batch.agent_mask.to(dtype=value.dtype)
                    reduced = (value * mask_f).sum(dim=value.ndim - 1)
            elif self.agent_logprob_reduction == "mean":
                if batch.agent_mask is None:
                    reduced = value.mean(dim=value.ndim - 1)
                else:
                    mask_f = batch.agent_mask.to(dtype=value.dtype)
                    denom = mask_f.sum(dim=value.ndim - 1).clamp_min(1.0)
                    reduced = (value * mask_f).sum(dim=value.ndim - 1) / denom
            else:
                raise ValueError(f"Unhandled {self.agent_logprob_reduction=}")
        elif value.ndim == batch.log_probs.ndim - 1:
            reduced = value
        else:
            raise ValueError(f"Unsupported extra loss ndim {value.ndim} for shape {tuple(value.shape)}")

        valid_mask = self._build_batch_valid_mask(batch, reduced)
        return masked_mean(reduced, valid_mask)

    def _build_batch_valid_mask(
            self,
            batch: PPOSamples,
            target: torch.Tensor,
    ) -> torch.Tensor | None:
        return self._combine_valid_masks(
            agent_mask=batch.agent_mask,
            time_mask=self._get_time_mask(batch),
            target=target,
        )

    @classmethod
    def _combine_valid_masks(
            cls,
            *,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
            target: torch.Tensor,
    ) -> torch.Tensor | None:
        valid_mask = cls._build_agent_valid_mask(agent_mask, target)
        time_valid_mask = cls._build_time_valid_mask(time_mask, target)
        if valid_mask is None:
            return time_valid_mask
        if time_valid_mask is None:
            return valid_mask
        return valid_mask & time_valid_mask

    @staticmethod
    def _build_agent_valid_mask(
            agent_mask: torch.Tensor | None,
            target: torch.Tensor,
    ) -> torch.Tensor | None:
        if agent_mask is None:
            return None
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if agent_mask.ndim == target.ndim:
            if agent_mask.shape != target.shape:
                raise ValueError(
                    f"Expected agent_mask shape {tuple(target.shape)}, got {tuple(agent_mask.shape)}"
                )
            return agent_mask
        if agent_mask.ndim == target.ndim + 1:
            if agent_mask.shape[:-1] != target.shape:
                raise ValueError(
                    f"Expected agent_mask prefix shape {tuple(target.shape)}, got {tuple(agent_mask.shape)}"
                )
            return agent_mask.any(dim=-1)
        raise ValueError(
            f"Unsupported agent_mask ndim {agent_mask.ndim} for target ndim {target.ndim}"
        )

    @staticmethod
    def _build_time_valid_mask(
            time_mask: torch.Tensor | None,
            target: torch.Tensor,
    ) -> torch.Tensor | None:
        if time_mask is None:
            return None
        if time_mask.dtype != torch.bool:
            raise ValueError(f"Expected time_mask dtype bool, got {time_mask.dtype}")
        if time_mask.ndim == target.ndim:
            if time_mask.shape != target.shape:
                raise ValueError(
                    f"Expected time_mask shape {tuple(target.shape)}, got {tuple(time_mask.shape)}"
                )
            return time_mask
        if time_mask.ndim + 1 == target.ndim:
            if time_mask.shape != target.shape[:-1]:
                raise ValueError(
                    f"Expected time_mask shape {tuple(target.shape[:-1])}, got {tuple(time_mask.shape)}"
                )
            return time_mask.unsqueeze(-1).expand_as(target)
        raise ValueError(
            f"Unsupported time_mask ndim {time_mask.ndim} for target ndim {target.ndim}"
        )

    def _normalize_advantages(
            self,
            batch: PPOSamples,
            advantages: torch.Tensor,
    ) -> torch.Tensor:
        step_valid_mask = self._build_time_valid_mask(self._get_time_mask(batch), advantages)
        if step_valid_mask is None:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        valid_f = step_valid_mask.to(dtype=advantages.dtype)
        num_valid = int(valid_f.sum().item())
        if num_valid <= 1:
            return advantages

        mean = (advantages * valid_f).sum() / valid_f.sum()
        variance = ((advantages - mean) ** 2 * valid_f).sum() / valid_f.sum()
        normalized = (advantages - mean) / torch.sqrt(variance + 1e-8)
        return torch.where(step_valid_mask, normalized, torch.zeros_like(normalized))

    @staticmethod
    def _get_time_mask(batch: PPOSamples) -> torch.Tensor | None:
        return getattr(batch, "time_mask", None)

    @classmethod
    def _get_action_metrics_actions(
            cls,
            batch: PPOSamples,
    ) -> torch.Tensor:
        actions = batch.actions
        if actions.ndim < 2:
            return actions

        valid_mask = cls._build_action_metrics_valid_mask(batch)
        if valid_mask is None:
            return actions.reshape(-1, actions.shape[-1])
        return actions[valid_mask]

    @classmethod
    def _build_action_metrics_valid_mask(
            cls,
            batch: PPOSamples,
    ) -> torch.Tensor | None:
        agent_mask = batch.agent_mask
        time_mask = cls._get_time_mask(batch)

        valid_mask = agent_mask
        if time_mask is not None:
            time_agent_mask = time_mask.unsqueeze(-1)
            valid_mask = time_agent_mask if valid_mask is None else valid_mask & time_agent_mask
        return valid_mask

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> bool:
        """
        :return: True if the command was executed successfully and updated the hyper parameters, False otherwise
        """
        if cmd == 'set_std':
            std = float(params)
            logger.warning(f"Setting action std to {std}")
            self.policy.action_dist.set_std(std)
            return True
        elif cmd == 'scale_std':
            multiplier = float(params)
            logger.warning(f"Scaling action std by {multiplier}")
            self.policy.action_dist.scale_std(multiplier)
            return True
        elif cmd == 'set_n_epochs' or cmd == 'set_num_epochs':
            n_epochs = int(params)
            logger.warning(f'Setting num epochs to {n_epochs}')
            self.n_epochs = n_epochs
            return True
        elif cmd == 'set_target_kl':
            target_kl = float(params)
            logger.warning(f'Setting target KL to {target_kl}')
            self.target_kl = target_kl
            return True
        elif cmd == "set_clip_range":
            clip_range = float(params)
            logger.warning(f"Setting clip_range to {clip_range}")
            self.clip_range = clip_range
            return True
        elif cmd == "set_clip_range_vf":
            clip_range_vf = float(params)
            logger.warning(f"Setting clip_range_vf to {clip_range_vf}")
            self.clip_range_vf = clip_range_vf
            return True
        elif cmd == "set_mc_ent_coef":
            mc_ent_coef = float(params)
            logger.warning(f"Setting mc_ent_coef to {mc_ent_coef}")
            self.mc_ent_coef = mc_ent_coef
            return True
        elif cmd == "set_vf_coef":
            vf_coef = float(params)
            logger.warning(f"Setting vf_coef to {vf_coef}")
            self.vf_coef = vf_coef
            return True
        elif cmd in {"set_batch_size", "batch_size"}:
            batch_size = int(params)
            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {batch_size}")
            logger.warning(f"Setting batch_size to {batch_size}")
            self.batch_size = batch_size
            return True
        elif cmd in {"set_virtual_mini_batches", "virtual_mini_batches"}:
            virtual_mini_batches = self._validate_virtual_mini_batches(int(params))
            if self.batch_size % virtual_mini_batches != 0:
                raise ValueError(
                    f"batch_size must be divisible by virtual_mini_batches, got "
                    f"batch_size={self.batch_size} and virtual_mini_batches={virtual_mini_batches}"
                )
            logger.warning(f"Setting virtual_mini_batches to {virtual_mini_batches}")
            self.virtual_mini_batches = virtual_mini_batches
            return True
        elif cmd in {"set_normalize_advantage", "normalize_advantage"}:
            normalize_advantage = _parse_bool(params)
            logger.warning(f"Setting normalize_advantage to {normalize_advantage}")
            self.normalize_advantage = normalize_advantage
            return True
        elif cmd in {"set_gsde_prob", "set_gsde_probability", "gsde_prob"}:
            probability = float(params)
            if not (0.0 < probability < 1.0):
                raise ValueError(f"gsde probability must be in (0, 1), got {probability}")
            logger.warning(f"Setting gsde_reset_mode to probability={probability}")
            self.gsde_reset_mode = GSDEProbabilityResetMode(probability=probability)
            return True
        elif cmd in {"set_gsde_interval", "gsde_interval"}:
            interval = int(params)
            if interval <= 0:
                raise ValueError(f"gsde interval must be > 0, got {interval}")
            logger.warning(f"Setting gsde_reset_mode to interval={interval}")
            self.gsde_reset_mode = GSDEIntervalResetMode(interval=interval)
            return True
        elif cmd in {"disable_gsde_reset", "gsde_reset_off"}:
            logger.warning("Disabling gsde_reset_mode")
            self.gsde_reset_mode = None
            return True
        elif cmd in {"set_max_grad_norm", "max_grad_norm"}:
            max_grad_norm = float(params)
            if max_grad_norm <= 0:
                raise ValueError(f"max_grad_norm must be > 0, got {max_grad_norm}")
            logger.warning(f"Setting max_grad_norm to {max_grad_norm}")
            self.max_grad_norm = max_grad_norm
            return True
        elif cmd == "set_gamma":
            gamma = float(params)
            logger.warning(f"Setting gamma to {gamma}")
            self.gamma = gamma
            self.rollout_buffer.gamma = gamma
            return True
        elif cmd == "set_gae_lambda":
            gae_lambda = float(params)
            logger.warning(f"Setting gae_lambda to {gae_lambda}")
            self.gae_lambda = gae_lambda
            self.rollout_buffer.gae_lambda = gae_lambda
            return True
        elif cmd in {"set_extra_loss_weights", "set_loss_weights", "loss_weights"}:
            parsed_weights = json.loads(params)
            if not isinstance(parsed_weights, dict):
                raise ValueError(
                    "set_extra_loss_weights expects a JSON object, e.g. "
                    "set_extra_loss_weights:{\"entropy\":0.001}"
                )
            if not parsed_weights:
                logger.warning("No extra loss weights provided.")
                return False
            weights = {str(key): float(value) for key, value in parsed_weights.items()}
            logger.warning(f"Updating extra loss weights: {weights}")
            self.policy.update_loss_weights(**weights)
            return True
        elif cmd in {"set_wm_loss_coef", "set_world_model_loss_coef", "wm_loss_coef"}:
            coef = float(params)
            logger.warning(f"Setting world_model_loss_coef to {coef}")
            self.policy.update_loss_weights(world_model_loss_coef=coef)
            return True
        elif cmd in {"set_wm_num_next_steps", "set_world_model_num_next_steps", "wm_num_next_steps"}:
            num_next_steps = int(params)
            if num_next_steps < 1:
                raise ValueError(f"world_model_num_next_steps must be >= 1, got {num_next_steps}")
            if not isinstance(self.sampler_config, PPOWMSamplerConfig):
                raise ValueError(
                    "Current sampler_config does not expose num_next_steps; cannot configure world-model sampler horizon."
                )
            logger.warning(f"Setting world_model_num_next_steps to {num_next_steps}")
            self.sampler_config = replace(self.sampler_config, num_next_steps=num_next_steps)
            return True
        elif cmd in {"set_wm_target_tau", "set_world_model_target_tau", "wm_target_tau"}:
            param = params.strip().lower()
            if param in {"none", "null", ""}:
                tau: float | None = None
                logger.warning("Disabling world_model_target_tau")
            else:
                tau = float(params)
                if not (0.0 < tau <= 1.0):
                    raise ValueError(f"world_model_target_tau must be in (0, 1], got {tau}")
                logger.warning(f"Setting world_model_target_tau to {tau}")
            if not hasattr(self.policy, "world_model_target_tau"):
                raise ValueError(
                    "Policy does not expose world_model_target_tau; cannot configure target-network update rate."
                )
            setattr(self.policy, "world_model_target_tau", tau)
            return True
        elif cmd in {"set_act_ent_loss_coef", "set_sub_ent_loss_coef", "set_action_ent_loss_coef"}:
            sub_dist_idx, value = self._parse_indexed_float_params(
                params,
                idx_keys=("act", "idx", "index", "sub_dist"),
                value_keys=("value", "coef", "ent_loss_coef", "ent", "entropy"),
            )
            logger.warning(f"Setting entropy loss coef for action sub-dist {sub_dist_idx} to {value}")
            self.policy.update_loss_weights(**{f"act{sub_dist_idx}_ent_loss_coef": value})
            return True
        elif cmd in {"disable_auto_lr", "auto_lr_off", "disable_automatic_lr", "disable_auto_learning_rate"}:
            _ = params
            if self.automatic_lr is None:
                logger.warning("Automatic LR is not configured; nothing to disable.")
                return False
            if not self._auto_lr_enabled:
                logger.warning("Automatic LR is already disabled.")
                return False
            self._auto_lr_enabled = False
            logger.warning("Disabled automatic LR.")
            return True
        elif cmd in {"enable_auto_lr", "auto_lr_on", "enable_automatic_lr", "enable_auto_learning_rate"}:
            _ = params
            if self.automatic_lr is None:
                logger.warning("Automatic LR is not configured; cannot enable it.")
                return False
            if self.target_kl is None:
                raise ValueError("Cannot enable automatic LR when target_kl is None.")
            if self._auto_lr_enabled:
                logger.warning("Automatic LR is already enabled.")
                return False
            self._auto_lr_enabled = True
            self._auto_lr_state = {}
            logger.warning("Enabled automatic LR (resetting no-early-stop counter).")
            return True
        else:
            return super()._execute_command(cmd, params, extra_run_metadata)

    def _get_optimizer_state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def _apply_optimizer_state_dict(
            self,
            state_dict: dict[str, Any],
            missing_keys: list[str],
            unexpected_keys: list[str],
    ) -> None:
        if missing_keys:
            raise NotImplementedError()
        if unexpected_keys:
            state_dict = state_dict.copy()
            for k in unexpected_keys:
                state_dict.pop(k, None)

        self.optimizer.load_state_dict(state_dict)
        self._move_optimizer_state_to_device(self.train_device)
        try:
            param_group = self.optimizer.param_groups[0]
            lr = float(param_group["lr"]) / float(param_group.get("lr_multiplier", 1.0))
        except (KeyError, IndexError, TypeError, ValueError):
            return
        self.learning_rate = lr

    def _move_optimizer_state_to_device(self, device: torch.device) -> None:
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v) and v.device != device:
                    state[k] = v.to(device)

    def _apply_learning_rate(self, lr: LearningRate) -> None:
        assert isinstance(lr, float)
        for param_group in self.optimizer.param_groups:
            lr_multiplier = float(param_group.get("lr_multiplier", 1.0))
            param_group["lr"] = lr * lr_multiplier

    @staticmethod
    def _normalize_parameter_lr_multipliers(
            parameter_lr_multipliers: Mapping[str, float] | None,
    ) -> dict[str, float]:
        if not parameter_lr_multipliers:
            return {}

        normalized: dict[str, float] = {}
        for raw_prefix, raw_multiplier in parameter_lr_multipliers.items():
            prefix = raw_prefix.strip(".")
            multiplier = float(raw_multiplier)
            if not prefix:
                raise ValueError("Parameter LR multiplier prefixes must not be empty.")
            if multiplier <= 0:
                raise ValueError(f"Parameter LR multiplier for {prefix!r} must be > 0, got {multiplier}.")
            if multiplier == 1.0:
                continue
            normalized[prefix] = multiplier
        return normalized

    def _make_optimizer_param_groups(self, base_lr: float) -> Any:
        if not self.parameter_lr_multipliers:
            return self.policy.parameters()

        param_groups: dict[float, list[torch.nn.Parameter]] = {}
        matched_prefixes: set[str] = set()
        group_names: dict[float, set[str]] = {}
        for name, param in self.policy.named_parameters():
            lr_multiplier, prefix = self._resolve_parameter_lr_multiplier(name)
            param_groups.setdefault(lr_multiplier, []).append(param)
            group_names.setdefault(lr_multiplier, set()).add(prefix or "default")
            if prefix is not None:
                matched_prefixes.add(prefix)

        unmatched_prefixes = sorted(set(self.parameter_lr_multipliers) - matched_prefixes)
        if unmatched_prefixes:
            logger.warning(f"Parameter LR multiplier prefixes did not match any parameters: {unmatched_prefixes}")

        return [
            {
                "params": params,
                "lr": base_lr * lr_multiplier,
                "lr_multiplier": lr_multiplier,
                "name": "+".join(sorted(group_names[lr_multiplier])),
            }
            for lr_multiplier, params in sorted(
                param_groups.items(),
                key=lambda item: (item[0] != 1.0, item[0]),
            )
        ]

    def _resolve_parameter_lr_multiplier(self, parameter_name: str) -> tuple[float, str | None]:
        best_prefix: str | None = None
        best_multiplier = 1.0
        for prefix, multiplier in self.parameter_lr_multipliers.items():
            if self._parameter_name_matches_prefix(parameter_name, prefix) and (
                    best_prefix is None or len(prefix) > len(best_prefix)
            ):
                best_prefix = prefix
                best_multiplier = multiplier
        return best_multiplier, best_prefix

    @staticmethod
    def _parameter_name_matches_prefix(parameter_name: str, prefix: str) -> bool:
        name_parts = parameter_name.split(".")
        prefix_parts = prefix.split(".")
        if len(prefix_parts) > len(name_parts):
            return False
        for start_idx in range(len(name_parts) - len(prefix_parts) + 1):
            if name_parts[start_idx:start_idx + len(prefix_parts)] == prefix_parts:
                return True
        return False

    @staticmethod
    def _serialize_rollout_mode(rollout_mode: PPORolloutMode) -> dict[str, Any]:
        if isinstance(rollout_mode, WholeEpisodesRolloutMode):
            return {"mode": "whole_episodes", "n_episodes_per_rollout": rollout_mode.n_episodes_per_rollout}
        if isinstance(rollout_mode, StepsRolloutMode):
            return {"mode": "steps", "n_steps_per_rollout": rollout_mode.n_steps_per_rollout}
        return {"mode": type(rollout_mode).__name__}

    @staticmethod
    def _serialize_gsde_reset_mode(mode: GSDEResetMode | None) -> dict[str, Any] | None:
        if mode is None:
            return None
        if isinstance(mode, GSDEIntervalResetMode):
            return {"mode": "interval", "interval": mode.interval}
        if isinstance(mode, GSDEProbabilityResetMode):
            return {"mode": "probability", "probability": mode.probability}
        return {"mode": type(mode).__name__}

    @staticmethod
    def _resolve_rollout_buffer_max_episode_length(
            *,
            rollout_mode: PPORolloutMode,
            max_episode_length: int,
            n_envs: int,
    ) -> int:
        if max_episode_length <= 0:
            raise ValueError(f"max_episode_length must be > 0, got {max_episode_length}")
        if n_envs <= 0:
            raise ValueError(f"n_envs must be > 0, got {n_envs}")
        if isinstance(rollout_mode, WholeEpisodesRolloutMode):
            return max_episode_length
        if isinstance(rollout_mode, StepsRolloutMode):
            return min(max_episode_length, math.ceil(rollout_mode.n_steps_per_rollout / n_envs))
        raise TypeError(f"Unknown rollout_mode type: {type(rollout_mode)}")

    @staticmethod
    def _parse_indexed_float_params(
            params: str,
            *,
            idx_keys: tuple[str, ...],
            value_keys: tuple[str, ...],
    ) -> tuple[int, float]:
        parsed = None
        try:
            parsed = json.loads(params)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict):
            idx_key = next((key for key in idx_keys if key in parsed), None)
            value_key = next((key for key in value_keys if key in parsed), None)
            if idx_key is None or value_key is None:
                raise ValueError(
                    f"Expected JSON keys {idx_keys} and {value_keys}, got {tuple(parsed.keys())}"
                )
            return int(parsed[idx_key]), float(parsed[value_key])

        values = [part.strip() for part in params.split(",")]
        if len(values) != 2:
            raise ValueError(
                "Expected params as '<index>,<value>' or JSON object, "
                "e.g. set_act_ent_loss_coef:0,0.01 or "
                "set_act_ent_loss_coef:{\"act\":0,\"value\":0.01}"
            )
        return int(values[0]), float(values[1])
