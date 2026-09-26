from collections.abc import Callable
from dataclasses import dataclass, field
import shutil
import sys
from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy, DelegatingPPOPolicyTemporalStateMixin
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import (
    PPOWMBatchSampler,
    PPOWMSampler,
    PPOWMSamples,
    PPOWMSamplerConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, NextObsPredMixin
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.algos.world_modeling.wm_recurrent_batch import build_wm_target_time_mask
from swarmbots.learn.losses import LossDict, LossMetrics
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


@dataclass(frozen=True)
class NOPWorldModelConfig:
    n_agents: int
    local_latent_dim: int
    action_dim: int
    world_model_loss_coef: float = 1.0
    compile_modules: bool = False
    compile_mode: str = "default"
    act_fn_cls: ActivationFactory = nn.GELU
    transition_model_dropout: float = 0.0
    d_model_transition_model: int = 128
    nhead_transition_model: int = 4
    num_layers_transition_model: int = 2
    dim_feedforward_transition_model: int = 256
    add_agent_embeddings_transition_model: bool = False
    transition_model_coembed_hidden_dims: list[int] | None = None
    transition_model_head_hidden_dims: list[int] | None = None
    transition_model_predict_delta: bool = True
    wm_pre_transition_dims: list[int] | None = None
    wm_pre_predictors_dims: list[int] | None = None
    wm_scalar_predictor_hidden_dims: list[int] | None = None
    wm_angle_predictor_hidden_dims: list[int] | None = None
    wm_rot6d_predictor_hidden_dims: list[int] | None = None
    wm_binary_predictor_hidden_dims: list[int] | None = None
    wm_global_pool_hidden_dims: list[int] | None = None
    wm_global_scalar_predictor_hidden_dims: list[int] | None = None
    wm_global_rot6d_predictor_hidden_dims: list[int] | None = None
    wm_pre_transition_init_gain: float = 1.0
    wm_pre_predictors_init_gain: float = 1.0
    wm_predictor_init_gain: float = 0.01
    transition_model_coembed_init_gain: float = 1.0
    transition_model_head_init_gain: float = 0.01
    transition_model_transformer_ff_init_gain: float | None = 1.0
    scalar_loss_fn: str | nn.Module | None = None
    next_obs_pred_config: NextObsPredConfig = field(default_factory=NextObsPredConfig)


class NextObsPredWrapper(
    DelegatingPPOPolicyTemporalStateMixin,
    BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig],
    NextObsPredMixin,
):

    def __init__(
            self,
            policy: BasePPOPolicy[PPOSamples, PPOSamplerConfig],
            *,
            world_model_config: NOPWorldModelConfig,
    ) -> None:
        super().__init__()
        if world_model_config.n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {world_model_config.n_agents}")
        if world_model_config.local_latent_dim < 1:
            raise ValueError(f"local_latent_dim must be >= 1, got {world_model_config.local_latent_dim}")
        if world_model_config.action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {world_model_config.action_dim}")
        if world_model_config.world_model_loss_coef < 0:
            raise ValueError(f"world_model_loss_coef must be >= 0, got {world_model_config.world_model_loss_coef}")
        if not (0.0 <= world_model_config.transition_model_dropout < 1.0):
            raise ValueError(
                "transition_model_dropout must be in [0, 1), "
                f"got {world_model_config.transition_model_dropout}"
            )
        if world_model_config.compile_modules:
            _ensure_torch_compile_available(compile_mode=world_model_config.compile_mode)
        self.policy = policy
        self._setup_world_model_from_config(world_model_config)

    @property
    def action_dist(self) -> HybridActionDistribution:
        return self.policy.action_dist

    @property
    def has_popart(self) -> bool:
        return self.policy.has_popart

    def update_value_normalizer(self, targets: torch.Tensor) -> None:
        self.policy.update_value_normalizer(targets)

    def normalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return self.policy.normalize_values(values)

    def get_value_normalizer_metrics(self) -> dict[str, float]:
        return self.policy.get_value_normalizer_metrics()

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )

    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.policy.predict_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
        )

    def _evaluate_actions(
            self,
            batch: PPOWMSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics, torch.Tensor]:
        return self.policy._evaluate_actions(
            batch=batch,
            action_splitter=action_splitter,
        )

    def evaluate_actions(
            self,
            batch: PPOWMSamples,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[torch.Tensor, torch.Tensor, LossDict, LossMetrics]:
        log_probs, values, extra_losses, extra_loss_metrics, local_latents = self._evaluate_actions(
            batch=batch,
            action_splitter=action_splitter,
        )
        wm_target_time_mask = build_wm_target_time_mask(
            wm_target_time_mask=batch.wm_target_time_mask,
            sequence_time_mask=getattr(batch, "time_mask", None),
        )
        next_obs_pred_kwargs = {
            "local_latents": local_latents,
            "next_local_obs": batch.next_local_obs,
            "actions": batch.wm_actions,
            "local_obs": batch.local_obs,
            "agent_mask": batch.wm_agent_mask,
            "loss_agent_mask": batch.wm_loss_agent_mask,
            "time_mask": wm_target_time_mask,
        }
        if self.has_global_next_obs_pred_targets:
            next_obs_pred_kwargs["next_global_obs"] = batch.next_global_obs
            next_obs_pred_kwargs["global_obs"] = batch.global_obs
        next_obs_pred_loss, nop_loss_metrics = self.compute_next_obs_pred_loss(**next_obs_pred_kwargs)
        world_model_loss_scaled = self.world_model_loss_coef * next_obs_pred_loss
        merged_extra_losses = dict(extra_losses)
        merged_extra_losses["world_model"] = world_model_loss_scaled
        merged_extra_loss_metrics = dict(extra_loss_metrics)
        merged_extra_loss_metrics.update(nop_loss_metrics)
        merged_extra_loss_metrics["wm_loss"] = next_obs_pred_loss.item()
        merged_extra_loss_metrics["wm_loss_scaled"] = world_model_loss_scaled.item()
        return log_probs, values, merged_extra_losses, merged_extra_loss_metrics

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOWMSamplerConfig,
    ) -> BaseWMSampler[Any, PPOWMSamplerConfig]:
        if type(config) is PPOWMSamplerConfig:
            return PPOWMSampler(
                episodes=episodes,
                config=config,
                requires_previous_actions=self.requires_previous_actions(),
            )
        sampler = self.policy.make_sampler(episodes=episodes, config=config)
        if not isinstance(sampler, BaseWMSampler):
            raise ValueError('Policy does not create a WM Sampler!')
        return sampler

    def supports_rollout_batch_sampler(self, config: PPOWMSamplerConfig) -> bool:
        if type(config) is PPOWMSamplerConfig:
            return True
        return self.policy.supports_rollout_batch_sampler(config)

    def make_rollout_batch_sampler(
            self,
            rollout_batch: PPORolloutBatch,
            config: PPOWMSamplerConfig,
    ) -> BaseWMSampler[Any, PPOWMSamplerConfig]:
        if type(config) is PPOWMSamplerConfig:
            return PPOWMBatchSampler(
                rollout_batch=rollout_batch,
                config=config,
                requires_previous_actions=self.requires_previous_actions(),
            )
        sampler = self.policy.make_rollout_batch_sampler(rollout_batch=rollout_batch, config=config)
        if not isinstance(sampler, BaseWMSampler):
            raise ValueError("Policy does not create a WM Sampler!")
        return sampler

    def after_optimizer_step(self) -> None:
        self.policy.after_optimizer_step()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **self.policy.get_hyper_parameters(),
            "next_obs_pred_wrapper": {
                "world_model_loss_coef": self.world_model_loss_coef,
                "compile_modules": self._wm_compile_modules,
                "compile_mode": self._wm_compile_mode,
                "n_agents": self._wm_n_agents,
                "local_latent_dim": self._wm_local_latent_dim,
                "action_dim": self._wm_action_dim,
                "act_fn_cls": str(self._wm_act_fn_cls),
                "transition_model_dropout": self._wm_transition_model_dropout,
                "world_model_config": self.get_next_obs_pred_hyper_parameters(
                    pre_transition_dims=self._wm_pre_transition_dims,
                    pre_predictors_dims=self._wm_pre_predictors_dims,
                    scalar_predictor_hidden_dims=self._wm_scalar_predictor_hidden_dims,
                    angle_predictor_hidden_dims=self._wm_angle_predictor_hidden_dims,
                    rot6d_predictor_hidden_dims=self._wm_rot6d_predictor_hidden_dims,
                    binary_predictor_hidden_dims=self._wm_binary_predictor_hidden_dims,
                    global_pool_hidden_dims=self._wm_global_pool_hidden_dims,
                    global_scalar_predictor_hidden_dims=self._wm_global_scalar_predictor_hidden_dims,
                    global_rot6d_predictor_hidden_dims=self._wm_global_rot6d_predictor_hidden_dims,
                ),
                "wm_pre_transition_init_gain": self._wm_pre_transition_init_gain,
                "wm_pre_predictors_init_gain": self._wm_pre_predictors_init_gain,
                "wm_predictor_init_gain": self._wm_predictor_init_gain,
            },
        }

    def get_grad_norms(self) -> dict[str, float]:
        grad_norms = self.policy.get_grad_norms()
        grad_norms.update(
            {
                "wm_pre_transition_transform": self._module_grad_norm(self.pre_transition_transform),
                "wm_transition_model": self._module_grad_norm(self.transition_model),
                "wm_pre_predictors_transform": self._module_grad_norm(self.pre_predictors_transform),
                "wm_local_scalars_predictor": self._module_grad_norm(self.local_scalars_predictor),
                "wm_local_angles_predictor": self._module_grad_norm(self.local_angles_predictor),
                "wm_local_rot6ds_predictor": self._module_grad_norm(self.local_rot6ds_predictor),
                "wm_local_binaries_predictor": self._module_grad_norm(self.local_binaries_predictor),
                "wm_global_pool_encoder": self._module_grad_norm(self.global_pool_encoder),
                "wm_global_scalars_predictor": self._module_grad_norm(self.global_scalars_predictor),
                "wm_global_rot6ds_predictor": self._module_grad_norm(self.global_rot6ds_predictor),
            }
        )
        return grad_norms

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        return self.policy.act(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
        )

    def requires_previous_actions(self) -> bool:
        return self.policy.requires_previous_actions()

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)

        scalar_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("scalar_loss_weight", "scalar"),
        )
        if scalar_weight is not None:
            alias, value = scalar_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.scalar_loss_weight = value

        angle_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("angle_loss_weight", "angle"),
        )
        if angle_weight is not None:
            alias, value = angle_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.angle_loss_weight = value

        rot6d_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("rot6d_loss_weight", "rot6d"),
        )
        if rot6d_weight is not None:
            alias, value = rot6d_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.rot6d_loss_weight = value

        binary_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("binary_loss_weight", "binary"),
        )
        if binary_weight is not None:
            alias, value = binary_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.binary_loss_weight = value

        global_scalar_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("global_scalar_loss_weight", "global_scalar"),
        )
        if global_scalar_weight is not None:
            alias, value = global_scalar_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.global_scalar_loss_weight = value

        global_rot6d_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("global_rot6d_loss_weight", "global_rot6d"),
        )
        if global_rot6d_weight is not None:
            alias, value = global_rot6d_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.global_rot6d_loss_weight = value

        world_model_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("world_model_loss_coef", "wm_loss_coef"),
        )
        if world_model_weight is not None:
            alias, value = world_model_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.world_model_loss_coef = value

        self.policy.update_loss_weights(**remaining_weights)

    def _setup_world_model_from_config(self, world_model_config: NOPWorldModelConfig) -> None:
        next_obs_pred_config = world_model_config.next_obs_pred_config
        self.world_model_loss_coef = float(world_model_config.world_model_loss_coef)
        self._wm_compile_modules = world_model_config.compile_modules
        self._wm_compile_mode = world_model_config.compile_mode
        self._wm_n_agents = int(world_model_config.n_agents)
        self._wm_local_latent_dim = int(world_model_config.local_latent_dim)
        self._wm_action_dim = int(world_model_config.action_dim)
        self._wm_act_fn_cls = world_model_config.act_fn_cls
        self._wm_transition_model_dropout = float(world_model_config.transition_model_dropout)
        self._wm_pre_transition_dims = self._copy_optional_list(world_model_config.wm_pre_transition_dims)
        self._wm_pre_predictors_dims = self._copy_optional_list(world_model_config.wm_pre_predictors_dims)
        self._wm_pre_transition_init_gain = float(world_model_config.wm_pre_transition_init_gain)
        self._wm_pre_predictors_init_gain = float(world_model_config.wm_pre_predictors_init_gain)
        self._wm_predictor_init_gain = float(world_model_config.wm_predictor_init_gain)
        self._wm_scalar_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_scalar_predictor_hidden_dims
        )
        self._wm_angle_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_angle_predictor_hidden_dims
        )
        self._wm_rot6d_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_rot6d_predictor_hidden_dims
        )
        self._wm_binary_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_binary_predictor_hidden_dims
        )
        self._wm_global_pool_hidden_dims = self._copy_optional_list(world_model_config.wm_global_pool_hidden_dims)
        self._wm_global_scalar_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_global_scalar_predictor_hidden_dims
        )
        self._wm_global_rot6d_predictor_hidden_dims = self._copy_optional_list(
            world_model_config.wm_global_rot6d_predictor_hidden_dims
        )

        scalar_loss_fn = self._resolve_scalar_loss_fn(world_model_config.scalar_loss_fn)

        pre_transition_transform = None
        wm_latent_dim = self._wm_local_latent_dim
        if world_model_config.wm_pre_transition_dims:
            pre_transition_transform = MLP(
                input_dim=self._wm_local_latent_dim,
                hidden_dims=[*world_model_config.wm_pre_transition_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(world_model_config.wm_pre_transition_init_gain),
                act_fn_cls=self._wm_act_fn_cls,
            )
            wm_latent_dim = world_model_config.wm_pre_transition_dims[-1]

        pre_predictors_transform = None
        wm_pre_predictors_dim = wm_latent_dim
        if world_model_config.wm_pre_predictors_dims:
            pre_predictors_transform = MLP(
                input_dim=wm_latent_dim,
                hidden_dims=[*world_model_config.wm_pre_predictors_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(world_model_config.wm_pre_predictors_init_gain),
                act_fn_cls=self._wm_act_fn_cls,
            )
            wm_pre_predictors_dim = world_model_config.wm_pre_predictors_dims[-1]

        angle_output_multiplier = 1 if next_obs_pred_config.predict_delta else 2
        rot6d_output_multiplier = 3 if next_obs_pred_config.predict_delta else 6

        local_scalars_predictor = _build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_scalar_target_indices)
                if next_obs_pred_config.local_scalar_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_scalar_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )
        local_angles_predictor = _build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_angle_target_indices) * angle_output_multiplier
                if next_obs_pred_config.local_angle_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_angle_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )
        local_rot6ds_predictor = _build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_rot6d_target_indices) * rot6d_output_multiplier
                if next_obs_pred_config.local_rot6d_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_rot6d_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )
        local_binaries_predictor = _build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_binary_target_indices)
                if next_obs_pred_config.local_binary_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_binary_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )
        has_global_targets = (
            bool(next_obs_pred_config.global_scalar_target_indices)
            or bool(next_obs_pred_config.global_rot6d_target_indices)
        )
        global_pool_encoder = None
        global_predictor_input_dim = wm_pre_predictors_dim
        if has_global_targets and world_model_config.wm_global_pool_hidden_dims:
            global_pool_encoder = MLP(
                input_dim=wm_pre_predictors_dim,
                hidden_dims=[*world_model_config.wm_global_pool_hidden_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(world_model_config.wm_pre_predictors_init_gain),
                act_fn_cls=self._wm_act_fn_cls,
            )
            global_predictor_input_dim = world_model_config.wm_global_pool_hidden_dims[-1]
        global_scalars_predictor = _build_predictor(
            input_dim=global_predictor_input_dim,
            output_dim=(
                len(next_obs_pred_config.global_scalar_target_indices)
                if next_obs_pred_config.global_scalar_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_global_scalar_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )
        global_rot6ds_predictor = _build_predictor(
            input_dim=global_predictor_input_dim,
            output_dim=(
                len(next_obs_pred_config.global_rot6d_target_indices) * rot6d_output_multiplier
                if next_obs_pred_config.global_rot6d_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_global_rot6d_predictor_hidden_dims,
            act_fn_cls=self._wm_act_fn_cls,
            linear_init_gain=world_model_config.wm_predictor_init_gain,
        )

        transition_model_config = TransformerTransitionModelConfig(
            n_agents=self._wm_n_agents,
            latent_dim=wm_latent_dim,
            action_dim=self._wm_action_dim,
            d_model=world_model_config.d_model_transition_model,
            nhead=world_model_config.nhead_transition_model,
            num_layers=world_model_config.num_layers_transition_model,
            dim_feedforward=world_model_config.dim_feedforward_transition_model,
            dropout=self._wm_transition_model_dropout,
            act_fn_cls=self._wm_act_fn_cls,
            add_agent_embeddings=world_model_config.add_agent_embeddings_transition_model,
            predict_delta=world_model_config.transition_model_predict_delta,
            coembed_mlp_hidden_dims=world_model_config.transition_model_coembed_hidden_dims,
            head_mlp_hidden_dims=world_model_config.transition_model_head_hidden_dims,
            coembed_init_gain=world_model_config.transition_model_coembed_init_gain,
            head_init_gain=world_model_config.transition_model_head_init_gain,
            transformer_ff_init_gain=world_model_config.transition_model_transformer_ff_init_gain,
        )
        self.setup_next_obs_pred(
            transition_model=TransformerTransitionModel(config=transition_model_config),
            config=next_obs_pred_config,
            pre_transition_transform=pre_transition_transform,
            pre_predictors_transform=pre_predictors_transform,
            scalar_loss_fn=scalar_loss_fn,
            local_scalars_predictor=local_scalars_predictor,
            local_angles_predictor=local_angles_predictor,
            local_rot6ds_predictor=local_rot6ds_predictor,
            local_binaries_predictor=local_binaries_predictor,
            global_pool_encoder=global_pool_encoder,
            global_scalars_predictor=global_scalars_predictor,
            global_rot6ds_predictor=global_rot6ds_predictor,
        )
        self._apply_optional_compile()

    def _apply_optional_compile(self) -> None:
        if not self._wm_compile_modules:
            return

        self.pre_transition_transform = self._compile_module(self.pre_transition_transform)
        self.transition_model = self._compile_module(self.transition_model)
        self.pre_predictors_transform = self._compile_module(self.pre_predictors_transform)
        if self.local_scalars_predictor is not None:
            self.local_scalars_predictor = self._compile_module(self.local_scalars_predictor)
        if self.local_angles_predictor is not None:
            self.local_angles_predictor = self._compile_module(self.local_angles_predictor)
        if self.local_rot6ds_predictor is not None:
            self.local_rot6ds_predictor = self._compile_module(self.local_rot6ds_predictor)
        if self.local_binaries_predictor is not None:
            self.local_binaries_predictor = self._compile_module(self.local_binaries_predictor)
        if self.global_pool_encoder is not None:
            self.global_pool_encoder = self._compile_module(self.global_pool_encoder)
        if self.global_scalars_predictor is not None:
            self.global_scalars_predictor = self._compile_module(self.global_scalars_predictor)
        if self.global_rot6ds_predictor is not None:
            self.global_rot6ds_predictor = self._compile_module(self.global_rot6ds_predictor)
        self._compute_next_obs_pred_loss_fn = self._compile_callable(self._compute_next_obs_pred_loss_impl)

    def _compile_module(
            self,
            module: nn.Module,
    ) -> nn.Module:
        if isinstance(module, nn.Identity):
            return module
        return torch.compile(
            module,
            mode=self._wm_compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    def _compile_callable(
            self,
            fn: Callable[..., Any],
    ) -> Callable[..., Any]:
        return torch.compile(
            fn,
            mode=self._wm_compile_mode,
            fullgraph=False,
            dynamic=False,
        )

    @staticmethod
    def _resolve_scalar_loss_fn(scalar_loss_fn: str | nn.Module | None) -> nn.Module:
        if isinstance(scalar_loss_fn, nn.Module):
            return scalar_loss_fn
        scalar_loss_name = (scalar_loss_fn or "mse").lower()
        if scalar_loss_name == "mse":
            return nn.MSELoss(reduction="none")
        if scalar_loss_name == "smooth_l1":
            return nn.SmoothL1Loss(reduction="none")
        raise ValueError(f"Unknown scalar_loss {scalar_loss_name!r}")


def _build_predictor(
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None,
        act_fn_cls: ActivationFactory,
        linear_init_gain: float = 0.01,
) -> nn.Module | None:
    if output_dim <= 0:
        return None
    return MLP(
        input_dim=input_dim,
        hidden_dims=[*(hidden_dims or []), output_dim] if hidden_dims else [output_dim],
        end_with_act_fn=False,
        linear_init=make_init_linear_orthogonal(linear_init_gain),
        act_fn_cls=act_fn_cls,
    )


def _ensure_torch_compile_available(*, compile_mode: str) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("NOPWorldModelConfig.compile_modules=True requires torch.compile support.")
    if sys.platform == "win32" and shutil.which("cl") is None:
        raise RuntimeError(
            "NOPWorldModelConfig.compile_modules=True on this Windows setup requires cl.exe on PATH for torch.compile."
        )
    if not compile_mode:
        raise ValueError("NOPWorldModelConfig.compile_mode must be a non-empty string when compile_modules=True.")
