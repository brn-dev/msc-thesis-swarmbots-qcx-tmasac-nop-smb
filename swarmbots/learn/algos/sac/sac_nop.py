from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayBatch,
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, NextObsPredMixin
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.serialization_utils import serialize_dataclass, serialize_value


class SACNOPLatentSource(Enum):
    CRITIC = "critic"
    ACTOR = "actor"
    BOTH = "both"
    SHARED_ENCODER = "shared_encoder"


@dataclass(frozen=True)
class SACNOPConfig:
    enabled: bool = False
    num_next_steps: int = 4
    latent_source: SACNOPLatentSource | str = SACNOPLatentSource.CRITIC
    skip_first_transition_for_critic: bool = True
    nop_latent_dim: int | None = None
    nop_loss_coef: float = 1.0
    compile_modules: bool = False
    compile_mode: str = "default"
    act_fn_cls: ActivationFactory = nn.GELU
    latent_projection_hidden_dims: list[int] | None = None
    pre_predictors_hidden_dims: list[int] | None = None
    scalar_predictor_hidden_dims: list[int] | None = None
    angle_predictor_hidden_dims: list[int] | None = None
    rot6d_predictor_hidden_dims: list[int] | None = None
    binary_predictor_hidden_dims: list[int] | None = None
    global_pool_hidden_dims: list[int] | None = None
    global_scalar_predictor_hidden_dims: list[int] | None = None
    global_rot6d_predictor_hidden_dims: list[int] | None = None
    latent_projection_init_gain: float = 1.0
    pre_predictors_init_gain: float = 1.0
    predictor_init_gain: float = 0.01
    transition_model_dropout: float = 0.0
    transition_model_d_model: int = 128
    transition_model_nhead: int = 4
    transition_model_num_layers: int = 2
    transition_model_dim_feedforward: int = 256
    transition_model_add_agent_embeddings: bool = False
    transition_model_predict_delta: bool = True
    transition_model_coembed_hidden_dims: list[int] | None = None
    transition_model_head_hidden_dims: list[int] | None = None
    transition_model_coembed_init_gain: float = 1.0
    transition_model_head_init_gain: float = 0.01
    transition_model_transformer_ff_init_gain: float | None = 1.0
    scalar_loss_fn: str | nn.Module | None = None
    next_obs_pred_config: NextObsPredConfig = field(default_factory=NextObsPredConfig)

    def __post_init__(self) -> None:
        if self.num_next_steps <= 0:
            raise ValueError(f"num_next_steps must be > 0, got {self.num_next_steps}")


@dataclass(frozen=True, slots=True)
class SACNOPSequenceBatch:
    """NOP origins use (B, O, ...) while future fields use (B, O, K, ...)."""

    local_obs: torch.Tensor
    global_obs: torch.Tensor | None
    agent_mask: torch.Tensor | None
    actions: torch.Tensor
    next_local_obs: torch.Tensor
    next_global_obs: torch.Tensor | None
    next_agent_mask: torch.Tensor | None
    train_mask: torch.Tensor


class SACNOPModule(nn.Module, NextObsPredMixin):
    def __init__(
            self,
            *,
            n_agents: int,
            source_latent_dim: int,
            action_dim: int,
            config: SACNOPConfig,
            name: str,
            skip_first_transition: bool = False,
    ) -> None:
        super().__init__()
        if source_latent_dim <= 0:
            raise ValueError(f"source_latent_dim must be > 0, got {source_latent_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if config.nop_loss_coef < 0:
            raise ValueError(f"nop_loss_coef must be >= 0, got {config.nop_loss_coef}")
        if config.nop_latent_dim is not None and config.nop_latent_dim <= 0:
            raise ValueError(f"nop_latent_dim must be > 0 when set, got {config.nop_latent_dim}")
        if not _has_next_obs_pred_targets(config.next_obs_pred_config):
            raise ValueError(
                "SACNOPConfig.enabled=True requires at least one next-observation prediction target."
            )

        self.name = name
        self.nop_loss_coef = float(config.nop_loss_coef)
        self.n_agents = int(n_agents)
        self.source_latent_dim = int(source_latent_dim)
        self.action_dim = int(action_dim)
        self.nop_latent_dim = self.source_latent_dim if config.nop_latent_dim is None else int(config.nop_latent_dim)
        self.latent_projection_hidden_dims = (
            [max(self.source_latent_dim, self.nop_latent_dim)]
            if config.latent_projection_hidden_dims is None
            else [*config.latent_projection_hidden_dims]
        )
        self.config = config

        latent_projection = self._build_projection(
            input_dim=self.source_latent_dim,
            output_dim=self.nop_latent_dim,
            hidden_dims=self.latent_projection_hidden_dims,
            act_fn_cls=config.act_fn_cls,
            init_gain=config.latent_projection_init_gain,
        )
        pre_predictors_transform = None
        predictor_input_dim = self.nop_latent_dim
        if config.pre_predictors_hidden_dims:
            pre_predictors_transform = MLP(
                input_dim=self.nop_latent_dim,
                hidden_dims=[*config.pre_predictors_hidden_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.pre_predictors_init_gain),
                act_fn_cls=config.act_fn_cls,
            )
            predictor_input_dim = config.pre_predictors_hidden_dims[-1]

        next_obs_config = config.next_obs_pred_config
        angle_output_multiplier = 1 if next_obs_config.predict_delta else 2
        rot6d_output_multiplier = 3 if next_obs_config.predict_delta else 6

        has_global_targets = (
            bool(next_obs_config.global_scalar_target_indices)
            or bool(next_obs_config.global_rot6d_target_indices)
        )
        global_pool_encoder = None
        global_predictor_input_dim = predictor_input_dim
        if has_global_targets and config.global_pool_hidden_dims:
            global_pool_encoder = MLP(
                input_dim=predictor_input_dim,
                hidden_dims=[*config.global_pool_hidden_dims],
                end_with_act_fn=True,
                linear_init=make_init_linear_orthogonal(config.pre_predictors_init_gain),
                act_fn_cls=config.act_fn_cls,
            )
            global_predictor_input_dim = config.global_pool_hidden_dims[-1]

        transition_model_config = TransformerTransitionModelConfig(
            n_agents=self.n_agents,
            latent_dim=self.nop_latent_dim,
            action_dim=self.action_dim,
            d_model=config.transition_model_d_model,
            nhead=config.transition_model_nhead,
            num_layers=config.transition_model_num_layers,
            dim_feedforward=config.transition_model_dim_feedforward,
            dropout=config.transition_model_dropout,
            act_fn_cls=config.act_fn_cls,
            add_agent_embeddings=config.transition_model_add_agent_embeddings,
            predict_delta=config.transition_model_predict_delta,
            coembed_mlp_hidden_dims=config.transition_model_coembed_hidden_dims,
            head_mlp_hidden_dims=config.transition_model_head_hidden_dims,
            coembed_init_gain=config.transition_model_coembed_init_gain,
            head_init_gain=config.transition_model_head_init_gain,
            transformer_ff_init_gain=config.transition_model_transformer_ff_init_gain,
        )
        self.setup_next_obs_pred(
            transition_model=TransformerTransitionModel(config=transition_model_config),
            config=next_obs_config,
            pre_transition_transform=latent_projection,
            pre_predictors_transform=pre_predictors_transform,
            scalar_loss_fn=self._resolve_scalar_loss_fn(config.scalar_loss_fn),
            local_scalars_predictor=self._build_predictor(
                input_dim=predictor_input_dim,
                output_dim=self._target_count(next_obs_config.local_scalar_target_indices),
                hidden_dims=config.scalar_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            local_angles_predictor=self._build_predictor(
                input_dim=predictor_input_dim,
                output_dim=self._target_count(next_obs_config.local_angle_target_indices) * angle_output_multiplier,
                hidden_dims=config.angle_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            local_rot6ds_predictor=self._build_predictor(
                input_dim=predictor_input_dim,
                output_dim=self._target_count(next_obs_config.local_rot6d_target_indices) * rot6d_output_multiplier,
                hidden_dims=config.rot6d_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            local_binaries_predictor=self._build_predictor(
                input_dim=predictor_input_dim,
                output_dim=self._target_count(next_obs_config.local_binary_target_indices),
                hidden_dims=config.binary_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            global_pool_encoder=global_pool_encoder,
            global_scalars_predictor=self._build_predictor(
                input_dim=global_predictor_input_dim,
                output_dim=self._target_count(next_obs_config.global_scalar_target_indices),
                hidden_dims=config.global_scalar_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            global_rot6ds_predictor=self._build_predictor(
                input_dim=global_predictor_input_dim,
                output_dim=self._target_count(next_obs_config.global_rot6d_target_indices) * rot6d_output_multiplier,
                hidden_dims=config.global_rot6d_predictor_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                init_gain=config.predictor_init_gain,
            ),
            skip_first_transition=skip_first_transition,
        )
        self._apply_optional_compile()

    def compute_loss(
            self,
            *,
            source_latents: torch.Tensor,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch | SACNOPSequenceBatch,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        next_obs_pred_loss, metrics = self.compute_next_obs_pred_loss(
            local_latents=source_latents,
            next_local_obs=batch.next_local_obs,
            actions=batch.actions,
            local_obs=batch.local_obs,
            next_global_obs=batch.next_global_obs if self.has_global_next_obs_pred_targets else None,
            global_obs=batch.global_obs if self.has_global_next_obs_pred_targets else None,
            agent_mask=batch.agent_mask,
            loss_agent_mask=batch.next_agent_mask,
            time_mask=(
                batch.train_mask
                if isinstance(batch, (OffPolicyReplayEpisodeSegmentBatch, SACNOPSequenceBatch))
                else None
            ),
        )
        scaled_loss = self.nop_loss_coef * next_obs_pred_loss
        prefixed_metrics = {f"{self.name}_nop_{key}": value for key, value in metrics.items()}
        prefixed_metrics[f"{self.name}_nop_loss"] = next_obs_pred_loss.item()
        prefixed_metrics[f"{self.name}_nop_loss_scaled"] = scaled_loss.item()
        return scaled_loss, prefixed_metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_agents": self.n_agents,
            "source_latent_dim": self.source_latent_dim,
            "action_dim": self.action_dim,
            "nop_latent_dim": self.nop_latent_dim,
            "nop_loss_coef": self.nop_loss_coef,
            "skip_first_transition": self.skip_first_transition,
            "compile_modules": self.config.compile_modules,
            "compile_mode": self.config.compile_mode,
            "act_fn_cls": serialize_value(self.config.act_fn_cls),
            "latent_projection_hidden_dims": [*self.latent_projection_hidden_dims],
            "pre_predictors_hidden_dims": self._copy_optional_list(self.config.pre_predictors_hidden_dims),
            "scalar_predictor_hidden_dims": self._copy_optional_list(self.config.scalar_predictor_hidden_dims),
            "angle_predictor_hidden_dims": self._copy_optional_list(self.config.angle_predictor_hidden_dims),
            "rot6d_predictor_hidden_dims": self._copy_optional_list(self.config.rot6d_predictor_hidden_dims),
            "binary_predictor_hidden_dims": self._copy_optional_list(self.config.binary_predictor_hidden_dims),
            "global_pool_hidden_dims": self._copy_optional_list(self.config.global_pool_hidden_dims),
            "global_scalar_predictor_hidden_dims": self._copy_optional_list(
                self.config.global_scalar_predictor_hidden_dims
            ),
            "global_rot6d_predictor_hidden_dims": self._copy_optional_list(
                self.config.global_rot6d_predictor_hidden_dims
            ),
            "latent_projection_init_gain": self.config.latent_projection_init_gain,
            "pre_predictors_init_gain": self.config.pre_predictors_init_gain,
            "predictor_init_gain": self.config.predictor_init_gain,
            "scalar_loss_fn": serialize_value(self.scalar_loss_fn),
            "next_obs_pred": self.get_next_obs_pred_hyper_parameters(
                pre_transition_dims=self.latent_projection_hidden_dims,
                pre_predictors_dims=self.config.pre_predictors_hidden_dims,
                scalar_predictor_hidden_dims=self.config.scalar_predictor_hidden_dims,
                angle_predictor_hidden_dims=self.config.angle_predictor_hidden_dims,
                rot6d_predictor_hidden_dims=self.config.rot6d_predictor_hidden_dims,
                binary_predictor_hidden_dims=self.config.binary_predictor_hidden_dims,
                global_pool_hidden_dims=self.config.global_pool_hidden_dims,
                global_scalar_predictor_hidden_dims=self.config.global_scalar_predictor_hidden_dims,
                global_rot6d_predictor_hidden_dims=self.config.global_rot6d_predictor_hidden_dims,
            ),
            "config": serialize_dataclass(self.config),
        }

    def _apply_optional_compile(self) -> None:
        if not self.config.compile_modules:
            return
        if not hasattr(torch, "compile") or not callable(torch.compile):
            raise RuntimeError("SACNOPConfig.compile_modules=True requires torch.compile support.")
        if not self.config.compile_mode:
            raise ValueError("SACNOPConfig.compile_mode must be a non-empty string when compile_modules=True.")

        self._compute_next_obs_pred_loss_fn = torch.compile(
            self._compute_next_obs_pred_loss_impl,
            mode=self.config.compile_mode,
            fullgraph=True,
            dynamic=False,
        )

    @staticmethod
    def _build_projection(
            *,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | None,
            act_fn_cls: ActivationFactory,
            init_gain: float,
    ) -> nn.Module:
        return MLP(
            input_dim=input_dim,
            hidden_dims=[*(hidden_dims or []), output_dim],
            end_with_act_fn=True,
            linear_init=make_init_linear_orthogonal(init_gain),
            act_fn_cls=act_fn_cls,
        )

    @staticmethod
    def _build_predictor(
            *,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | None,
            act_fn_cls: ActivationFactory,
            init_gain: float,
    ) -> nn.Module | None:
        if output_dim <= 0:
            return None
        return MLP(
            input_dim=input_dim,
            hidden_dims=[*(hidden_dims or []), output_dim],
            end_with_act_fn=False,
            linear_init=make_init_linear_orthogonal(init_gain),
            act_fn_cls=act_fn_cls,
        )

    @staticmethod
    def _target_count(indices: list[int] | None) -> int:
        return 0 if indices is None else len(indices)

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


def normalize_nop_latent_source(source: SACNOPLatentSource | str) -> SACNOPLatentSource:
    if isinstance(source, SACNOPLatentSource):
        return source
    try:
        return SACNOPLatentSource(source.lower())
    except ValueError as exc:
        valid = [item.value for item in SACNOPLatentSource]
        raise ValueError(f"Unknown NOP latent source {source!r}; expected one of {valid}") from exc


def _has_next_obs_pred_targets(config: NextObsPredConfig) -> bool:
    return any((
        bool(config.local_scalar_target_indices),
        bool(config.local_angle_target_indices),
        bool(config.local_rot6d_target_indices),
        bool(config.local_binary_target_indices),
        bool(config.global_scalar_target_indices),
        bool(config.global_rot6d_target_indices),
    ))
