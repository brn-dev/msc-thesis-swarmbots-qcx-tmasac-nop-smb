from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

import torch
from torch import nn
from torch.nn import functional as F

from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
)
from swarmbots.learn.algos.world_modeling.wm_recurrent_batch import flatten_recurrent_wm_batch
from swarmbots.learn.masking import build_valid_mask, masked_mean, restrict_loss_agent_mask
from swarmbots.learn.serialization_utils import serialize_value

class PredictDeltaMode(Enum):
    PER_STEP_DELTA = 1
    INITIAL_DELTA = 2


@dataclass(frozen=True)
class NextObsPredConfig:
    local_scalar_target_indices: Optional[list[int]] = None
    local_angle_target_indices: Optional[list[int]] = None
    local_rot6d_target_indices: Optional[list[int]] = None
    local_binary_target_indices: Optional[list[int]] = None
    global_scalar_target_indices: Optional[list[int]] = None
    global_rot6d_target_indices: Optional[list[int]] = None
    scalar_loss_weight: float = 1.0
    angle_loss_weight: float = 1.0
    rot6d_loss_weight: float = 1.0
    binary_loss_weight: float = 1.0
    global_scalar_loss_weight: float = 1.0
    global_rot6d_loss_weight: float = 1.0
    binary_target_ema_decay: float = 0.99
    binary_target_ema_eps: float = 1e-4
    predict_delta: Optional[PredictDeltaMode | bool] = PredictDeltaMode.PER_STEP_DELTA


class NextObsPredMixin(abc.ABC):
    pre_transition_transform: nn.Module
    transition_model: TransformerTransitionModel
    _compute_next_obs_pred_loss_fn: Callable[..., tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]]

    local_scalar_target_indices: Optional[list[int]]
    local_angle_target_sin_indices: Optional[list[int]]
    local_angle_target_cos_indices: Optional[list[int]]
    local_rot6d_target_indices: Optional[list[int]]
    local_binary_target_indices: Optional[list[int]]
    global_scalar_target_indices: Optional[list[int]]
    global_rot6d_target_indices: Optional[list[int]]

    scalar_loss_fn: Optional[nn.Module]

    local_scalars_predictor: Optional[nn.Module]
    local_angles_predictor: Optional[nn.Module]
    local_rot6ds_predictor: Optional[nn.Module]
    local_binaries_predictor: Optional[nn.Module]
    global_pool_encoder: Optional[nn.Module]
    global_scalars_predictor: Optional[nn.Module]
    global_rot6ds_predictor: Optional[nn.Module]

    scalar_loss_weight: float
    angle_loss_weight: float
    rot6d_loss_weight: float
    binary_loss_weight: float
    global_scalar_loss_weight: float
    global_rot6d_loss_weight: float

    predict_delta: bool
    predict_delta_mode: PredictDeltaMode | None
    skip_first_transition: bool

    binary_target_ema: Optional[torch.Tensor]
    binary_target_ema_decay: float
    binary_target_ema_eps: float

    def setup_next_obs_pred(
            self,
            transition_model: TransformerTransitionModel,
            config: NextObsPredConfig,
            pre_transition_transform: Optional[nn.Module] = None,
            pre_predictors_transform: Optional[nn.Module] = None,
            scalar_loss_fn: Optional[nn.Module] = None,
            local_scalars_predictor: Optional[nn.Module] = None,
            local_angles_predictor: Optional[nn.Module] = None,
            local_rot6ds_predictor: Optional[nn.Module] = None,
            local_binaries_predictor: Optional[nn.Module] = None,
            global_pool_encoder: Optional[nn.Module] = None,
            global_scalars_predictor: Optional[nn.Module] = None,
            global_rot6ds_predictor: Optional[nn.Module] = None,
            skip_first_transition: bool = False,
    ) -> None:
        local_scalar_target_indices = self._normalize_indices(config.local_scalar_target_indices)
        local_angle_target_indices = self._normalize_indices(config.local_angle_target_indices)
        local_rot6d_target_indices = self._normalize_indices(config.local_rot6d_target_indices)
        local_binary_target_indices = self._normalize_indices(config.local_binary_target_indices)
        global_scalar_target_indices = self._normalize_indices(config.global_scalar_target_indices)
        global_rot6d_target_indices = self._normalize_indices(config.global_rot6d_target_indices)
        local_angle_target_cos_indices = (
            [i + 1 for i in local_angle_target_indices] if local_angle_target_indices is not None else None
        )

        if local_scalar_target_indices is not None and scalar_loss_fn is None:
            raise ValueError("scalar_loss_fn is required when local_scalar_target_indices is provided")
        if local_scalar_target_indices is not None and local_scalars_predictor is None:
            raise ValueError("local_scalars_predictor is required when local_scalar_target_indices is provided")
        if local_angle_target_indices is not None and local_angles_predictor is None:
            raise ValueError("local_angles_predictor is required when local_angle_target_indices is provided")
        if local_rot6d_target_indices is not None and local_rot6ds_predictor is None:
            raise ValueError("local_rot6ds_predictor is required when local_rot6d_target_indices is provided")
        if local_binary_target_indices is not None and local_binaries_predictor is None:
            raise ValueError("local_binaries_predictor is required when local_binary_target_indices is provided")
        if global_scalar_target_indices is not None and scalar_loss_fn is None:
            raise ValueError("scalar_loss_fn is required when global_scalar_target_indices is provided")
        if global_scalar_target_indices is not None and global_scalars_predictor is None:
            raise ValueError("global_scalars_predictor is required when global_scalar_target_indices is provided")
        if global_rot6d_target_indices is not None and global_rot6ds_predictor is None:
            raise ValueError("global_rot6ds_predictor is required when global_rot6d_target_indices is provided")

        self.pre_transition_transform = (
            pre_transition_transform if pre_transition_transform is not None else nn.Identity()
        )
        self.transition_model = transition_model
        self.skip_first_transition = bool(skip_first_transition)
        self.pre_predictors_transform = (
            pre_predictors_transform if pre_predictors_transform is not None else nn.Identity()
        )

        self.local_scalar_target_indices = local_scalar_target_indices
        self.local_angle_target_sin_indices = local_angle_target_indices
        self.local_angle_target_cos_indices = local_angle_target_cos_indices
        self.local_rot6d_target_indices = local_rot6d_target_indices
        self.local_binary_target_indices = local_binary_target_indices
        self.global_scalar_target_indices = global_scalar_target_indices
        self.global_rot6d_target_indices = global_rot6d_target_indices

        self.scalar_loss_fn = scalar_loss_fn

        self.local_scalars_predictor = local_scalars_predictor
        self.local_angles_predictor = local_angles_predictor
        self.local_rot6ds_predictor = local_rot6ds_predictor
        self.local_binaries_predictor = local_binaries_predictor
        self.global_pool_encoder = global_pool_encoder
        self.global_scalars_predictor = global_scalars_predictor
        self.global_rot6ds_predictor = global_rot6ds_predictor

        self.scalar_loss_weight = config.scalar_loss_weight
        self.angle_loss_weight = config.angle_loss_weight
        self.rot6d_loss_weight = config.rot6d_loss_weight
        self.binary_loss_weight = config.binary_loss_weight
        self.global_scalar_loss_weight = config.global_scalar_loss_weight
        self.global_rot6d_loss_weight = config.global_rot6d_loss_weight
        
        self.predict_delta_mode = self._normalize_predict_delta_mode(config.predict_delta)
        self.predict_delta = self.predict_delta_mode is not None

        self.binary_target_ema_decay = config.binary_target_ema_decay
        self.binary_target_ema_eps = config.binary_target_ema_eps

        if local_binary_target_indices is not None:
            ema = torch.full((len(local_binary_target_indices),), 0.5)
            if hasattr(self, "register_buffer"):
                self.register_buffer("binary_target_ema", ema)
            else:
                self.binary_target_ema = ema
        else:
            self.binary_target_ema = None
        self._compute_next_obs_pred_loss_fn = self._compute_next_obs_pred_loss_impl

    def get_next_obs_pred_hyper_parameters(
            self,
            *,
            pre_transition_dims: Optional[list[int]] = None,
            pre_predictors_dims: Optional[list[int]] = None,
            scalar_predictor_hidden_dims: Optional[list[int]] = None,
            angle_predictor_hidden_dims: Optional[list[int]] = None,
            rot6d_predictor_hidden_dims: Optional[list[int]] = None,
            binary_predictor_hidden_dims: Optional[list[int]] = None,
            global_pool_hidden_dims: Optional[list[int]] = None,
            global_scalar_predictor_hidden_dims: Optional[list[int]] = None,
            global_rot6d_predictor_hidden_dims: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        return {
            "transition_model_config": self.transition_model.get_hyper_parameters(),
            "wm_pre_transition_dims": self._copy_optional_list(pre_transition_dims),
            "wm_pre_predictors_dims": self._copy_optional_list(pre_predictors_dims),
            "wm_scalar_predictor_hidden_dims": self._copy_optional_list(scalar_predictor_hidden_dims),
            "wm_angle_predictor_hidden_dims": self._copy_optional_list(angle_predictor_hidden_dims),
            "wm_rot6d_predictor_hidden_dims": self._copy_optional_list(rot6d_predictor_hidden_dims),
            "wm_binary_predictor_hidden_dims": self._copy_optional_list(binary_predictor_hidden_dims),
            "wm_global_pool_hidden_dims": self._copy_optional_list(global_pool_hidden_dims),
            "wm_global_scalar_predictor_hidden_dims": self._copy_optional_list(global_scalar_predictor_hidden_dims),
            "wm_global_rot6d_predictor_hidden_dims": self._copy_optional_list(global_rot6d_predictor_hidden_dims),
            "scalar_loss_fn": serialize_value(self.scalar_loss_fn),
            "next_obs_pred_config": {
                "local_scalar_target_indices": self._copy_optional_list(self.local_scalar_target_indices),
                "local_angle_target_indices": self._copy_optional_list(self.local_angle_target_sin_indices),
                "local_rot6d_target_indices": self._copy_optional_list(self.local_rot6d_target_indices),
                "local_binary_target_indices": self._copy_optional_list(self.local_binary_target_indices),
                "global_scalar_target_indices": self._copy_optional_list(self.global_scalar_target_indices),
                "global_rot6d_target_indices": self._copy_optional_list(self.global_rot6d_target_indices),
                "scalar_loss_weight": self.scalar_loss_weight,
                "angle_loss_weight": self.angle_loss_weight,
                "rot6d_loss_weight": self.rot6d_loss_weight,
                "binary_loss_weight": self.binary_loss_weight,
                "global_scalar_loss_weight": self.global_scalar_loss_weight,
                "global_rot6d_loss_weight": self.global_rot6d_loss_weight,
                "binary_target_ema_decay": self.binary_target_ema_decay,
                "binary_target_ema_eps": self.binary_target_ema_eps,
                "predict_delta": serialize_value(self.predict_delta_mode),
            },
        }


    def compute_scalar_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
            base_local_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_scalar_target_indices is None:
            return None

        pred_scalars = self.local_scalars_predictor(latent_preds)
        target_scalars = next_local_obs[..., self.local_scalar_target_indices]
        if self.predict_delta:
            if base_local_obs is None:
                raise ValueError("local_obs is required when predict_delta is True")
            target_scalars = target_scalars - base_local_obs[..., self.local_scalar_target_indices]
        scalar_losses = self.scalar_loss_fn(pred_scalars, target_scalars)
        loss_per_item = self._reduce_scalar_feature_loss(
            scalar_losses,
            target_shape=target_scalars.shape,
        )
        return self._masked_scalar_loss_mean(loss_per_item, valid_mask)

    def compute_angle_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
            base_local_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_angle_target_sin_indices is None:
            return None

        pred_angles = self.local_angles_predictor(latent_preds)
        if self.predict_delta:
            if base_local_obs is None:
                raise ValueError("local_obs is required when predict_delta is True")
            target_angles_sin = next_local_obs[..., self.local_angle_target_sin_indices]
            target_angles_cos = next_local_obs[..., self.local_angle_target_cos_indices]
            base_angles_sin = base_local_obs[..., self.local_angle_target_sin_indices]
            base_angles_cos = base_local_obs[..., self.local_angle_target_cos_indices]
            delta_angles_sin = target_angles_sin * base_angles_cos - target_angles_cos * base_angles_sin
            delta_angles_cos = target_angles_cos * base_angles_cos + target_angles_sin * base_angles_sin

            pred_angles = pred_angles.reshape(*pred_angles.shape[:-1], -1)
            pred_sin = torch.sin(pred_angles)
            pred_cos = torch.cos(pred_angles)
            cosine_sim = pred_sin * delta_angles_sin + pred_cos * delta_angles_cos
        else:
            target_angles_sin = next_local_obs[..., self.local_angle_target_sin_indices]
            target_angles_cos = next_local_obs[..., self.local_angle_target_cos_indices]
            target_angles = torch.stack((target_angles_sin, target_angles_cos), dim=-1)

            pred_pairs = pred_angles.reshape(*pred_angles.shape[:-1], -1, 2)
            target_pairs = target_angles.reshape(*target_angles.shape[:-2], -1, 2)
            cosine_sim = F.cosine_similarity(pred_pairs, target_pairs, dim=-1, eps=1e-8)

        loss_per_angle = 1.0 - cosine_sim
        loss_per_item = loss_per_angle.mean(dim=-1)
        return masked_mean(loss_per_item, valid_mask)

    def compute_rot6d_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
            base_local_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_rot6d_target_indices is None:
            return None

        return self._compute_rot6d_prediction_loss(
            predictor=self.local_rot6ds_predictor,
            latent_preds=latent_preds,
            target_obs=next_local_obs,
            target_start_indices=self.local_rot6d_target_indices,
            valid_mask=valid_mask,
            base_obs=base_local_obs,
        )

    def _compute_rot6d_prediction_loss(
            self,
            predictor: nn.Module,
            latent_preds: torch.Tensor,
            target_obs: torch.Tensor,
            target_start_indices: list[int],
            valid_mask: torch.Tensor | None,
            base_obs: torch.Tensor | None,
    ) -> torch.Tensor:
        target_indices = self._expand_rot6d_indices(target_start_indices)
        if self.predict_delta:
            if base_obs is None:
                raise ValueError("base_obs is required when predict_delta is True")
            pred_rotvecs = predictor(latent_preds)
            pred_rotvecs = pred_rotvecs.reshape(*pred_rotvecs.shape[:-1], -1, 3)

            target_rot6ds = target_obs[..., target_indices]
            base_rot6ds = base_obs[..., target_indices]
            target_rot6ds = target_rot6ds.reshape(*target_rot6ds.shape[:-1], -1, 6)
            base_rot6ds = base_rot6ds.reshape(*base_rot6ds.shape[:-1], -1, 6)

            target_mats = self._rot6d_to_matrix(target_rot6ds)
            base_mats = self._rot6d_to_matrix(base_rot6ds)
            target_mats = target_mats @ base_mats.transpose(-1, -2)

            pred_mats = self._rotvec_to_matrix(pred_rotvecs)
        else:
            pred_rot6ds = predictor(latent_preds)
            target_rot6ds = target_obs[..., target_indices]

            pred_rot6ds = pred_rot6ds.reshape(*pred_rot6ds.shape[:-1], -1, 6)
            target_rot6ds = target_rot6ds.reshape(*target_rot6ds.shape[:-1], -1, 6)

            pred_mats = self._rot6d_to_matrix(pred_rot6ds)
            target_mats = self._rot6d_to_matrix(target_rot6ds)

        trace_sim = (pred_mats * target_mats).sum(dim=(-1, -2))
        loss_per_rot = 1.0 - (trace_sim / 3.0)
        loss_per_item = loss_per_rot.mean(dim=-1)
        return masked_mean(loss_per_item, valid_mask)


    def compute_binary_loss(
            self,
            latent_preds: torch.Tensor,
            next_local_obs: torch.Tensor,
            valid_mask: torch.Tensor | None,
            base_local_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.local_binary_target_indices is None:
            return None

        pred_binaries = self.local_binaries_predictor(latent_preds)
        target_binaries = next_local_obs[..., self.local_binary_target_indices]
        ema = self._update_binary_target_ema(target_binaries, valid_mask)
        weights = self._binary_balance_weights(target_binaries, ema)
        binary_losses = F.binary_cross_entropy_with_logits(
            pred_binaries,
            target_binaries,
            reduction="none",
            weight=weights,
        )
        loss_per_item = self._reduce_feature_loss(binary_losses)
        return masked_mean(loss_per_item, valid_mask)

    def compute_global_scalar_loss(
            self,
            global_latent_preds: torch.Tensor | None,
            next_global_obs: torch.Tensor | None,
            valid_mask: torch.Tensor | None,
            base_global_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.global_scalar_target_indices is None:
            return None
        if global_latent_preds is None or next_global_obs is None:
            raise ValueError("global_latent_preds and next_global_obs are required for global scalar targets")

        pred_scalars = self.global_scalars_predictor(global_latent_preds)
        target_scalars = next_global_obs[..., self.global_scalar_target_indices]
        if self.predict_delta:
            if base_global_obs is None:
                raise ValueError("global_obs is required when predict_delta is True for global scalar targets")
            target_scalars = target_scalars - base_global_obs[..., self.global_scalar_target_indices]
        scalar_losses = self.scalar_loss_fn(pred_scalars, target_scalars)
        loss_per_item = self._reduce_scalar_feature_loss(
            scalar_losses,
            target_shape=target_scalars.shape,
        )
        return self._masked_scalar_loss_mean(loss_per_item, valid_mask)

    @staticmethod
    def _masked_scalar_loss_mean(
            loss_per_item: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if loss_per_item.ndim == 0 and valid_mask is not None:
            raise ValueError(
                "A fully reduced scalar_loss_fn cannot be combined with a validity mask; "
                "return an unreduced or per-item loss instead."
            )
        return masked_mean(loss_per_item, valid_mask)

    def compute_global_rot6d_loss(
            self,
            global_latent_preds: torch.Tensor | None,
            next_global_obs: torch.Tensor | None,
            valid_mask: torch.Tensor | None,
            base_global_obs: torch.Tensor | None,
    ) -> Optional[torch.Tensor]:
        if self.global_rot6d_target_indices is None:
            return None
        if global_latent_preds is None or next_global_obs is None:
            raise ValueError("global_latent_preds and next_global_obs are required for global rot6d targets")

        return self._compute_rot6d_prediction_loss(
            predictor=self.global_rot6ds_predictor,
            latent_preds=global_latent_preds,
            target_obs=next_global_obs,
            target_start_indices=self.global_rot6d_target_indices,
            valid_mask=valid_mask,
            base_obs=base_global_obs,
        )

    @property
    def has_local_next_obs_pred_targets(self) -> bool:
        return (
            self.local_scalar_target_indices is not None
            or self.local_angle_target_sin_indices is not None
            or self.local_rot6d_target_indices is not None
            or self.local_binary_target_indices is not None
        )

    @property
    def has_global_next_obs_pred_targets(self) -> bool:
        return self.global_scalar_target_indices is not None or self.global_rot6d_target_indices is not None

    def compute_next_obs_pred_loss(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            local_obs: torch.Tensor | None = None,
            next_global_obs: torch.Tensor | None = None,
            global_obs: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            loss_agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        (
            total_loss,
            scalar_loss,
            angle_loss,
            rot6d_loss,
            binary_loss,
            global_scalar_loss,
            global_rot6d_loss,
        ) = self._compute_next_obs_pred_loss_fn(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            next_global_obs=next_global_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            loss_agent_mask=loss_agent_mask,
            time_mask=time_mask,
        )
        metrics: dict[str, Any] = {}
        if scalar_loss is not None:
            scalar_loss_scaled = scalar_loss * self.scalar_loss_weight
            metrics["scalar_loss"] = scalar_loss.item()
            metrics["scalar_loss_scaled"] = scalar_loss_scaled.item()
        if angle_loss is not None:
            angle_loss_scaled = angle_loss * self.angle_loss_weight
            metrics["angle_loss"] = angle_loss.item()
            metrics["angle_loss_scaled"] = angle_loss_scaled.item()
        if rot6d_loss is not None:
            rot6d_loss_scaled = rot6d_loss * self.rot6d_loss_weight
            metrics["rot6d_loss"] = rot6d_loss.item()
            metrics["rot6d_loss_scaled"] = rot6d_loss_scaled.item()
        if binary_loss is not None:
            binary_loss_scaled = binary_loss * self.binary_loss_weight
            metrics["binary_loss"] = binary_loss.item()
            metrics["binary_loss_scaled"] = binary_loss_scaled.item()
        if global_scalar_loss is not None:
            global_scalar_loss_scaled = global_scalar_loss * self.global_scalar_loss_weight
            metrics["global_scalar_loss"] = global_scalar_loss.item()
            metrics["global_scalar_loss_scaled"] = global_scalar_loss_scaled.item()
        if global_rot6d_loss is not None:
            global_rot6d_loss_scaled = global_rot6d_loss * self.global_rot6d_loss_weight
            metrics["global_rot6d_loss"] = global_rot6d_loss.item()
            metrics["global_rot6d_loss_scaled"] = global_rot6d_loss_scaled.item()
        return total_loss, metrics

    def _compute_next_obs_pred_loss_impl(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            local_obs: torch.Tensor | None = None,
            next_global_obs: torch.Tensor | None = None,
            global_obs: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            loss_agent_mask: torch.Tensor | None = None,
            time_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        (
            local_latents,
            next_local_obs,
            actions,
            local_obs,
            next_global_obs,
            global_obs,
            agent_mask,
            loss_agent_mask,
            time_mask,
        ) = self._flatten_recurrent_next_obs_pred_inputs(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            next_global_obs=next_global_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            loss_agent_mask=loss_agent_mask,
            time_mask=time_mask,
        )

        local_latents = self.pre_transition_transform(local_latents)

        latent_preds, base_shape = self._predict_latents_and_base_shape(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            agent_mask=agent_mask,
            time_mask=time_mask,
        )
        latent_preds = self.pre_predictors_transform(latent_preds)

        base_local_obs = None
        if self.has_local_next_obs_pred_targets and self.predict_delta:
            if local_obs is None:
                raise ValueError("local_obs is required when predict_delta is True")
            base_local_obs = self._build_base_local_obs(local_obs, next_local_obs)

        base_global_obs = None
        if self.has_global_next_obs_pred_targets and self.predict_delta:
            if global_obs is None or next_global_obs is None:
                raise ValueError("global_obs and next_global_obs are required for global delta targets")
            base_global_obs = self._build_base_global_obs(global_obs, next_global_obs)

        effective_loss_agent_mask = restrict_loss_agent_mask(
            base_shape=base_shape,
            loss_agent_mask=loss_agent_mask,
            agent_mask=agent_mask,
        )
        valid_mask = build_valid_mask(
            base_shape=base_shape,
            device=latent_preds.device,
            agent_mask=effective_loss_agent_mask,
            time_mask=time_mask,
        )

        scalar_loss = self.compute_scalar_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
            base_local_obs=base_local_obs,
        )
        angle_loss = self.compute_angle_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
            base_local_obs=base_local_obs,
        )
        rot6d_loss = self.compute_rot6d_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
            base_local_obs=base_local_obs,
        )
        binary_loss = self.compute_binary_loss(
            latent_preds=latent_preds,
            next_local_obs=next_local_obs,
            valid_mask=valid_mask,
            base_local_obs=base_local_obs,
        )
        global_latent_preds = None
        global_valid_mask = None
        if self.has_global_next_obs_pred_targets:
            if next_global_obs is None:
                raise ValueError("next_global_obs is required for global next-observation prediction targets")
            global_latent_preds = self._pool_global_latent_preds(
                latent_preds,
                agent_mask=effective_loss_agent_mask,
            )
            global_valid_mask = self._build_global_valid_mask(
                base_shape=global_latent_preds.shape[:-1],
                time_mask=time_mask,
                agent_mask=effective_loss_agent_mask,
            )
        global_scalar_loss = self.compute_global_scalar_loss(
            global_latent_preds=global_latent_preds,
            next_global_obs=next_global_obs,
            valid_mask=global_valid_mask,
            base_global_obs=base_global_obs,
        )
        global_rot6d_loss = self.compute_global_rot6d_loss(
            global_latent_preds=global_latent_preds,
            next_global_obs=next_global_obs,
            valid_mask=global_valid_mask,
            base_global_obs=base_global_obs,
        )
        if (
            scalar_loss is None and angle_loss is None and rot6d_loss is None and binary_loss is None
            and global_scalar_loss is None and global_rot6d_loss is None
        ):
            raise ValueError("No next-observation prediction targets configured")

        total_loss = latent_preds.new_zeros(())
        if scalar_loss is not None:
            total_loss = total_loss + scalar_loss * self.scalar_loss_weight
        if angle_loss is not None:
            total_loss = total_loss + angle_loss * self.angle_loss_weight
        if rot6d_loss is not None:
            total_loss = total_loss + rot6d_loss * self.rot6d_loss_weight
        if binary_loss is not None:
            total_loss = total_loss + binary_loss * self.binary_loss_weight
        if global_scalar_loss is not None:
            total_loss = total_loss + global_scalar_loss * self.global_scalar_loss_weight
        if global_rot6d_loss is not None:
            total_loss = total_loss + global_rot6d_loss * self.global_rot6d_loss_weight
        return total_loss, scalar_loss, angle_loss, rot6d_loss, binary_loss, global_scalar_loss, global_rot6d_loss

    @staticmethod
    def _flatten_recurrent_next_obs_pred_inputs(
            *,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            local_obs: torch.Tensor | None,
            next_global_obs: torch.Tensor | None,
            global_obs: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
            loss_agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None, torch.Tensor | None, torch.Tensor | None,
    ]:
        flattened = flatten_recurrent_wm_batch(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            next_global_obs=next_global_obs,
            agent_mask=agent_mask,
            loss_agent_mask=loss_agent_mask,
            time_mask=time_mask,
        )
        if flattened is None:
            return local_latents, next_local_obs, actions, local_obs, next_global_obs, global_obs, agent_mask, loss_agent_mask, time_mask
        return (
            flattened.local_latents,
            flattened.next_local_obs,
            flattened.actions,
            flattened.local_obs,
            flattened.next_global_obs,
            None if global_obs is None else global_obs.reshape(flattened.local_latents.shape[0], -1),
            flattened.agent_mask,
            flattened.loss_agent_mask,
            flattened.time_mask,
        )

    @staticmethod
    def _normalize_indices(indices: Optional[list[int]]) -> Optional[list[int]]:
        if indices is None:
            return None
        if len(indices) == 0:
            return None
        return list(indices)

    @staticmethod
    def _copy_optional_list(values: Optional[list[int]]) -> Optional[list[int]]:
        if values is None:
            return None
        return list(values)

    @staticmethod
    def _normalize_predict_delta_mode(
            predict_delta: PredictDeltaMode | bool | None,
    ) -> PredictDeltaMode | None:
        if predict_delta is None or predict_delta is False:
            return None
        if isinstance(predict_delta, PredictDeltaMode):
            return predict_delta
        if predict_delta is True:
            return PredictDeltaMode.PER_STEP_DELTA
        raise ValueError(f"Unsupported predict_delta value: {predict_delta!r}")

    @staticmethod
    def _reduce_feature_loss(losses: torch.Tensor) -> torch.Tensor:
        if losses.ndim < 3:
            return losses
        return losses.mean(dim=-1)

    @staticmethod
    def _reduce_scalar_feature_loss(
            losses: torch.Tensor,
            *,
            target_shape: torch.Size,
    ) -> torch.Tensor:
        if losses.ndim == 0 or losses.shape == target_shape[:-1]:
            return losses
        if losses.shape == target_shape:
            return losses.mean(dim=-1)
        raise ValueError(
            "Expected scalar loss shape to be scalar, per-item "
            f"{tuple(target_shape[:-1])}, or unreduced {tuple(target_shape)}, got {tuple(losses.shape)}"
        )

    @staticmethod
    def _expand_rot6d_indices(indices: list[int]) -> list[int]:
        return [start + offset for start in indices for offset in range(6)]

    @staticmethod
    def _align_base_local_obs(local_obs: torch.Tensor, next_local_obs: torch.Tensor) -> torch.Tensor:
        if local_obs.ndim == next_local_obs.ndim:
            if local_obs.shape != next_local_obs.shape:
                raise ValueError(
                    f"Expected local_obs shape {tuple(next_local_obs.shape)}, got {tuple(local_obs.shape)}"
                )
            return local_obs
        if local_obs.ndim + 1 == next_local_obs.ndim:
            if local_obs.shape[0] != next_local_obs.shape[0]:
                raise ValueError(
                    f"Expected local_obs batch size {next_local_obs.shape[0]}, got {local_obs.shape[0]}"
                )
            if local_obs.shape[1] != next_local_obs.shape[2] or local_obs.shape[2] != next_local_obs.shape[3]:
                raise ValueError(
                    f"Expected local_obs shape (B, N, F)=({next_local_obs.shape[0]}, {next_local_obs.shape[2]}, "
                    f"{next_local_obs.shape[3]}), got {tuple(local_obs.shape)}"
                )
            return local_obs.unsqueeze(1)
        raise ValueError(
            f"Expected local_obs shape (B, N, F) or {tuple(next_local_obs.shape)}, got {tuple(local_obs.shape)}"
        )

    def _build_base_local_obs(self, local_obs: torch.Tensor, next_local_obs: torch.Tensor) -> torch.Tensor:
        base_local_obs = self._align_base_local_obs(local_obs, next_local_obs)
        if self.predict_delta_mode != PredictDeltaMode.PER_STEP_DELTA:
            return base_local_obs
        if base_local_obs.shape == next_local_obs.shape:
            return base_local_obs
        return torch.cat([base_local_obs, next_local_obs[:, :-1]], dim=1)

    @staticmethod
    def _align_base_global_obs(global_obs: torch.Tensor, next_global_obs: torch.Tensor) -> torch.Tensor:
        if global_obs.ndim == next_global_obs.ndim:
            if global_obs.shape != next_global_obs.shape:
                raise ValueError(
                    f"Expected global_obs shape {tuple(next_global_obs.shape)}, got {tuple(global_obs.shape)}"
                )
            return global_obs
        if global_obs.ndim + 1 == next_global_obs.ndim:
            if global_obs.shape[0] != next_global_obs.shape[0] or global_obs.shape[-1] != next_global_obs.shape[-1]:
                raise ValueError(
                    f"Expected global_obs shape (B, G)=({next_global_obs.shape[0]}, {next_global_obs.shape[-1]}), "
                    f"got {tuple(global_obs.shape)}"
                )
            return global_obs.unsqueeze(1)
        raise ValueError(
            f"Expected global_obs shape (B, G) or {tuple(next_global_obs.shape)}, got {tuple(global_obs.shape)}"
        )

    def _build_base_global_obs(self, global_obs: torch.Tensor, next_global_obs: torch.Tensor) -> torch.Tensor:
        base_global_obs = self._align_base_global_obs(global_obs, next_global_obs)
        if self.predict_delta_mode != PredictDeltaMode.PER_STEP_DELTA:
            return base_global_obs
        if base_global_obs.shape == next_global_obs.shape:
            return base_global_obs
        return torch.cat([base_global_obs, next_global_obs[:, :-1]], dim=1)

    def _pool_global_latent_preds(
            self,
            latent_preds: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        encoded = self.global_pool_encoder(latent_preds) if self.global_pool_encoder is not None else latent_preds
        if agent_mask is None:
            return encoded.mean(dim=-2)
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if encoded.ndim == 3:
            if agent_mask.shape != encoded.shape[:2]:
                raise ValueError(f"Expected agent_mask shape {tuple(encoded.shape[:2])}, got {tuple(agent_mask.shape)}")
            mask = agent_mask
        elif encoded.ndim == 4:
            b, t, n = encoded.shape[:3]
            if agent_mask.ndim == 2:
                if agent_mask.shape != (b, n):
                    raise ValueError(f"Expected agent_mask shape (B, N)=({b}, {n}), got {tuple(agent_mask.shape)}")
                mask = agent_mask[:, None, :]
            elif agent_mask.ndim == 3:
                if agent_mask.shape != (b, t, n):
                    raise ValueError(f"Expected agent_mask shape (B, T, N)=({b}, {t}, {n}), got {tuple(agent_mask.shape)}")
                mask = agent_mask
            else:
                raise ValueError(f"Expected agent_mask ndim 2 or 3, got {agent_mask.ndim}")
        else:
            raise ValueError(f"Expected latent_preds ndim 3 or 4, got {encoded.ndim}")

        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        denom = weights.sum(dim=-2).clamp_min(1.0)
        return (encoded * weights).sum(dim=-2) / denom

    @staticmethod
    def _build_global_valid_mask(
            *,
            base_shape: tuple[int, ...],
            time_mask: torch.Tensor | None,
            agent_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        valid_mask = None
        if agent_mask is not None:
            if len(base_shape) == 1:
                if agent_mask.ndim != 2 or agent_mask.shape[0] != base_shape[0]:
                    raise ValueError(
                        f"Expected single-step agent_mask shape (B, N) with B={base_shape[0]}, "
                        f"got {tuple(agent_mask.shape)}"
                    )
                valid_mask = agent_mask.any(dim=-1)
            elif len(base_shape) == 2:
                if agent_mask.ndim == 2:
                    if agent_mask.shape[0] != base_shape[0]:
                        raise ValueError(
                            f"Expected multi-step agent_mask leading dimension {base_shape[0]}, "
                            f"got {tuple(agent_mask.shape)}"
                        )
                    valid_mask = agent_mask.any(dim=-1, keepdim=True).expand(base_shape)
                elif agent_mask.ndim == 3 and agent_mask.shape[:2] == base_shape:
                    valid_mask = agent_mask.any(dim=-1)
                else:
                    raise ValueError(
                        f"Expected multi-step agent_mask shape (B, N) or (B, T, N) with "
                        f"(B, T)={base_shape}, got {tuple(agent_mask.shape)}"
                    )
            else:
                raise ValueError(f"Expected global prediction base shape (B,) or (B, T), got {base_shape}")

        if time_mask is not None:
            if len(base_shape) != 2:
                raise ValueError("time_mask is only supported for multi-step global loss")
            if time_mask.shape != base_shape or time_mask.dtype != torch.bool:
                raise ValueError(
                    f"Expected time_mask shape {base_shape} and dtype bool, "
                    f"got {tuple(time_mask.shape)} / {time_mask.dtype}"
                )
            valid_mask = time_mask if valid_mask is None else valid_mask & time_mask
        return valid_mask

    def _update_binary_target_ema(
            self,
            target_binaries: torch.Tensor,
            valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.binary_target_ema is None:
            raise ValueError("binary_target_ema is not initialized for binary targets")

        target_binaries = target_binaries.float()
        if valid_mask is None:
            feature_mean = target_binaries.mean(dim=tuple(range(target_binaries.ndim - 1)))
        else:
            weights = valid_mask.to(dtype=target_binaries.dtype).unsqueeze(-1)
            valid_count = weights.sum(dim=tuple(range(weights.ndim - 1)))
            feature_mean = (
                (target_binaries * weights).sum(dim=tuple(range(target_binaries.ndim - 1)))
                / valid_count.clamp_min(1.0)
            )
            feature_mean = torch.where(valid_count > 0, feature_mean, self.binary_target_ema)

        if self.training:
            with torch.no_grad():
                self.binary_target_ema.mul_(self.binary_target_ema_decay).add_(
                    feature_mean * (1.0 - self.binary_target_ema_decay)
                )
        return self.binary_target_ema

    def _binary_balance_weights(self, target_binaries: torch.Tensor, ema: torch.Tensor) -> torch.Tensor:
        eps = self.binary_target_ema_eps
        pos_weights = 0.5 / ema.clamp(min=eps)
        neg_weights = 0.5 / (1.0 - ema).clamp(min=eps)
        return target_binaries * pos_weights + (1.0 - target_binaries) * neg_weights

    def _predict_latents_and_base_shape(
            self,
            local_latents: torch.Tensor,
            next_local_obs: torch.Tensor,
            actions: torch.Tensor,
            agent_mask: torch.Tensor | None,
            time_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if actions.ndim == 3:
            if time_mask is not None:
                raise ValueError("time_mask is only supported for multi-step loss (actions shape (B, T, N, A))")
            if agent_mask is not None and agent_mask.ndim != 2:
                raise ValueError(f"Expected agent_mask shape (B, N), got {tuple(agent_mask.shape)}")
            if next_local_obs.ndim != 3:
                raise ValueError(f"Expected next_local_obs shape (B, N, F), got {tuple(next_local_obs.shape)}")
            if next_local_obs.shape[:2] != local_latents.shape[:2]:
                raise ValueError(
                    f"Expected next_local_obs shape (B, N, F)=({local_latents.shape[0]}, {local_latents.shape[1]}, F), "
                    f"got {tuple(next_local_obs.shape)}"
                )
            latent_preds = (
                local_latents
                if self.skip_first_transition
                else self.transition_model(local_latents, actions, agent_mask=agent_mask)
            )
            return latent_preds, latent_preds.shape[:2]

        if actions.ndim == 4:
            b, t, n, _a = actions.shape
            if local_latents.shape[0] != b or local_latents.shape[1] != n:
                raise ValueError(
                    f"Expected local_latents shape (B, N, D)=({b}, {n}, D), got {tuple(local_latents.shape)}"
                )
            if next_local_obs.ndim != 4 or next_local_obs.shape[:3] != (b, t, n):
                raise ValueError(
                    f"Expected next_local_obs shape (B, T, N, F)=({b}, {t}, {n}, F), got {tuple(next_local_obs.shape)}"
                )
            if self.skip_first_transition:
                first_latent_pred = local_latents.unsqueeze(1)
                if t == 1:
                    latent_preds = first_latent_pred
                else:
                    transition_agent_mask = (
                        agent_mask[:, 1:]
                        if agent_mask is not None and agent_mask.ndim == 3
                        else agent_mask
                    )
                    remaining_latent_preds = self.transition_model.predict_n_steps(
                        local_latents,
                        actions[:, 1:],
                        agent_mask=transition_agent_mask,
                    )
                    latent_preds = torch.cat((first_latent_pred, remaining_latent_preds), dim=1)
            else:
                latent_preds = self.transition_model.predict_n_steps(local_latents, actions, agent_mask=agent_mask)
            return latent_preds, latent_preds.shape[:3]

        raise ValueError(
            f"Expected actions shape (B, N, A) or (B, T, N, A), got {tuple(actions.shape)}"
        )

    @staticmethod
    def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]
        b1 = F.normalize(a1, dim=-1)
        a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(a2_proj, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    @staticmethod
    def _rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.norm(rotvec, dim=-1)
        axis = rotvec / angle.clamp_min(1e-8).unsqueeze(-1)
        x, y, z = axis.unbind(-1)
        zeros = torch.zeros_like(x)
        k_mat = torch.stack(
            (
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ),
            dim=-1,
        ).reshape(*rotvec.shape[:-1], 3, 3)
        sin = torch.sin(angle)[..., None, None]
        cos = torch.cos(angle)[..., None, None]
        eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(*rotvec.shape[:-1], 3, 3)
        return eye + sin * k_mat + (1.0 - cos) * (k_mat @ k_mat)
