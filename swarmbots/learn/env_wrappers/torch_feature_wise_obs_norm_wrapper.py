from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import (
    NormalizeObservations,
    build_obs_norm_tensor_operation,
    should_compile_obs_norm_tensor_operations_by_default,
)
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class TorchFeatureWiseObsNormWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        obs_key: str,
        scalar_feature_indices: list[int] | np.ndarray,
        quaternion_indices: list[int] | np.ndarray,
        eps: float = 1e-6,
        compile_tensor_operations: bool | None = None,
        tensor_operations_compile_mode: str = "reduce-overhead",
    ) -> None:
        super().__init__(env)
        if obs_key not in self.observation_space.spaces:
            raise ValueError(f'Expected "{obs_key}" key in observation space')

        self.obs_key = obs_key
        obs_dim = int(self.observation_space[obs_key].shape[-1])
        self._scalar_indices = self._as_index_array(
            scalar_feature_indices,
            max_index=obs_dim,
            name=f"{obs_key}_scalar_feature_indices",
        ).to(self.device)
        self._quaternion_slices = self._as_quaternion_slices(
            quaternion_indices,
            max_index=obs_dim,
            name=f"{obs_key}_quaternion_indices",
        ).to(self.device)
        self._ensure_no_overlap(self._scalar_indices, self._quaternion_slices, name=obs_key)

        self.obs_rms: TorchRunningMeanStd | None = None
        if self._scalar_indices.numel() > 0:
            self.obs_rms = TorchRunningMeanStd(shape=(self._scalar_indices.numel(),), device=self.device)

        self._eps = float(eps)
        self._update_running_mean = True
        self._compile_tensor_operations = compile_tensor_operations
        self._tensor_operations_compile_mode = tensor_operations_compile_mode
        self._normalize_observations = self._build_normalize_observations()

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool) -> None:
        self._update_running_mean = setting

    def observations(self, observations: TorchObs) -> TorchObs:
        obs = observations[self.obs_key]

        if self.obs_rms is None and self._quaternion_slices.numel() == 0:
            return observations

        statistics_mask = self._statistics_mask(
            obs=obs,
            agent_mask=observations.get("agent_mask", None),
        )
        if self.obs_rms is None:
            running_mean = torch.empty((0,), device=obs.device, dtype=torch.float64)
            running_var = torch.empty((0,), device=obs.device, dtype=torch.float64)
            running_count = torch.zeros((), device=obs.device, dtype=torch.float64)
        else:
            running_mean = self.obs_rms.mean
            running_var = self.obs_rms.var
            running_count = self.obs_rms.count

        normalized_obs, next_mean, next_var, next_count = self._normalize_observations(
            obs,
            statistics_mask,
            self._scalar_indices,
            self._quaternion_slices,
            running_mean,
            running_var,
            running_count,
            self._eps,
            self._update_running_mean and self.obs_rms is not None,
        )
        if self._tensor_operations_are_compiled:
            # reduce-overhead reuses CUDA-graph output buffers, while returned observations
            # remain live until the rollout snapshots them before the next environment step.
            normalized_obs = normalized_obs.clone()
        if self.obs_rms is not None and self._update_running_mean:
            self.obs_rms.mean.copy_(next_mean)
            self.obs_rms.var.copy_(next_var)
            self.obs_rms.count.copy_(next_count)

        normalized_observations = dict(observations)
        normalized_observations[self.obs_key] = normalized_obs
        return normalized_observations

    def _transform_infos(self, infos: dict[str, Any]) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        old_update_running_mean = self._update_running_mean
        self._update_running_mean = False
        try:
            return super()._transform_infos(infos)
        finally:
            self._update_running_mean = old_update_running_mean

    def set_device(self, device: torch.device | str) -> None:
        old_device_type = self.device.type
        if self.obs_rms is not None:
            self.obs_rms.to(device)
        super().set_device(device)
        self._scalar_indices = self._scalar_indices.to(self.device)
        self._quaternion_slices = self._quaternion_slices.to(self.device)
        if self.device.type != old_device_type and self._compile_tensor_operations is None:
            self._normalize_observations = self._build_normalize_observations()

    def _statistics_mask(
        self,
        *,
        obs: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            agent_mask is not None
            and agent_mask.ndim == 2
            and obs.ndim >= 3
            and tuple(agent_mask.shape) == tuple(obs.shape[:2])
        ):
            return agent_mask
        return torch.ones(obs.shape[:-1], device=obs.device, dtype=torch.bool)

    def _build_normalize_observations(self) -> NormalizeObservations:
        compile_operation = (
            should_compile_obs_norm_tensor_operations_by_default(self.device)
            if self._compile_tensor_operations is None
            else self._compile_tensor_operations
        )
        self._tensor_operations_are_compiled = compile_operation
        return build_obs_norm_tensor_operation(
            compile_operation=compile_operation,
            compile_mode=self._tensor_operations_compile_mode,
        )

    @staticmethod
    def _as_index_array(
        indices: list[int] | np.ndarray,
        *,
        max_index: int,
        name: str,
    ) -> torch.Tensor:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return torch.empty((0,), dtype=torch.long)
        if arr.min() < 0 or arr.max() >= max_index:
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {arr}")
        return torch.as_tensor(arr, dtype=torch.long)

    @staticmethod
    def _as_quaternion_slices(
        indices: list[int] | np.ndarray,
        *,
        max_index: int,
        name: str,
    ) -> torch.Tensor:
        starts = np.asarray(indices, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            return torch.empty((0, 4), dtype=torch.long)
        if starts.min() < 0 or np.any(starts + 3 >= max_index):
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {starts}")
        return torch.as_tensor(np.stack([starts + i for i in range(4)], axis=1), dtype=torch.long)

    @staticmethod
    def _ensure_no_overlap(
        scalar_indices: torch.Tensor,
        quaternion_slices: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if scalar_indices.numel() == 0 or quaternion_slices.numel() == 0:
            return
        quat_indices = torch.unique(quaternion_slices.reshape(-1))
        overlap = np.intersect1d(scalar_indices.cpu().numpy(), quat_indices.cpu().numpy())
        if overlap.size > 0:
            raise ValueError(f"{name} scalar indices overlap with quaternion indices: {overlap.tolist()}")
