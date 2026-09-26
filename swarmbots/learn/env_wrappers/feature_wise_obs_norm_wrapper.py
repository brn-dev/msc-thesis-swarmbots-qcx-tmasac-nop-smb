from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.logger import warn
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd


class FeatureWiseObsNormWrapper(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    Normalizes selected observation features for a single obs dict entry and
    canonicalizes quaternion signs.
    """

    def __init__(
        self,
        env: VectorEnv,
        obs_key: str,
        scalar_feature_indices: list[int] | np.ndarray,
        quaternion_indices: list[int] | np.ndarray,
        eps: float = 1e-6,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            obs_key=obs_key,
            scalar_feature_indices=scalar_feature_indices,
            quaternion_indices=quaternion_indices,
            eps=eps,
        )
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the "
                f"`SameStep` autoreset api. Read https://farama.org/Vector-Autoreset-Mode "
                f"for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.SAME_STEP}

        if not isinstance(self.env.single_observation_space, gym.spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(self.env.single_observation_space)}")
        if obs_key not in self.env.single_observation_space.spaces:
            raise ValueError(f'Expected "{obs_key}" key in observation space')

        self.obs_key = obs_key
        space: gym.spaces.Box = self.env.single_observation_space[obs_key]  # type: ignore[assignment]
        obs_dim = int(space.shape[-1])

        self._scalar_indices = self._as_index_array(
            scalar_feature_indices, max_index=obs_dim, name=f"{obs_key}_scalar_feature_indices"
        )
        self._quaternion_slices = self._as_quaternion_slices(
            quaternion_indices, max_index=obs_dim, name=f"{obs_key}_quaternion_indices"
        )
        self._ensure_no_overlap(self._scalar_indices, self._quaternion_slices, name=obs_key)

        self.obs_rms: RunningMeanStd | None = None
        if self._scalar_indices.size > 0:
            self.obs_rms = RunningMeanStd(
                shape=(self._scalar_indices.size,),
                dtype=space.dtype,
            )

        self._eps = float(eps)
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        obs = observations[self.obs_key]

        if self._update_running_mean:
            if self.obs_rms is not None:
                scalars = obs[..., self._scalar_indices]
                agent_mask = observations.get("agent_mask", None)
                batch = self._prepare_batch(scalars, agent_mask)
                if batch.size > 0:
                    if batch.shape[-1] != self.obs_rms.mean.shape[-1]:
                        raise ValueError(
                            f'FeatureWiseObsNormWrapper("{self.obs_key}") batch width mismatch: '
                            f"{batch.shape[-1]} != {self.obs_rms.mean.shape[-1]}"
                        )
                    self.obs_rms.update(batch)

        if self.obs_rms is None and self._quaternion_slices.size == 0:
            return observations

        normalized_obs = np.array(obs, copy=True)
        if self.obs_rms is not None:
            scalars = normalized_obs[..., self._scalar_indices]
            normalized_obs[..., self._scalar_indices] = (scalars - self.obs_rms.mean) / np.sqrt(
                self.obs_rms.var + self._eps
            )

        if self._quaternion_slices.size > 0:
            self._normalize_quaternion_signs(normalized_obs, self._quaternion_slices)

        normalized_observations = dict(observations)
        normalized_observations[self.obs_key] = normalized_obs
        return normalized_observations

    def _prepare_batch(
        self,
        samples: np.ndarray,
        agent_mask: np.ndarray | None,
    ) -> np.ndarray:
        if (
            agent_mask is not None
            and agent_mask.ndim == 2
            and samples.ndim >= 3
            and agent_mask.shape == samples.shape[:2]
        ):
            samples = samples[agent_mask.astype(bool)]
            if samples.ndim == 1:
                return samples.reshape(1, -1)
            return samples.reshape(-1, samples.shape[-1])
        if samples.ndim == 1:
            return samples.reshape(1, -1)
        return samples.reshape(-1, samples.shape[-1])

    @staticmethod
    def _as_index_array(
        indices: list[int] | np.ndarray, *, max_index: int, name: str
    ) -> np.ndarray:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return arr
        if arr.min() < 0 or arr.max() >= max_index:
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {arr}")
        return arr

    @staticmethod
    def _as_quaternion_slices(
        indices: list[int] | np.ndarray, *, max_index: int, name: str
    ) -> np.ndarray:
        starts = np.asarray(indices, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            return np.empty((0, 4), dtype=np.int64)
        if starts.min() < 0 or np.any(starts + 3 >= max_index):
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {starts}")
        return np.stack([starts + i for i in range(4)], axis=1)

    @staticmethod
    def _normalize_quaternion_signs(obs: np.ndarray, quat_slices: np.ndarray) -> None:
        quats = obs[..., quat_slices]
        sign = np.where(quats[..., 0] < 0, -1.0, 1.0)
        quats = quats * sign[..., None]
        obs[..., quat_slices] = quats

    @staticmethod
    def _ensure_no_overlap(
        scalar_indices: np.ndarray, quaternion_slices: np.ndarray, *, name: str
    ) -> None:
        if scalar_indices.size == 0 or quaternion_slices.size == 0:
            return
        quat_indices = np.unique(quaternion_slices.reshape(-1))
        overlap = np.intersect1d(scalar_indices, quat_indices)
        if overlap.size > 0:
            raise ValueError(
                f"{name} scalar indices overlap with quaternion indices: {overlap.tolist()}"
            )
