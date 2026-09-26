from collections.abc import Callable
from functools import lru_cache

import torch


NormalizedObservations = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
NormalizeObservations = Callable[..., NormalizedObservations]


def _normalize_observations(
    obs: torch.Tensor,
    statistics_mask: torch.Tensor,
    scalar_indices: torch.Tensor,
    quaternion_slices: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    eps: float,
    update_statistics: bool,
) -> NormalizedObservations:
    scalar_samples = obs[..., scalar_indices].to(dtype=running_mean.dtype)
    next_mean = running_mean
    next_var = running_var
    next_count = running_count

    if update_statistics:
        sample_mask = statistics_mask.unsqueeze(-1)
        weights = sample_mask.to(dtype=running_mean.dtype)
        masked_samples = torch.where(sample_mask, scalar_samples, 0.0)
        reduction_dims = tuple(range(statistics_mask.ndim))
        batch_count = weights.sum()
        safe_batch_count = batch_count.clamp_min(1.0)
        batch_mean = masked_samples.sum(dim=reduction_dims) / safe_batch_count
        centered_samples = torch.where(sample_mask, scalar_samples - batch_mean, 0.0)
        batch_var = centered_samples.square().sum(dim=reduction_dims) / safe_batch_count

        total_count = running_count + batch_count
        safe_total_count = total_count.clamp_min(1.0)
        delta = batch_mean - running_mean
        merged_mean = running_mean + delta * batch_count / safe_total_count
        merged_m2 = (
            running_var * running_count
            + batch_var * batch_count
            + delta.square() * running_count * batch_count / safe_total_count
        )
        has_samples = batch_count > 0
        next_mean = torch.where(has_samples, merged_mean, running_mean)
        next_var = torch.where(has_samples, merged_m2 / safe_total_count, running_var)
        next_count = torch.where(has_samples, total_count, running_count)

    normalized_obs = obs.clone()
    normalized_obs[..., scalar_indices] = (
        obs[..., scalar_indices] - next_mean.to(dtype=obs.dtype)
    ) / torch.sqrt(next_var.to(dtype=obs.dtype) + eps)

    quaternions = normalized_obs[..., quaternion_slices]
    quaternion_signs = torch.where(quaternions[..., 0] < 0, -1.0, 1.0).to(dtype=obs.dtype)
    normalized_obs[..., quaternion_slices] = quaternions * quaternion_signs.unsqueeze(-1)
    return normalized_obs, next_mean, next_var, next_count


def should_compile_obs_norm_tensor_operations_by_default(device: torch.device) -> bool:
    return device.type == "cuda" and hasattr(torch, "compile") and callable(torch.compile)


@lru_cache(maxsize=None)
def build_obs_norm_tensor_operation(
    *,
    compile_operation: bool,
    compile_mode: str,
) -> NormalizeObservations:
    if not compile_operation:
        return _normalize_observations
    if not hasattr(torch, "compile") or not callable(torch.compile):
        raise RuntimeError("Compiling observation normalization requires torch.compile support.")
    if not compile_mode:
        raise ValueError("tensor_operations_compile_mode must be non-empty when compilation is enabled.")
    return torch.compile(
        _normalize_observations,
        mode=compile_mode,
        fullgraph=True,
        dynamic=False,
    )
