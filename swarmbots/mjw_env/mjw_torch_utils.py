from __future__ import annotations

import torch

from swarmbots.mj_env.float_or_dist_params import (
    FloatOrBoundedDistParams,
    FloatOrDistParams,
    NormalDistParams,
    SplitUniformDistParams,
    TruncatedNormalDistParams,
    UniformDistParams,
)


def to_device_bool_tensor(
    value: object,
    *,
    device: torch.device,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=torch.bool)
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def sample_float_or_dist(
    value: FloatOrDistParams,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    if isinstance(value, (float, int)):
        return torch.full(shape, float(value), device=device, dtype=torch.float32)
    if isinstance(value, UniformDistParams):
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(
            float(value.low),
            float(value.high),
            generator=generator,
        )
    if isinstance(value, SplitUniformDistParams):
        unit_samples = torch.rand(
            shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        sample_positive_side = torch.randint(
            0,
            2,
            shape,
            device=device,
            dtype=torch.int8,
            generator=generator,
        ).bool()
        negative_samples = float(value.low) + unit_samples * (
            -float(value.margin) - float(value.low)
        )
        positive_samples = float(value.margin) + unit_samples * (
            float(value.high) - float(value.margin)
        )
        return torch.where(sample_positive_side, positive_samples, negative_samples)
    if isinstance(value, NormalDistParams):
        return torch.empty(shape, device=device, dtype=torch.float32).normal_(
            float(value.mean),
            float(value.std),
            generator=generator,
        )
    if isinstance(value, TruncatedNormalDistParams):
        samples = torch.empty(shape, device=device, dtype=torch.float32).normal_(
            float(value.mean),
            float(value.std),
            generator=generator,
        )
        if value.sampling_mode == "clamp":
            return samples.clamp_(float(value.low), float(value.high))
        if value.sampling_mode == "rejection":
            low = float(value.low)
            high = float(value.high)
            invalid_mask = (samples < low) | (samples > high)
            while bool(invalid_mask.any()):
                samples[invalid_mask] = torch.empty(
                    int(invalid_mask.sum().item()),
                    device=device,
                    dtype=torch.float32,
                ).normal_(float(value.mean), float(value.std), generator=generator)
                invalid_mask = (samples < low) | (samples > high)
            return samples
    raise TypeError(f"Unsupported distribution type: {type(value).__name__}")


def sample_float_or_bounded_dist(
    value: FloatOrBoundedDistParams,
    *,
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return sample_float_or_dist(value, shape=shape, device=device, generator=generator)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> torch.Tensor:
    mask_f = mask.to(dtype=values.dtype)
    denom = torch.clamp(mask_f.sum(dim=dim), min=1.0)
    return (values * mask_f).sum(dim=dim) / denom
