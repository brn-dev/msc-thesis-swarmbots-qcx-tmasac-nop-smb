from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils import dlpack

from swarmbots.learn.torch_device import as_device

def to_torch_tensor(
    value: Any,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    target_device = as_device(device)
    if isinstance(value, torch.Tensor):
        return value.to(device=target_device, dtype=dtype if dtype is not None else value.dtype)

    if hasattr(value, "__dlpack__"):
        try:
            tensor = dlpack.from_dlpack(value)
            if dtype is not None or tensor.device != target_device:
                tensor = tensor.to(device=target_device, dtype=dtype if dtype is not None else tensor.dtype)
            return tensor
        except Exception:
            if hasattr(value, "__array__"):
                return torch.as_tensor(np.asarray(value), device=target_device, dtype=dtype)

    return torch.as_tensor(value, device=target_device, dtype=dtype)


def to_numpy_array(value: Any, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(dtype, copy=False) if dtype is not None else value
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    if hasattr(value, "__dlpack__"):
        try:
            tensor = dlpack.from_dlpack(value)
            array = tensor.detach().cpu().numpy()
            return array.astype(dtype, copy=False) if dtype is not None else array
        except Exception:
            pass
    return np.asarray(value, dtype=dtype)


def to_backend_array(
    value: Any,
    *,
    backend: str,
    dtype: Any | None = None,
) -> Any:
    normalized_backend = backend.lower()
    if normalized_backend == "numpy":
        return to_numpy_array(value, dtype=dtype)
    if normalized_backend == "torch":
        return to_torch_tensor(value, device=value.device if isinstance(value, torch.Tensor) else "cpu", dtype=dtype)

    raise ValueError(f"Unsupported backend: {backend}")


def normalize_reset_mask_options(
    options: dict[str, Any] | None,
    *,
    num_envs: int,
) -> dict[str, Any] | None:
    if options is None or "reset_mask" not in options:
        return options

    reset_mask = to_numpy_array(options["reset_mask"], dtype=np.bool_).reshape(-1)
    if reset_mask.shape != (num_envs,):
        raise ValueError(f"Expected reset_mask shape ({num_envs},), got {reset_mask.shape}")

    normalized_options = dict(options)
    normalized_options["reset_mask"] = reset_mask
    return normalized_options
