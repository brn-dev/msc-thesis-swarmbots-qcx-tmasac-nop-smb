

import pathlib
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from swarmbots.learn.env_wrappers.feature_wise_obs_norm_wrapper import FeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.torch_device import as_device


def load_checkpoint(path: str | pathlib.Path, map_location: Any | None = "cpu") -> Any:
    return torch.load(pathlib.Path(path), map_location=map_location, weights_only=False)


def extract_policy_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
        return checkpoint["policy_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def align_torch_compile_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    target_keys: Iterable[str],
) -> dict[str, Any]:
    """Align state-dict keys across compiled and uncompiled module wrappers."""

    def canonical_key(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    target_by_canonical_key: dict[str, str] = {}
    for target_key in target_keys:
        canonical = canonical_key(target_key)
        previous = target_by_canonical_key.setdefault(canonical, target_key)
        if previous != target_key:
            raise ValueError(
                "Target policy has ambiguous torch.compile state-dict keys: "
                f"{previous!r} and {target_key!r}"
            )

    aligned: dict[str, Any] = {}
    source_by_target_key: dict[str, str] = {}
    for source_key, value in state_dict.items():
        target_key = target_by_canonical_key.get(canonical_key(source_key), source_key)
        previous_source = source_by_target_key.setdefault(target_key, source_key)
        if previous_source != source_key:
            raise ValueError(
                "Checkpoint has ambiguous torch.compile state-dict keys: "
                f"{previous_source!r} and {source_key!r} both map to {target_key!r}"
            )
        aligned[target_key] = value
    return aligned


def migrate_tmasac_removed_connector_action_dims(
    state_dict: Mapping[str, Any],
    target_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove trailing connector columns while preserving TMASAC action-input weights."""
    migrated = dict(state_dict)
    migrated_keys: list[str] = []
    trailing_input_weight_suffixes = (
        ".local_action_encoder.0.weight",
        ".observation_action_encoder.action_encoder.0.weight",
        ".transition_model.coembed.0.weight",
    )
    trailing_input_norm_suffixes = (
        ".local_action_input_norm.weight",
        ".local_action_input_norm.bias",
    )

    for key, value in state_dict.items():
        canonical_key = ".".join(
            part for part in key.split(".") if part != "_orig_mod"
        )
        target = target_state_dict.get(key)
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            continue
        if value.shape == target.shape:
            continue

        if (
            canonical_key.endswith(trailing_input_weight_suffixes)
            and value.ndim == 2
            and target.ndim == 2
            and value.shape[0] == target.shape[0]
            and value.shape[1] > target.shape[1]
        ):
            migrated[key] = value[:, : target.shape[1]]
            migrated_keys.append(key)
        elif (
            canonical_key.endswith(trailing_input_norm_suffixes)
            and value.ndim == 1
            and target.ndim == 1
            and value.shape[0] > target.shape[0]
        ):
            migrated[key] = value[: target.shape[0]]
            migrated_keys.append(key)

    if migrated_keys:
        logger.warning(
            "Migrated checkpoint tensors after removing trailing action dimensions: "
            f"{migrated_keys}"
        )
    return migrated


def extract_optimizer_state_dict(checkpoint: Any) -> Optional[dict[str, Any]]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("optimizer_state_dict", None)
    return None


def extract_env_state(checkpoint: Any) -> Optional[list[dict[str, Any]]]:
    if isinstance(checkpoint, dict):
        env_state = checkpoint.get("env_state", None)
        if env_state is None:
            return None
        if not isinstance(env_state, list):
            raise TypeError(f"env_state must be a list, got {type(env_state)}")
        return env_state
    return None


def copy_running_mean_std(src: Any, dst: Any) -> None:
    if tuple(dst.mean.shape) != tuple(src.mean.shape):
        raise ValueError(f'{dst.mean.shape = } is not equal {src.mean.shape = }')
    if tuple(dst.var.shape) != tuple(src.var.shape):
        raise ValueError(f'{dst.var.shape = } is not equal {src.var.shape = }')
    if isinstance(dst.mean, torch.Tensor):
        dst.mean = torch.as_tensor(src.mean, device=dst.mean.device, dtype=dst.mean.dtype).clone()
        dst.var = torch.as_tensor(src.var, device=dst.var.device, dtype=dst.var.dtype).clone()
        dst.count = torch.as_tensor(src.count, device=dst.count.device, dtype=dst.count.dtype).clone()
        return

    dst.mean = np.asarray(src.mean).copy()
    dst.var = np.asarray(src.var).copy()
    dst.count = float(torch.as_tensor(src.count, device="cpu", dtype=torch.float64).item())


def clone_running_mean_std(src: Any) -> Any:
    if isinstance(src.mean, torch.Tensor):
        return SimpleNamespace(
            mean=src.mean.detach().clone(),
            var=src.var.detach().clone(),
            count=torch.as_tensor(src.count, device=src.count.device, dtype=src.count.dtype).detach().clone(),
        )

    return SimpleNamespace(
        mean=np.asarray(src.mean).copy(),
        var=np.asarray(src.var).copy(),
        count=float(torch.as_tensor(src.count, device="cpu", dtype=torch.float64).item()),
    )


def iter_running_mean_std(env: Any) -> list[tuple[str, Any]]:
    rms_entries: list[tuple[str, Any]] = []
    for attr_name in dir(env):
        if not attr_name.endswith("_rms"):
            continue

        value = getattr(env, attr_name, None)
        if value is None:
            continue
        if all(hasattr(value, field) for field in ("mean", "var", "count")):
            rms_entries.append((attr_name, value))
    return rms_entries


def _wrapper_class_aliases(env: Any) -> tuple[str, ...]:
    current_class = type(env).__name__
    if isinstance(env, TorchFeatureWiseObsNormWrapper):
        return current_class, "FeatureWiseObsNormWrapper"
    if isinstance(env, TorchNormalizeRewardWrapper):
        return current_class, "NormalizeReward"
    return (current_class,)

def capture_env_state(env: Any) -> list[dict[str, Any]]:
    env_state: list[dict[str, Any]] = []
    current_env = env
    while hasattr(current_env, "env"):
        wrapper_state: dict[str, Any] = {}

        if isinstance(current_env, (FeatureWiseObsNormWrapper, TorchFeatureWiseObsNormWrapper)):
            wrapper_state['obs_key'] = current_env.obs_key

        for attr_name, value in iter_running_mean_std(current_env):
            wrapper_state[attr_name] = clone_running_mean_std(value)

        if wrapper_state:
            wrapper_state["wrapper_class"] = type(current_env).__name__
            env_state.append(wrapper_state)

        current_env = current_env.env
    return env_state


def apply_env_state(env: Any, env_state: Optional[list[dict[str, Any]]]) -> None:
    if not env_state:
        return

    current_env = env
    search_start_idx = 0

    while hasattr(current_env, "env"):
        rms_entries = dict(iter_running_mean_std(current_env))
        has_rms = bool(rms_entries)
        if has_rms:
            current_classes = _wrapper_class_aliases(current_env)
            match_idx = None
            has_rms_key = lambda state: any(key.endswith("_rms") for key in state)
            candidates = [
                i
                for i in range(search_start_idx, len(env_state))
                if env_state[i].get("wrapper_class") in current_classes and has_rms_key(env_state[i])
            ]
            if candidates:
                if isinstance(current_env, (FeatureWiseObsNormWrapper, TorchFeatureWiseObsNormWrapper)):
                    for i in candidates:
                        if env_state[i].get("obs_key") == current_env.obs_key:
                            match_idx = i
                            break
                    if match_idx is None:
                        if any("obs_key" in env_state[i] for i in candidates):
                            raise ValueError(
                                f"Obs keys not equal: {current_env.obs_key} vs "
                                f"{[env_state[i].get('obs_key') for i in candidates]}"
                            )
                        match_idx = candidates[0]
                else:
                    match_idx = candidates[0]

            if match_idx is None:
                # Hard fail so we avoid easy to overlook errors
                raise ValueError(
                    f"No saved env_state entry for wrapper {current_classes[0]}; normalization stats not restored for it."
                )
            else:
                saved_state = env_state[match_idx]
                if saved_state.get("wrapper_class") not in current_classes:
                    logger.warning(
                        f"Wrapper type mismatch during env_state apply: "
                        f"{saved_state.get('wrapper_class')} vs {current_classes[0]}"
                    )

                for attr_name, saved_value in saved_state.items():
                    if attr_name.endswith("_rms") and attr_name in rms_entries:
                        copy_running_mean_std(saved_value, rms_entries[attr_name])

                search_start_idx = match_idx + 1

        current_env = current_env.env


def freeze_env_normalization(env: Any) -> None:
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "update_running_mean"):
            current_env.update_running_mean = False
        current_env = current_env.env


def move_env_to_device(env: Any, device: torch.device | str) -> None:
    target_device = as_device(device)
    current_env = env
    while True:
        if hasattr(current_env, "set_device"):
            current_env.set_device(target_device)
        elif hasattr(current_env, "device"):
            current_env.device = target_device

        if not hasattr(current_env, "env"):
            break
        current_env = current_env.env
