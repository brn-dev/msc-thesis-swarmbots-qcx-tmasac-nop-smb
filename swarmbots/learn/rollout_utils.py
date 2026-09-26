from typing import Any

import numpy as np
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import to_torch_tensor


def append_episode_infos(
        *,
        episode_infos: list[dict[str, Any]],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> None:
    info_source = infos
    if "episode" not in info_source:
        final_info = infos.get("final_info", None)
        if isinstance(final_info, dict) and "episode" in final_info:
            info_source = final_info
        else:
            return

    episode_stats = info_source["episode"]
    if not isinstance(episode_stats, dict):
        raise ValueError(f"Expected infos['episode'] to be a dict, got {type(episode_stats)}")

    episode_mask = to_torch_tensor(
        info_source.get("_episode", dones),
        device=dones.device,
        dtype=torch.bool,
    ).reshape(-1)
    if tuple(episode_mask.shape) != tuple(dones.shape):
        raise ValueError(f"Expected infos['_episode'] shape {tuple(dones.shape)}, got {tuple(episode_mask.shape)}")
    if not torch.equal(episode_mask, dones):
        raise ValueError("Expected infos['_episode'] to match computed dones.")

    for env_idx in torch.nonzero(episode_mask, as_tuple=False).flatten().tolist():
        episode_info = {}
        for key, values in episode_stats.items():
            if key.startswith("_"):
                continue
            value_mask = episode_stats.get(f"_{key}", None)
            if value_mask is not None and not bool(to_python_episode_stat(value_mask[env_idx])):
                continue
            episode_info[key] = to_python_episode_stat(values[env_idx])
        for scenario_key in ("scenario_id", "scenario_name"):
            scenario_values = info_source.get(scenario_key, infos.get(scenario_key, None))
            if scenario_values is not None:
                episode_info[scenario_key] = to_python_episode_stat(scenario_values[env_idx])
        episode_infos.append(episode_info)


def extract_bootstrap_obs(
        *,
        env: BaseLearnEnvWrapper,
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not torch.any(dones):
        return next_obs

    final_obs_mask = _validated_final_obs_mask(infos=infos, dones=dones)
    expected_obs_keys = set(next_obs)
    bootstrap_obs = {key: value.clone() for key, value in next_obs.items()}
    final_obs_value = infos["final_obs"]
    if isinstance(final_obs_value, dict):
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_value)
        _validate_final_obs_keys(final_obs=final_obs, expected_obs_keys=expected_obs_keys)
        for key in expected_obs_keys:
            bootstrap_obs[key][final_obs_mask] = final_obs[key][final_obs_mask]
        return bootstrap_obs

    final_obs_entries = np.asarray(final_obs_value, dtype=object).reshape(-1)

    for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_entries[env_idx])
        _validate_final_obs_keys(final_obs=final_obs, expected_obs_keys=expected_obs_keys)
        for key in expected_obs_keys:
            bootstrap_obs[key][env_idx] = final_obs[key]

    return bootstrap_obs


def extract_terminal_obs(
        *,
        env: BaseLearnEnvWrapper,
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    if not torch.any(dones):
        return None

    final_obs_mask = _validated_final_obs_mask(infos=infos, dones=dones)
    expected_obs_keys = set(next_obs)
    final_obs_value = infos["final_obs"]
    if isinstance(final_obs_value, dict):
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_value)
        _validate_final_obs_keys(final_obs=final_obs, expected_obs_keys=expected_obs_keys)
        return {key: final_obs[key][final_obs_mask] for key in expected_obs_keys}

    final_obs_entries = np.asarray(final_obs_value, dtype=object).reshape(-1)
    terminal_obs: dict[str, list[torch.Tensor]] = {key: [] for key in expected_obs_keys}

    for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_entries[env_idx])
        _validate_final_obs_keys(final_obs=final_obs, expected_obs_keys=expected_obs_keys)
        for key in expected_obs_keys:
            terminal_obs[key].append(final_obs[key])

    return {key: torch.stack(values, dim=0) for key, values in terminal_obs.items()}


def _validated_final_obs_mask(
        *,
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> torch.Tensor:
    if "final_obs" not in infos or "_final_obs" not in infos:
        raise ValueError("SAME_STEP rollouts require infos['final_obs'] and infos['_final_obs'] for done environments.")

    final_obs_mask = to_torch_tensor(infos["_final_obs"], device=dones.device, dtype=torch.bool).reshape(-1)
    if tuple(final_obs_mask.shape) != tuple(dones.shape):
        raise ValueError(f"Expected infos['_final_obs'] shape {tuple(dones.shape)}, got {tuple(final_obs_mask.shape)}")
    if not torch.equal(final_obs_mask, dones):
        raise ValueError("Expected infos['_final_obs'] to match computed dones.")
    return final_obs_mask


def _final_obs_to_torch(
        *,
        env: BaseLearnEnvWrapper,
        final_obs: Any,
) -> dict[str, torch.Tensor]:
    try:
        return env._obs_to_torch(final_obs)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Expected final_obs to contain observation key {missing_key!r}.") from exc


def _validate_final_obs_keys(
        *,
        final_obs: dict[str, torch.Tensor],
        expected_obs_keys: set[str],
) -> None:
    missing_keys = expected_obs_keys.difference(final_obs)
    if missing_keys:
        raise ValueError(f"Expected final_obs to contain observation keys {sorted(missing_keys)}.")


def to_python_episode_stat(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def snapshot_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs.items()}


def initial_previous_actions(
        *,
        obs: dict[str, torch.Tensor],
        n_agent_actions: int,
        policy: BasePolicy | None = None,
) -> torch.Tensor | None:
    if policy is not None and not policy.requires_previous_actions():
        return None

    local_obs = obs["local_obs"]
    n_envs, n_agents = local_obs.shape[:2]
    return torch.zeros(
        (n_envs, n_agents, n_agent_actions),
        dtype=local_obs.dtype,
        device=local_obs.device,
    )


def sample_random_actions(
        action_space: VectorHybridActionSpace,
        *,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    action_dict = action_space.sample()
    actions = action_space.concat_actions(action_dict)
    return to_torch_tensor(actions, device=device, dtype=dtype)
