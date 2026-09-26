from dataclasses import dataclass
from functools import lru_cache
import shutil
import sys
from collections.abc import Callable

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment


@dataclass(slots=True)
class WMEpisodeWindows:
    multi_step_actions: torch.Tensor
    next_local_obs: torch.Tensor
    wm_target_time_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: MaybeTensor
    wm_loss_agent_mask: MaybeTensor


_CompiledWMWindowOutputsFn = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        MaybeTensor,
        MaybeTensor,
        int,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        MaybeTensor,
        MaybeTensor,
    ],
]

_compiled_wm_window_outputs_fns: dict[str, _CompiledWMWindowOutputsFn] = {}


def reset_wm_window_helper_compile_cache() -> None:
    _compiled_wm_window_outputs_fns.clear()
    dynamo_module = getattr(torch, "_dynamo", None)
    if dynamo_module is not None and hasattr(dynamo_module, "reset"):
        dynamo_module.reset()


def pad_time_axis(
        tensor: torch.Tensor,
        *,
        padding_len: int,
        pad_value: bool | int | float = 0,
) -> torch.Tensor:
    if padding_len < 0:
        raise ValueError(f"padding_len must be >= 0, got {padding_len}")
    if padding_len == 0:
        return tensor

    padding = torch.full(
        (padding_len, *tensor.shape[1:]),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=0)


def build_wm_episode_windows(
        episode: PPOEpisodeSegment,
        *,
        num_next_steps: int,
        compile_modules: bool = False,
        compile_mode: str = "default",
) -> WMEpisodeWindows:
    _validate_episode_for_wm_windows(episode, num_next_steps)
    if episode.final_local_obs is None:
        raise ValueError("final_local_obs must be provided to build world-model windows")
    if episode.final_global_obs is None:
        raise ValueError("final_global_obs must be provided to build world-model windows")

    batched_windows = _build_wm_windows_from_batched_tensors(
        local_obs=episode.local_obs.unsqueeze(0),
        global_obs=episode.global_obs.unsqueeze(0),
        actions=episode.actions.unsqueeze(0),
        final_local_obs=episode.final_local_obs.unsqueeze(0),
        final_global_obs=episode.final_global_obs.unsqueeze(0),
        agent_mask=None if episode.agent_mask is None else episode.agent_mask.unsqueeze(0),
        final_agent_mask=(
            None
            if episode.final_agent_mask is None
            else episode.final_agent_mask.unsqueeze(0)
        ),
        num_next_steps=num_next_steps,
        compile_modules=compile_modules,
        compile_mode=compile_mode,
    )
    return _squeeze_wm_episode_windows_batch_dim(batched_windows)


def build_wm_episode_windows_batch(
        episodes: list[PPOEpisodeSegment],
        *,
        num_next_steps: int,
        compile_modules: bool = False,
        compile_mode: str = "default",
) -> WMEpisodeWindows:
    if not episodes:
        raise ValueError("episodes must not be empty")

    expected_num_steps = _validate_episode_for_wm_windows(episodes[0], num_next_steps)
    has_agent_mask = episodes[0].agent_mask is not None

    local_obs_list: list[torch.Tensor] = []
    global_obs_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    final_local_obs_list: list[torch.Tensor] = []
    final_global_obs_list: list[torch.Tensor] = []
    agent_mask_list: list[torch.Tensor] = []
    final_agent_mask_list: list[torch.Tensor] = []

    for episode in episodes:
        num_steps = _validate_episode_for_wm_windows(episode, num_next_steps)
        if num_steps != expected_num_steps:
            raise ValueError(
                f"All episodes must have the same number of steps, got {num_steps} and {expected_num_steps}"
            )
        if (episode.agent_mask is not None) != has_agent_mask:
            raise ValueError("agent_mask must be provided for all episodes or none")
        if episode.final_local_obs is None:
            raise ValueError("final_local_obs must be provided to build world-model windows")
        if episode.final_global_obs is None:
            raise ValueError("final_global_obs must be provided to build world-model windows")

        local_obs_list.append(episode.local_obs)
        global_obs_list.append(episode.global_obs)
        actions_list.append(episode.actions)
        final_local_obs_list.append(episode.final_local_obs)
        final_global_obs_list.append(episode.final_global_obs)

        if has_agent_mask:
            if episode.agent_mask is None:
                raise ValueError("agent_mask must be provided when agent masks are enabled")
            if episode.final_agent_mask is None:
                raise ValueError("final_agent_mask must be provided when agent masks are enabled")
            agent_mask_list.append(episode.agent_mask)
            final_agent_mask_list.append(episode.final_agent_mask)

    return _build_wm_windows_from_batched_tensors(
        local_obs=torch.stack(tuple(local_obs_list), dim=0),
        global_obs=torch.stack(tuple(global_obs_list), dim=0),
        actions=torch.stack(tuple(actions_list), dim=0),
        final_local_obs=torch.stack(tuple(final_local_obs_list), dim=0),
        final_global_obs=torch.stack(tuple(final_global_obs_list), dim=0),
        agent_mask=None if not has_agent_mask else torch.stack(tuple(agent_mask_list), dim=0),
        final_agent_mask=(
            None
            if not has_agent_mask
            else torch.stack(tuple(final_agent_mask_list), dim=0)
        ),
        num_next_steps=num_next_steps,
        compile_modules=compile_modules,
        compile_mode=compile_mode,
    )


def _build_time_windows(
        tensor: torch.Tensor,
        *,
        window_size: int,
        num_steps: int,
) -> torch.Tensor:
    windows = tensor.unfold(0, window_size, 1)[:num_steps]
    if tensor.ndim == 1:
        return windows.contiguous()

    permute_dims = (0, tensor.ndim, *range(1, tensor.ndim))
    return windows.permute(*permute_dims).contiguous()


def _validate_episode_for_wm_windows(
        episode: PPOEpisodeSegment,
        num_next_steps: int,
) -> int:
    if num_next_steps < 1:
        raise ValueError(f"num_next_steps must be >= 1, got {num_next_steps}")

    num_steps = int(episode.local_obs.shape[0])
    if episode.actions.shape[0] != num_steps:
        raise ValueError(
            f"Expected actions to have {num_steps} timesteps, got {episode.actions.shape[0]}"
        )
    if episode.global_obs.shape[0] != num_steps:
        raise ValueError(
            f"Expected global_obs to have {num_steps} timesteps, got {episode.global_obs.shape[0]}"
        )
    if episode.final_local_obs is None:
        raise ValueError("final_local_obs must be provided to build world-model windows")
    if episode.final_global_obs is None:
        raise ValueError("final_global_obs must be provided to build world-model windows")

    has_agent_mask = episode.agent_mask is not None
    if has_agent_mask != (episode.final_agent_mask is not None):
        raise ValueError("agent_mask and final_agent_mask must either both be set or both be None")

    return num_steps


def _build_wm_windows_from_batched_tensors(
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        final_local_obs: torch.Tensor,
        final_global_obs: torch.Tensor,
        agent_mask: MaybeTensor,
        final_agent_mask: MaybeTensor,
        num_next_steps: int,
        compile_modules: bool = False,
        compile_mode: str = "default",
) -> WMEpisodeWindows:
    outputs_fn = _build_wm_window_outputs_impl
    if compile_modules:
        outputs_fn = _get_compiled_wm_window_outputs_fn(compile_mode=compile_mode)
    (
        multi_step_actions,
        next_local_obs,
        wm_target_time_mask,
        next_global_obs,
        wm_agent_mask,
        wm_loss_agent_mask,
    ) = outputs_fn(
        local_obs,
        global_obs,
        actions,
        final_local_obs,
        final_global_obs,
        agent_mask,
        final_agent_mask,
        num_next_steps,
    )
    return WMEpisodeWindows(
        multi_step_actions=multi_step_actions,
        next_local_obs=next_local_obs,
        wm_target_time_mask=wm_target_time_mask,
        next_global_obs=next_global_obs,
        wm_agent_mask=wm_agent_mask,
        wm_loss_agent_mask=wm_loss_agent_mask,
    )


def _build_wm_window_outputs_impl(
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        final_local_obs: torch.Tensor,
        final_global_obs: torch.Tensor,
        agent_mask: MaybeTensor,
        final_agent_mask: MaybeTensor,
        num_next_steps: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    MaybeTensor,
    MaybeTensor,
]:
    batch_size, num_steps = local_obs.shape[:2]
    has_agent_mask = agent_mask is not None

    next_local_obs = torch.cat((local_obs[:, 1:], final_local_obs.unsqueeze(1)), dim=1)
    next_global_obs = torch.cat((global_obs[:, 1:], final_global_obs.unsqueeze(1)), dim=1)

    wm_agent_mask: MaybeTensor = None
    wm_loss_agent_mask: MaybeTensor = None
    if has_agent_mask:
        if final_agent_mask is None:
            raise ValueError("final_agent_mask must be provided when agent masks are enabled")
        wm_agent_mask = agent_mask
        wm_loss_agent_mask = torch.cat(
            (agent_mask[:, 1:], final_agent_mask.unsqueeze(1)),
            dim=1,
        )

    padding_len = num_next_steps - 1
    padded_actions = _pad_batched_time_axis(actions, padding_len=padding_len, pad_value=0)
    padded_next_local_obs = _pad_batched_time_axis(next_local_obs, padding_len=padding_len, pad_value=0)
    padded_next_global_obs = _pad_batched_time_axis(next_global_obs, padding_len=padding_len, pad_value=0)
    padded_validity = _pad_batched_time_axis(
        torch.ones((batch_size, num_steps), dtype=torch.bool, device=local_obs.device),
        padding_len=padding_len,
        pad_value=False,
    )

    padded_wm_agent_mask: MaybeTensor = None
    padded_wm_loss_agent_mask: MaybeTensor = None
    if has_agent_mask:
        assert wm_agent_mask is not None
        assert wm_loss_agent_mask is not None
        padded_wm_agent_mask = _pad_batched_time_axis(wm_agent_mask, padding_len=padding_len, pad_value=True)
        padded_wm_loss_agent_mask = _pad_batched_time_axis(
            wm_loss_agent_mask,
            padding_len=padding_len,
            pad_value=True,
        )

    return (
        _build_batched_time_windows(
            padded_actions,
            window_size=num_next_steps,
            num_steps=num_steps,
        ),
        _build_batched_time_windows(
            padded_next_local_obs,
            window_size=num_next_steps,
            num_steps=num_steps,
        ),
        _build_batched_time_windows(
            padded_validity,
            window_size=num_next_steps,
            num_steps=num_steps,
        ),
        _build_batched_time_windows(
            padded_next_global_obs,
            window_size=num_next_steps,
            num_steps=num_steps,
        ),
        (
            None
            if padded_wm_agent_mask is None
            else _build_batched_time_windows(
                padded_wm_agent_mask,
                window_size=num_next_steps,
                num_steps=num_steps,
            )
        ),
        (
            None
            if padded_wm_loss_agent_mask is None
            else _build_batched_time_windows(
                padded_wm_loss_agent_mask,
                window_size=num_next_steps,
                num_steps=num_steps,
            )
        ),
    )


@lru_cache(maxsize=None)
def _ensure_wm_window_helper_compile_available_cached(compile_mode: str) -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("WM window helper compile requires torch.compile support.")
    if sys.platform.startswith("win") and shutil.which("cl.exe") is None:
        raise RuntimeError(
            "WM window helper compile on this Windows setup requires cl.exe on PATH for torch.compile."
        )


def ensure_wm_window_helper_compile_available(*, compile_mode: str) -> None:
    if not compile_mode:
        raise ValueError("WM window helper compile_mode must be a non-empty string when compile is enabled.")
    _ensure_wm_window_helper_compile_available_cached(compile_mode)


def _get_compiled_wm_window_outputs_fn(*, compile_mode: str) -> _CompiledWMWindowOutputsFn:
    cached = _compiled_wm_window_outputs_fns.get(compile_mode)
    if cached is not None:
        return cached

    ensure_wm_window_helper_compile_available(compile_mode=compile_mode)
    compiled_fn = torch.compile(
        _build_wm_window_outputs_impl,
        mode=compile_mode,
    )
    _compiled_wm_window_outputs_fns[compile_mode] = compiled_fn
    return compiled_fn


def _pad_batched_time_axis(
        tensor: torch.Tensor,
        *,
        padding_len: int,
        pad_value: bool | int | float = 0,
) -> torch.Tensor:
    if padding_len < 0:
        raise ValueError(f"padding_len must be >= 0, got {padding_len}")
    if padding_len == 0:
        return tensor

    padding = torch.full(
        (tensor.shape[0], padding_len, *tensor.shape[2:]),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=1)


def _build_batched_time_windows(
        tensor: torch.Tensor,
        *,
        window_size: int,
        num_steps: int,
) -> torch.Tensor:
    windows = tensor.unfold(1, window_size, 1)[:, :num_steps]
    if tensor.ndim == 2:
        return windows.contiguous()

    permute_dims = (0, 1, windows.ndim - 1, *range(2, windows.ndim - 1))
    return windows.permute(*permute_dims).contiguous()


def _squeeze_wm_episode_windows_batch_dim(windows: WMEpisodeWindows) -> WMEpisodeWindows:
    return WMEpisodeWindows(
        multi_step_actions=windows.multi_step_actions.squeeze(0),
        next_local_obs=windows.next_local_obs.squeeze(0),
        wm_target_time_mask=windows.wm_target_time_mask.squeeze(0),
        next_global_obs=windows.next_global_obs.squeeze(0),
        wm_agent_mask=None if windows.wm_agent_mask is None else windows.wm_agent_mask.squeeze(0),
        wm_loss_agent_mask=(
            None
            if windows.wm_loss_agent_mask is None
            else windows.wm_loss_agent_mask.squeeze(0)
        ),
    )
