from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlattenedRecurrentWMBatch:
    local_latents: torch.Tensor
    next_local_obs: torch.Tensor
    actions: torch.Tensor
    local_obs: torch.Tensor | None
    next_global_obs: torch.Tensor | None
    agent_mask: torch.Tensor | None
    loss_agent_mask: torch.Tensor | None
    time_mask: torch.Tensor | None


def flatten_recurrent_wm_batch(
        *,
        local_latents: torch.Tensor,
        next_local_obs: torch.Tensor,
        actions: torch.Tensor,
        local_obs: torch.Tensor | None = None,
        next_global_obs: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        loss_agent_mask: torch.Tensor | None = None,
        time_mask: torch.Tensor | None = None,
) -> FlattenedRecurrentWMBatch | None:
    if local_latents.ndim == 3:
        return None
    if local_latents.ndim != 4:
        raise ValueError(
            f"Expected local_latents shape (B, N, D) or (B, S, N, D), got {tuple(local_latents.shape)}"
        )
    if actions.ndim != 5:
        raise ValueError(f"Expected recurrent actions shape (B, S, T, N, A), got {tuple(actions.shape)}")
    if next_local_obs.ndim != 5:
        raise ValueError(
            f"Expected recurrent next_local_obs shape (B, S, T, N, F), got {tuple(next_local_obs.shape)}"
        )

    batch_size, sequence_length, n_agents, latent_dim = local_latents.shape
    num_next_steps = actions.shape[2]
    flat_batch_size = batch_size * sequence_length

    flat_local_obs = None
    if local_obs is not None:
        flat_local_obs = local_obs.reshape(flat_batch_size, n_agents, local_obs.shape[-1])

    flat_next_global_obs = None
    if next_global_obs is not None:
        if next_global_obs.ndim == 4:
            flat_next_global_obs = next_global_obs.reshape(
                flat_batch_size,
                num_next_steps,
                next_global_obs.shape[-1],
            )
        elif next_global_obs.ndim == 3:
            flat_next_global_obs = next_global_obs.reshape(flat_batch_size, next_global_obs.shape[-1])
        else:
            raise ValueError(f"Expected recurrent next_global_obs ndim 3 or 4, got {next_global_obs.ndim}")

    flat_time_mask = None
    if time_mask is not None:
        if time_mask.dtype != torch.bool:
            raise ValueError(f"Expected time_mask dtype torch.bool, got {time_mask.dtype}")
        flat_time_mask = time_mask.reshape(flat_batch_size, num_next_steps)

    return FlattenedRecurrentWMBatch(
        local_latents=local_latents.reshape(flat_batch_size, n_agents, latent_dim),
        next_local_obs=next_local_obs.reshape(
            flat_batch_size,
            num_next_steps,
            n_agents,
            next_local_obs.shape[-1],
        ),
        actions=actions.reshape(
            flat_batch_size,
            num_next_steps,
            n_agents,
            actions.shape[-1],
        ),
        local_obs=flat_local_obs,
        next_global_obs=flat_next_global_obs,
        agent_mask=_flatten_recurrent_mask(
            mask=agent_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            num_next_steps=num_next_steps,
            name="agent_mask",
        ),
        loss_agent_mask=_flatten_recurrent_mask(
            mask=loss_agent_mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_agents=n_agents,
            num_next_steps=num_next_steps,
            name="loss_agent_mask",
        ),
        time_mask=flat_time_mask,
    )


def build_wm_target_time_mask(
        *,
        wm_target_time_mask: torch.Tensor | None,
        sequence_time_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if wm_target_time_mask is None or sequence_time_mask is None:
        return wm_target_time_mask
    if wm_target_time_mask.dtype != torch.bool:
        raise ValueError(f"Expected wm_target_time_mask dtype torch.bool, got {wm_target_time_mask.dtype}")
    if sequence_time_mask.dtype != torch.bool:
        raise ValueError(f"Expected sequence_time_mask dtype torch.bool, got {sequence_time_mask.dtype}")
    if wm_target_time_mask.ndim != sequence_time_mask.ndim + 1:
        raise ValueError(
            f"Expected wm_target_time_mask ndim {sequence_time_mask.ndim + 1}, "
            f"got {wm_target_time_mask.ndim}"
        )
    if wm_target_time_mask.shape[:-1] != sequence_time_mask.shape:
        raise ValueError(
            f"Expected wm_target_time_mask prefix shape {tuple(sequence_time_mask.shape)}, "
            f"got {tuple(wm_target_time_mask.shape[:-1])}"
        )
    return wm_target_time_mask & sequence_time_mask.unsqueeze(-1)


def _flatten_recurrent_mask(
        *,
        mask: torch.Tensor | None,
        batch_size: int,
        sequence_length: int,
        n_agents: int,
        num_next_steps: int,
        name: str,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise ValueError(f"Expected {name} dtype torch.bool, got {mask.dtype}")

    flat_batch_size = batch_size * sequence_length
    if mask.ndim == 3:
        return mask.reshape(flat_batch_size, n_agents)
    if mask.ndim == 4:
        return mask.reshape(flat_batch_size, num_next_steps, n_agents)
    raise ValueError(f"Expected {name} ndim 3 or 4, got {mask.ndim}")
