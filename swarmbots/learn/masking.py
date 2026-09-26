import torch


def build_valid_mask(
    *,
    base_shape: tuple[int, ...],
    device: torch.device,
    agent_mask: torch.Tensor | None,
    time_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if agent_mask is None and time_mask is None:
        return None

    valid = torch.ones(base_shape, dtype=torch.bool, device=device)

    if agent_mask is not None:
        if agent_mask.dtype != torch.bool:
            raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")
        if len(base_shape) == 2:
            if agent_mask.shape != base_shape:
                raise ValueError(f"Expected agent_mask shape {base_shape}, got {tuple(agent_mask.shape)}")
            valid = valid & agent_mask
        elif len(base_shape) == 3:
            b, t, n = base_shape
            if agent_mask.ndim == 2:
                if agent_mask.shape != (b, n):
                    raise ValueError(f"Expected agent_mask shape (B, N)=({b}, {n}), got {tuple(agent_mask.shape)}")
                valid = valid & agent_mask[:, None, :]
            elif agent_mask.ndim == 3:
                if agent_mask.shape != (b, t, n):
                    raise ValueError(
                        f"Expected agent_mask shape (B, T, N)=({b}, {t}, {n}), got {tuple(agent_mask.shape)}"
                    )
                valid = valid & agent_mask
            else:
                raise ValueError(f"Expected agent_mask ndim 2 or 3, got {agent_mask.ndim}")
        else:
            raise ValueError(f"Unsupported base_shape rank {len(base_shape)}")

    if time_mask is not None:
        if len(base_shape) != 3:
            raise ValueError("time_mask is only supported for multi-step loss (actions shape (B, T, N, A))")
        b, t, _n = base_shape
        if time_mask.shape != (b, t) or time_mask.dtype != torch.bool:
            raise ValueError(
                f"Expected time_mask shape (B, T)=({b}, {t}) and dtype bool, got {tuple(time_mask.shape)} / {time_mask.dtype}"
            )
        valid = valid & time_mask[:, :, None]

    return valid


def restrict_loss_agent_mask(
    *,
    base_shape: tuple[int, ...],
    loss_agent_mask: torch.Tensor | None,
    agent_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if loss_agent_mask is None and agent_mask is None:
        return None
    if loss_agent_mask is None:
        return agent_mask
    if agent_mask is None:
        return loss_agent_mask
    if loss_agent_mask.dtype != torch.bool:
        raise ValueError(f"Expected loss_agent_mask dtype bool, got {loss_agent_mask.dtype}")
    if agent_mask.dtype != torch.bool:
        raise ValueError(f"Expected agent_mask dtype bool, got {agent_mask.dtype}")

    if len(base_shape) == 2:
        if loss_agent_mask.shape != base_shape:
            raise ValueError(f"Expected loss_agent_mask shape {base_shape}, got {tuple(loss_agent_mask.shape)}")
        if agent_mask.shape != base_shape:
            raise ValueError(f"Expected agent_mask shape {base_shape}, got {tuple(agent_mask.shape)}")
        return loss_agent_mask & agent_mask

    if len(base_shape) != 3:
        raise ValueError(f"Unsupported base_shape rank {len(base_shape)}")

    b, t, n = base_shape
    if loss_agent_mask.ndim == 2:
        if loss_agent_mask.shape != (b, n):
            raise ValueError(f"Expected loss_agent_mask shape (B, N)=({b}, {n}), got {tuple(loss_agent_mask.shape)}")
    elif loss_agent_mask.ndim == 3:
        if loss_agent_mask.shape != (b, t, n):
            raise ValueError(
                f"Expected loss_agent_mask shape (B, T, N)=({b}, {t}, {n}), got {tuple(loss_agent_mask.shape)}"
            )
    else:
        raise ValueError(f"Expected loss_agent_mask ndim 2 or 3, got {loss_agent_mask.ndim}")

    if agent_mask.ndim == 2:
        if agent_mask.shape != (b, n):
            raise ValueError(f"Expected agent_mask shape (B, N)=({b}, {n}), got {tuple(agent_mask.shape)}")
    elif agent_mask.ndim == 3:
        if agent_mask.shape != (b, t, n):
            raise ValueError(
                f"Expected agent_mask shape (B, T, N)=({b}, {t}, {n}), got {tuple(agent_mask.shape)}"
            )
    else:
        raise ValueError(f"Expected agent_mask ndim 2 or 3, got {agent_mask.ndim}")

    if loss_agent_mask.ndim == 2 and agent_mask.ndim == 2:
        return loss_agent_mask & agent_mask
    if loss_agent_mask.ndim == 3 and agent_mask.ndim == 3:
        return loss_agent_mask & agent_mask
    if loss_agent_mask.ndim == 2:
        return loss_agent_mask[:, None, :] & agent_mask
    return loss_agent_mask & agent_mask[:, None, :]


def masked_mean(loss_per_item: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return loss_per_item.mean()
    if valid.shape != loss_per_item.shape:
        raise ValueError(
            f"Expected valid mask shape {tuple(loss_per_item.shape)}, got {tuple(valid.shape)}"
        )
    valid_f = valid.to(dtype=loss_per_item.dtype)
    denom = valid_f.sum().clamp_min(1.0)
    return (loss_per_item * valid_f).sum() / denom
