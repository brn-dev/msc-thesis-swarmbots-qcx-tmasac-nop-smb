import torch


def check_sequence_inputs(inputs: torch.Tensor) -> tuple[int, int, int]:
    if inputs.ndim != 3:
        raise ValueError(f"Expected inputs shape (B, T, H), got {tuple(inputs.shape)}")
    return inputs.shape


def check_mask(
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        sequence_length: int,
        name: str,
) -> None:
    if mask is None:
        return
    if mask.shape != (batch_size, sequence_length):
        raise ValueError(f"Expected {name} shape ({batch_size}, {sequence_length}), got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise ValueError(f"Expected {name} dtype torch.bool, got {mask.dtype}")


def reset_state(state: tuple[torch.Tensor, ...], reset_t: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(
        item.masked_fill(reset_t.reshape(-1, *([1] * (item.ndim - 1))), 0.0)
        for item in state
    )


def select_state(
        next_state: tuple[torch.Tensor, ...],
        previous_state: tuple[torch.Tensor, ...],
        valid_t: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.where(valid_t.reshape(-1, *([1] * (next_item.ndim - 1))), next_item, previous_item)
        for next_item, previous_item in zip(next_state, previous_state)
    )
