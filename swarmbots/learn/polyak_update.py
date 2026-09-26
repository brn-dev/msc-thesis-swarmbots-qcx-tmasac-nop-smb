import torch
from torch import nn


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"{tau=} must be in [0, 1]")

    source_parameters = tuple(source.parameters())
    target_parameters = tuple(target.parameters())
    if len(source_parameters) != len(target_parameters):
        raise ValueError(
            "Source and target modules must have the same number of parameters: "
            f"{len(source_parameters)} != {len(target_parameters)}"
        )
    if target_parameters:
        torch._foreach_lerp_(target_parameters, source_parameters, tau)

    for src_buf, tgt_buf in zip(source.buffers(), target.buffers(), strict=True):
        tgt_buf.copy_(src_buf)
