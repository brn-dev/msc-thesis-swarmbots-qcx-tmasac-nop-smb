from __future__ import annotations

import torch


def quat_to_rot6d_torch(quats: torch.Tensor) -> torch.Tensor:
    if quats.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {tuple(quats.shape)}")

    q = quats / torch.clamp(torch.linalg.vector_norm(quats, dim=-1, keepdim=True), min=1e-8)
    w, x, y, z = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)

    return torch.stack((r00, r10, r20, r01, r11, r21), dim=-1)
