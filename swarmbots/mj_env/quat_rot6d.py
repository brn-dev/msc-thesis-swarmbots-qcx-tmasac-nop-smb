import numpy as np


def quat_to_rot6d(quats: np.ndarray, axis: int) -> np.ndarray:
    if quats.shape[axis] != 4:
        raise ValueError(f"Expected quaternion size 4 on axis {axis}, got {quats.shape[axis]}")

    q = np.moveaxis(quats, axis, -1).astype(np.float64, copy=False)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norms, 1e-8)

    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

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

    rot6d = np.stack([r00, r10, r20, r01, r11, r21], axis=-1)
    return np.moveaxis(rot6d, -1, axis)