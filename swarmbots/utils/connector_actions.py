from __future__ import annotations

import numpy as np
from gymnasium import spaces


def connector_action_space(shape: tuple[int, ...], *, continuous: bool) -> spaces.Box | spaces.MultiBinary:
    if continuous:
        return spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
    return spaces.MultiBinary(shape)
