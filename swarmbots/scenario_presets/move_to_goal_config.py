from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, UniformDistParams, eval_fodp


@dataclass(frozen=True, slots=True)
class AbsoluteGoalConfig:
    x: FloatOrDistParams = 0.0
    y: FloatOrDistParams = 3.0


@dataclass(frozen=True, slots=True)
class RelativePolarGoalConfig:
    distance: FloatOrDistParams
    angle: FloatOrDistParams = UniformDistParams(0.0, 2.0 * math.pi)


MoveToGoalConfig = AbsoluteGoalConfig | RelativePolarGoalConfig


def sample_move_to_goal_position(
    *,
    goal: MoveToGoalConfig,
    rng: np.random.Generator,
    swarm_start_location: np.ndarray,
) -> np.ndarray:
    if isinstance(goal, AbsoluteGoalConfig):
        return np.array(
            [
                eval_fodp(goal.x, rng),
                eval_fodp(goal.y, rng),
            ],
            dtype=float,
        )
    if isinstance(goal, RelativePolarGoalConfig):
        distance = eval_fodp(goal.distance, rng)
        angle = eval_fodp(goal.angle, rng)
        return np.array(
            [
                float(swarm_start_location[0]) + distance * math.cos(angle),
                float(swarm_start_location[1]) + distance * math.sin(angle),
            ],
            dtype=float,
        )
    raise TypeError(f"Unsupported move-to goal config: {type(goal).__name__}")
