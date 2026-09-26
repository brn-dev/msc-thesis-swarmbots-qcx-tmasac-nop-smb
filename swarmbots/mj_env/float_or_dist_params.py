import abc
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger


@dataclass(frozen=True, slots=True)
class DistParams(abc.ABC):
    pass


@dataclass(frozen=True, slots=True)
class BoundedDistParams(DistParams, abc.ABC):
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class UniformDistParams(BoundedDistParams):
    @staticmethod
    def from_midpoint_and_width(midpoint: float, width: float) -> "UniformDistParams":
        width_half = width / 2.0
        return UniformDistParams(
            low=midpoint - width_half,
            high=midpoint + width_half
        )


@dataclass(frozen=True, slots=True)
class SplitUniformDistParams(BoundedDistParams):
    margin: float

    def __post_init__(self) -> None:
        if self.margin < 0.0:
            raise ValueError(f"Expected margin >= 0, got {self.margin}")
        if self.low >= -self.margin or self.high <= self.margin:
            raise ValueError(
                "Expected non-empty intervals on both sides of the margin, "
                f"got low={self.low}, high={self.high}, margin={self.margin}"
            )


TruncatedNormalDistSamplingMode = Literal['clamp', 'rejection']

@dataclass(frozen=True, slots=True)
class TruncatedNormalDistParams(BoundedDistParams):
    mean: float
    std: float
    sampling_mode: TruncatedNormalDistSamplingMode = 'clamp'

@dataclass(frozen=True, slots=True)
class NormalDistParams(DistParams):
    mean: float
    std: float


FloatOrDistParams = float | DistParams
FloatOrBoundedDistParams = float | BoundedDistParams
FloatOrDistParams2D = tuple[FloatOrDistParams, FloatOrDistParams]

REJECTION_SAMPLING_WARNING_THRESHOLD = 10


def fodp_low(fobdp: FloatOrBoundedDistParams) -> float:
    if isinstance(fobdp, (float, int)):
        return float(fobdp)
    return float(fobdp.low)


def eval_fodp(fodp: FloatOrDistParams, rng: np.random.Generator) -> float:
    if isinstance(fodp, (float, int)):
        return float(fodp)
    if isinstance(fodp, UniformDistParams):
        return float(rng.uniform(low=fodp.low, high=fodp.high))
    if isinstance(fodp, SplitUniformDistParams):
        if rng.integers(0, 2) == 0:
            return float(rng.uniform(low=fodp.low, high=-fodp.margin))
        return float(rng.uniform(low=fodp.margin, high=fodp.high))
    if isinstance(fodp, TruncatedNormalDistParams):
        return eval_truncated_normal(fodp, rng)
    if isinstance(fodp, NormalDistParams):
        return rng.normal(fodp.mean, fodp.std)
    raise ValueError(f'{fodp = }')


def eval_fodp_2d(
        fodp_2d: FloatOrDistParams2D,
        rng: np.random.Generator
) -> tuple[float, float]:
    return (
        eval_fodp(fodp_2d[0], rng),
        eval_fodp(fodp_2d[1], rng),
    )


def eval_fodp_3d(fodp_3d: tuple[FloatOrDistParams, FloatOrDistParams, FloatOrDistParams], rng: np.random.Generator):
    return (
        eval_fodp(fodp_3d[0], rng),
        eval_fodp(fodp_3d[1], rng),
        eval_fodp(fodp_3d[2], rng),
    )


def eval_truncated_normal(dp: TruncatedNormalDistParams, rng: np.random.Generator):
    x = rng.normal(dp.mean, dp.std)

    if dp.sampling_mode == 'clamp':
        x = np.clip(x, dp.low, dp.high)
    elif dp.sampling_mode == 'rejection':
        counter = 0
        while x < dp.low or x > dp.high:
            counter += 1
            x = rng.normal(dp.mean, dp.std)
        if counter > REJECTION_SAMPLING_WARNING_THRESHOLD:
            logger.warning(f'Rejected {counter} samples for {dp}')
    else:
        raise ValueError(f'{dp.sampling_mode } = ')

    return float(x)
