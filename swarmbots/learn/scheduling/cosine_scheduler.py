import math
from dataclasses import dataclass
from typing import Any, Optional

from swarmbots.learn.scheduling.chainable_scheduler import ChainableScheduler, SchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleResult, ScheduleUnit


@dataclass(frozen=True)
class CosineSchedulerConfig(SchedulerConfig):
    duration: int
    start_value: float
    final_value: float
    bias: float = 1.0
    sharpness: float = 1.0


class CosineScheduler(ChainableScheduler):

    def __init__(
            self,
            unit: ScheduleUnit,
            duration: int,
            start_value: float,
            final_value: float,
            bias: float = 1.0,
            sharpness: float = 1.0,
            name: Optional[str] = None
    ):
        super().__init__(unit=unit, name=name)
        if not math.isfinite(bias) or bias <= 0:
            raise ValueError(f"bias must be finite and > 0, got {bias}")
        if not math.isfinite(sharpness) or sharpness <= 0:
            raise ValueError(f"sharpness must be finite and > 0, got {sharpness}")

        self.duration = duration
        self.start_value = start_value
        self.final_value = final_value
        self.bias = bias
        self.sharpness = sharpness

    def get_duration(self) -> int:
        return self.duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"unit={self.unit.name}, "
            f"duration={self.duration}, "
            f"start_value={self.start_value}, "
            f"final_value={self.final_value}, "
            f"bias={self.bias}, "
            f"sharpness={self.sharpness}, "
            f"name={self.name!r})"
        )

    def schedule(
            self,
            progress: float,
            old_value: float,
            state: dict[str, Any],
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        if self.duration <= 0:
            new_value = self.final_value
        else:
            progress = min(max(progress, 0.0), 1.0)
            biased_progress = progress ** self.bias
            interpolation = 0.5 * (1.0 + math.cos(math.pi * biased_progress))
            decay_progress = 1.0 - interpolation
            decay_progress = self._apply_symmetric_sharpness(decay_progress, self.sharpness)
            interpolation = 1.0 - decay_progress
            new_value = self.final_value + (self.start_value - self.final_value) * interpolation

        if abs(new_value - old_value) < 1e-8:
            return {"new_value": None, "event": self.name_prefix + "hold"}
        return {"new_value": new_value, "event": self.name_prefix + "cosine_annealing"}

    @staticmethod
    def _apply_symmetric_sharpness(value: float, sharpness: float) -> float:
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        if value < 0.5:
            return 0.5 * (2.0 * value) ** sharpness
        return 1.0 - 0.5 * (2.0 * (1.0 - value)) ** sharpness
