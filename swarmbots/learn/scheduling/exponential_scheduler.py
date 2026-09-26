import math
from dataclasses import dataclass
from typing import Optional, Any

from swarmbots.learn.scheduling.chainable_scheduler import ChainableScheduler, SchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleResult, ScheduleUnit

@dataclass(frozen=True)
class ExponentialSchedulerConfig(SchedulerConfig):
    duration: int
    start_value: float
    final_value: float
    base: float = math.e

class ExponentialScheduler(ChainableScheduler):

    def __init__(
            self,
            unit: ScheduleUnit,
            duration: int,
            start_value: float,
            final_value: float,
            base: float = math.e,
            name: Optional[str] = None
    ):
        super().__init__(unit=unit, name=name)
        if not math.isfinite(base) or base <= 0:
            raise ValueError(f"base must be finite and > 0, got {base}")

        self.duration = duration
        self.start_value = start_value
        self.final_value = final_value
        self.base = base

    def get_duration(self) -> int:
        return self.duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"unit={self.unit.name}, "
            f"duration={self.duration}, "
            f"start_value={self.start_value}, "
            f"final_value={self.final_value}, "
            f"base={self.base}, "
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
            progress = min(progress, 1.0)
            if math.isclose(self.base, 1.0):
                interpolation = progress
            else:
                interpolation = (self.base ** progress - 1.0) / (self.base - 1.0)

            new_value = self.start_value + (self.final_value - self.start_value) * interpolation

        if abs(new_value - old_value) < 1e-8:
            return {"new_value": None, "event": self.name_prefix + "hold"}
        return {"new_value": new_value, "event": self.name_prefix + "exponential"}
