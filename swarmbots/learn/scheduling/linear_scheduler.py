from dataclasses import dataclass
from typing import Optional, Any

from swarmbots.learn.scheduling.chainable_scheduler import ChainableScheduler, SchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleResult, ScheduleUnit


@dataclass(frozen=True)
class LinearSchedulerConfig(SchedulerConfig):
    duration: int
    start_value: float
    final_value: float


class LinearScheduler(ChainableScheduler):

    def __init__(
            self,
            unit: ScheduleUnit,
            duration: int,
            start_value: float,
            final_value: float,
            name: Optional[str] = None
    ):
        super().__init__(unit=unit, name=name)

        self.duration = duration
        self.start_value = start_value
        self.final_value = final_value

    def get_duration(self) -> int:
        return self.duration

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"unit={self.unit.name}, "
            f"duration={self.duration}, "
            f"start_value={self.start_value}, "
            f"final_value={self.final_value}, "
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
            new_value = self.start_value + (self.final_value - self.start_value) * progress

        if abs(new_value - old_value) < 1e-8:
            return {"new_value": None, "event": self.name_prefix + "hold"}
        return {"new_value": new_value, "event": self.name_prefix + "linear"}
