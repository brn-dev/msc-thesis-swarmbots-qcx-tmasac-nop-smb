import abc
from dataclasses import dataclass
from typing import Optional, Any

from swarmbots.learn.scheduling.schedulers import ScheduleResult, ScheduleUnit


@dataclass(frozen=True)
class SchedulerConfig:
    unit: ScheduleUnit

class ChainableScheduler(abc.ABC):

    def __init__(
            self,
            unit: ScheduleUnit,
            name: Optional[str]
    ):
        self.unit = unit

        self.name = name
        self.name_prefix = name + "-" if name else ""

    def __call__(
            self,
            old_value: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        t = ScheduleUnit.get(
            self.unit,
            n_iterations=n_iterations,
            n_model_updates=n_model_updates,
            n_timesteps=n_timesteps
        )
        duration = self.get_duration()
        progress = 1.0 if duration <= 0 else t / duration
        return self.schedule(
            progress=progress,
            old_value=old_value,
            state=state,
            metrics=metrics,
        )

    @abc.abstractmethod
    def schedule(
            self,
            progress: float,
            old_value: float,
            state: dict[str, Any],
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_duration(self) -> int:
        raise NotImplementedError()



# class ScheduleChain(ChainableScheduler):
#
#     def __init__(self, unit: ScheduleUnit, schedulers: list[ChainableScheduler]):
#         super().__init__(unit)
#         self.schedulers = schedulers
#
#         self.scheduler_durations = [s.get_duration() for s in schedulers]
#         self.scheduler_starts = [0] + np.cumsum(self.scheduler_durations).tolist()[:-1]
#
#         self.duration = sum(self.scheduler_durations)
#
#         self.scheduler_idx = 0
#
#     def get_duration(self) -> int:
#         return self.duration
#
#     def schedule(
#             self,
#             progress: float,
#             old_value: float,
#             state: dict[str, Any],
#             metrics: dict[str, Any],
#     ) -> ScheduleResult:
#
#
#
#
# class ScheduleChainBuilder:
#
#     def __init__(self, unit: ScheduleUnit):
#         super().__init__(unit)
#         self.schedulers: list[tuple[int, ChainableScheduler]] = []
#
#     def chain(self, scheduler: ChainableScheduler):
#         if len(self.schedulers) == 0:
#             self.schedulers.append((0, scheduler))
#             return
#
#         prev_start, prev_scheduler = self.schedulers[-1]
#         prev_end = prev_start + prev_scheduler.get_duration()
#         self.schedulers.append((prev_end, scheduler))
