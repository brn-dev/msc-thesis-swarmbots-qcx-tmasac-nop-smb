from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NotRequired, Optional, Protocol, TypedDict

from swarmbots.learn.serialization_utils import serialize_fn


class ScheduleResult(TypedDict):
    new_value: Optional[float]
    msg: NotRequired[str]
    event: NotRequired[str]


class Scheduler(Protocol):
    def __call__(
            self,
            old_value: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> ScheduleResult:
        ...


class ScheduleApplier(Protocol):
    def __call__(self, new_value: float) -> None:
        ...


class ScheduleValueGetter(Protocol):
    def __call__(self) -> float:
        ...


@dataclass(slots=True)
class ScheduledHyperParameter:
    name: str
    scheduler: Scheduler
    get_value: ScheduleValueGetter
    apply: ScheduleApplier
    enabled: bool = True
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SchedulerStepResult:
    name: str
    old_value: float
    new_value: float
    updated: bool
    event: str | None
    msg: str | None


class SchedulerManager:
    def __init__(
            self,
            scheduled_hyper_parameters: list[ScheduledHyperParameter] | None = None,
    ) -> None:
        self.scheduled_hyper_parameters: list[ScheduledHyperParameter] = (
            [] if scheduled_hyper_parameters is None else list(scheduled_hyper_parameters)
        )

    def add(self, scheduled_hyper_parameter: ScheduledHyperParameter) -> None:
        self.scheduled_hyper_parameters.append(scheduled_hyper_parameter)

    def step(
            self,
            *,
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            metrics: dict[str, Any],
    ) -> list[SchedulerStepResult]:
        step_results: list[SchedulerStepResult] = []
        for scheduled in self.scheduled_hyper_parameters:
            if not scheduled.enabled:
                continue

            old_value = float(scheduled.get_value())
            update_result = scheduled.scheduler(
                old_value=old_value,
                state=scheduled.state,
                n_iterations=n_iterations,
                n_model_updates=n_model_updates,
                n_timesteps=n_timesteps,
                metrics=metrics,
            )

            requested_value = update_result.get("new_value")
            event = update_result.get("event")
            msg = update_result.get("msg")

            if requested_value is None:
                step_results.append(
                    SchedulerStepResult(
                        name=scheduled.name,
                        old_value=old_value,
                        new_value=old_value,
                        updated=False,
                        event=event,
                        msg=msg,
                    )
                )
                continue

            scheduled.apply(float(requested_value))
            new_value = float(scheduled.get_value())
            step_results.append(
                SchedulerStepResult(
                    name=scheduled.name,
                    old_value=old_value,
                    new_value=new_value,
                    updated=(new_value != old_value),
                    event=event,
                    msg=msg,
                )
            )

        return step_results

    def serialize(self) -> list[dict[str, Any]]:
        return [
            {
                "name": scheduled.name,
                "enabled": scheduled.enabled,
                "scheduler": serialize_fn(scheduled.scheduler),
                "get_value": serialize_fn(scheduled.get_value),
                "apply": serialize_fn(scheduled.apply),
            }
            for scheduled in self.scheduled_hyper_parameters
        ]

class ScheduleUnit(Enum):
    ITERATIONS = 1
    MODEL_UPDATES = 2
    TIMESTEPS = 3

    @staticmethod
    def get(
            unit: 'ScheduleUnit',
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int
    ) -> int:
        if unit == ScheduleUnit.ITERATIONS:
            return n_iterations
        if unit == ScheduleUnit.MODEL_UPDATES:
            return n_model_updates
        if unit == ScheduleUnit.TIMESTEPS:
            return n_timesteps
        raise ValueError(f'Invalid unit {unit}')
