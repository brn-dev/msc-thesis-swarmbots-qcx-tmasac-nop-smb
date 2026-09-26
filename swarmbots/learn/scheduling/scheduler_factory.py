from typing import TypeAlias

from swarmbots.learn.scheduling.chainable_scheduler import ChainableScheduler, SchedulerConfig
from swarmbots.learn.scheduling.cosine_scheduler import (
    CosineScheduler,
    CosineSchedulerConfig,
)
from swarmbots.learn.scheduling.exponential_scheduler import (
    ExponentialScheduler,
    ExponentialSchedulerConfig,
)
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler, LinearSchedulerConfig

SchedulerFactoryConfig: TypeAlias = (
        LinearSchedulerConfig
        | ExponentialSchedulerConfig
        | CosineSchedulerConfig
)


def make_scheduler(
        config: SchedulerConfig,
        name: str | None = None,
) -> ChainableScheduler:
    if isinstance(config, LinearSchedulerConfig):
        return LinearScheduler(
            unit=config.unit,
            duration=config.duration,
            start_value=config.start_value,
            final_value=config.final_value,
            name=name,
        )

    if isinstance(config, ExponentialSchedulerConfig):
        return ExponentialScheduler(
            unit=config.unit,
            duration=config.duration,
            start_value=config.start_value,
            final_value=config.final_value,
            base=config.base,
            name=name,
        )

    if isinstance(config, CosineSchedulerConfig):
        return CosineScheduler(
            unit=config.unit,
            duration=config.duration,
            start_value=config.start_value,
            final_value=config.final_value,
            bias=config.bias,
            sharpness=config.sharpness,
            name=name,
        )

    raise TypeError(f"Unsupported scheduler config type: {type(config).__name__}")


def scheduler_factory_make_scheduler(
        config: SchedulerConfig,
        name: str | None = None,
) -> ChainableScheduler:
    return make_scheduler(config=config, name=name)
