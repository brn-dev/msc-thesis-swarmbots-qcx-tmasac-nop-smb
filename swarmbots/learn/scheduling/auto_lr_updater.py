from typing import Any, Optional

import numpy as np

from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRateUpdateResult, AutomaticLearningRateUpdater
from swarmbots.learn.scheduling.scheduler_factory import SchedulerFactoryConfig, make_scheduler
from swarmbots.learn.scheduling.schedulers import ScheduleUnit
from swarmbots.learn.summary_statistics import SummaryStatistics


EARLY_STOP_EPOCH_DECAY_FACTORS = {
    1: 0.8,
    2: 0.85,
}


def make_auto_lr_updater(
        warm_scheduler_config: SchedulerFactoryConfig | None = None,
        early_stop_epoch_decay_limit: int = 4,
        max_kl_div: float = 0.0125,
) -> AutomaticLearningRateUpdater:
    if warm_scheduler_config is not None and warm_scheduler_config.unit != ScheduleUnit.ITERATIONS:
        raise ValueError(
            "make_auto_lr_updater only supports warm_scheduler_config.unit == "
            f"{ScheduleUnit.ITERATIONS.name}, got {warm_scheduler_config.unit.name}"
        )

    warmup_scheduler = None if warm_scheduler_config is None else make_scheduler(warm_scheduler_config)
    warmup_duration = None if warm_scheduler_config is None else warm_scheduler_config.duration

    def auto_lr_updater(
            old_lr: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        if early_stop_epoch is not None and early_stop_epoch == 0:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = 0.75
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'epoch={early_stop_epoch}',
                'event': 'zero_epoch_hit'
            }

        if early_stop_kl_div is not None and early_stop_kl_div > max_kl_div:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = np.clip(0.9 - early_stop_kl_div * 50, 0.4, 0.9)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'kl={early_stop_kl_div:.3f}',
                'event': 'max_kl_hit'
            }

        clip_frac_stats: Optional[SummaryStatistics] = metrics.get('clip_frac', None)
        if clip_frac_stats and clip_frac_stats.mean > 0.19:
            state['counter'] = 0
            state['warmup'] = False
            clip_frac = clip_frac_stats.mean
            decay_factor = np.clip(1 - clip_frac, 0.5, 0.9)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'{clip_frac=:.3f}',
                'event': 'max_clip_frac_hit'
            }

        if early_stop_epoch is not None and early_stop_epoch < early_stop_epoch_decay_limit:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = EARLY_STOP_EPOCH_DECAY_FACTORS.get(early_stop_epoch, 0.9)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'epoch={early_stop_epoch}',
                'event': 'min_epoch_hit'
            }

        warmup_unit = None if warm_scheduler_config is None else warm_scheduler_config.unit
        warmup_step = None
        if warmup_unit is not None:
            warmup_step = ScheduleUnit.get(
                unit=warmup_unit,
                n_iterations=n_iterations,
                n_model_updates=n_model_updates,
                n_timesteps=n_timesteps,
            )

        warmup: bool = (
            state.get('warmup', warmup_scheduler is not None)
            and warmup_scheduler is not None
            and warmup_duration is not None
            and warmup_step is not None
            and warmup_step <= warmup_duration
        )
        state['warmup'] = warmup
        if warmup:
            if warmup_scheduler is None:
                raise RuntimeError("warmup_scheduler must be configured when warmup is enabled")
            schedule_result = warmup_scheduler(
                old_value=old_lr,
                state=state,
                n_iterations=n_iterations,
                n_model_updates=n_model_updates,
                n_timesteps=n_timesteps,
                metrics=metrics,
            )
            new_lr = old_lr if schedule_result['new_value'] is None else schedule_result['new_value']
            return {
                'new_lr': new_lr,
                'msg': f'Warmup ({warmup_step}/{warmup_duration} {warmup_unit.name.lower()})',
                'event': 'warmup'
            }

        counter = state.get('counter', 0) + 1

        if counter >= 2:
            state['counter'] = 0
            return {'new_lr': old_lr * 1.1}

        state['counter'] = counter
        return {'new_lr': None}

    return auto_lr_updater
