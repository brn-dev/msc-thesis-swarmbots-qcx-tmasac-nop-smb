from __future__ import annotations

import json
from typing import Any, Callable, Mapping


DEFAULT_RECORDING_SCHEDULE: dict[float, int] = {
    25: 5,
    50: 5,
    75: 5,
    90: 5,
    100: 5,
}

DEFAULT_LIVE_RECORDING_SCHEDULE: dict[float, int] = {
    25: 5,
    50: 5,
    75: 5,
    90: 5,
    95: 5,
}


def format_recording_percentage(percentage: float) -> str:
    if float(percentage).is_integer():
        return f"{int(percentage):03d}"
    return str(percentage).rstrip("0").rstrip(".").replace(".", "p")


def install_scheduled_recordings(
    *,
    algorithm: Any,
    total_timesteps: int,
    start_timesteps: int = 0,
    schedule: Mapping[float, int],
) -> Callable[[Any, dict[str, Any], int], None]:
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be positive, got {total_timesteps}")
    if start_timesteps < 0 or start_timesteps >= total_timesteps:
        raise ValueError(
            "start_timesteps must be in [0, total_timesteps), got "
            f"{start_timesteps=} {total_timesteps=}"
        )

    pending_milestones: list[tuple[float, int, int]] = []
    current_timesteps = int(algorithm.n_total_timesteps)
    seen_targets: set[int] = set()
    for percentage, num_episodes in sorted(schedule.items()):
        if percentage < 0 or percentage > 100:
            raise ValueError(f"Recording percentage must be in [0, 100], got {percentage}")
        if num_episodes <= 0:
            raise ValueError(f"Scheduled recording episode count must be positive, got {num_episodes}")

        target_timesteps = max(
            start_timesteps + 1,
            start_timesteps
            + int((total_timesteps - start_timesteps) * percentage / 100),
        )
        if target_timesteps in seen_targets or target_timesteps <= current_timesteps:
            continue
        seen_targets.add(target_timesteps)
        pending_milestones.append((percentage, target_timesteps, num_episodes))

    def scheduled_recording_hook(
        algorithm: Any,
        metrics: dict[str, Any],
        rollout_steps: int,
    ) -> None:
        _ = metrics, rollout_steps
        while pending_milestones and algorithm.n_total_timesteps >= pending_milestones[0][1]:
            percentage, _target_timesteps, num_recording_episodes = pending_milestones.pop(0)
            actual_timesteps = algorithm.n_total_timesteps
            percentage_label = format_recording_percentage(percentage)
            prefix = f"record_{percentage_label}pct_{actual_timesteps}_steps"
            algorithm.execute_command(
                "record",
                json.dumps({"episodes": num_recording_episodes, "prefix": prefix}),
                extra_run_metadata=None,
            )

    return scheduled_recording_hook
