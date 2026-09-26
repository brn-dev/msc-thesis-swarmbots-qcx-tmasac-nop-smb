from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_experiment_common import run_thesis_ppo_experiment
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

ParallelEnvAlgorithmVariant = Literal["mat_ind", "mat_qcx"]

EXPERIMENT_RUN_NAME = "thesis_parallel_env_ablation_po_wall_medium"
ROLLOUT_BATCH_SIZE = 4096
PARALLEL_ENV_CONFIGS = (
    (1024, 4),
    (512, 8),
    (256, 16),
    (128, 32),
    (64, 64),
)
TRAINING_PARALLEL_ENV_CONFIGS = PARALLEL_ENV_CONFIGS[1:]
SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {"continuous_connector_actions": True},
)


def run_experiment(
    *,
    algorithm_variant: ParallelEnvAlgorithmVariant,
    num_envs: int,
    rollout_steps_per_env: int,
    entrypoint_path: Path,
) -> None:
    rollout_config = (num_envs, rollout_steps_per_env)
    if rollout_config not in PARALLEL_ENV_CONFIGS:
        raise ValueError(f"Unsupported parallel environment config: {rollout_config}")

    rollout_batch_size = num_envs * rollout_steps_per_env
    if rollout_batch_size != ROLLOUT_BATCH_SIZE:
        raise ValueError(
            f"Expected rollout batch size {ROLLOUT_BATCH_SIZE}, got {rollout_batch_size}"
        )

    run_thesis_ppo_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        variant=algorithm_variant,
        entrypoint_path=entrypoint_path,
        num_envs=num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        variant_name=f"{algorithm_variant}_{num_envs}x{rollout_steps_per_env}",
    )
