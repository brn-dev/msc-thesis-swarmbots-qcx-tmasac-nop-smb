from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_multi_payload_goal_easy_tmasac.scripts.common import (
    SCENARIO_KWARGS as EASY_MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS,
)
from experiments.tmasac_experiment_common import run_tmasac_experiment

EXPERIMENT_RUN_NAME = (
    "post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune"
)
SINGLE_CONTINUATION_EXPERIMENT_RUN_NAME = (
    "post_thesis_mjw_multi_payload_goal_single_continue_200m"
)
ADDITIONAL_TIMESTEPS = 100_000_000
SINGLE_CONTINUATION_ADDITIONAL_TIMESTEPS = 200_000_000
EVALUATION_MILESTONES = (25.0, 50.0, 75.0, 100.0)
TransferTaskVariant = Literal["easy", "very_easy", "single"]
SCENARIO_KWARGS_BY_VARIANT: dict[TransferTaskVariant, dict[str, object]] = {
    "easy": dict(EASY_MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS),
    "very_easy": {
        **EASY_MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS,
        "active_payload_count_probs": {1: 0.9, 2: 0.1},
    },
    "single": {
        **EASY_MULTI_PAYLOAD_GOAL_SCENARIO_KWARGS,
        "max_payloads": 1,
        "active_payload_count_probs": {1: 1.0},
    },
}
RUN_VARIANT_NAMES: dict[TransferTaskVariant, str] = {
    "easy": "tmasac_from_disconnected_finetune",
    "very_easy": "tmasac_from_disconnected_finetune_very_easy",
    "single": "tmasac_from_disconnected_finetune_single",
}


def run_experiment(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
    task_variant: TransferTaskVariant = "easy",
) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)

    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="multi_payload_goal",
        scenario_kwargs=SCENARIO_KWARGS_BY_VARIANT[task_variant],
        evaluation_milestones=EVALUATION_MILESTONES,
        variant="tmasac_baseline",
        variant_name=RUN_VARIANT_NAMES[task_variant],
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=ADDITIONAL_TIMESTEPS,
        transfer_checkpoint=True,
    )


def continue_single_experiment(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)

    run_tmasac_experiment(
        experiment_run_name=SINGLE_CONTINUATION_EXPERIMENT_RUN_NAME,
        scenario_name="multi_payload_goal",
        scenario_kwargs=SCENARIO_KWARGS_BY_VARIANT["single"],
        evaluation_milestones=EVALUATION_MILESTONES,
        variant="tmasac_baseline",
        variant_name="tmasac_from_disconnected_finetune_single",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=SINGLE_CONTINUATION_ADDITIONAL_TIMESTEPS,
        transfer_checkpoint=False,
    )


def _resolve_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path
