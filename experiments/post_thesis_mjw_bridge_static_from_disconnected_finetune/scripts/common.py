from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_bridge_static_tmasac.scripts.common import (
    SCENARIO_KWARGS as BRIDGE_SCENARIO_KWARGS,
)
from experiments.tmasac_experiment_common import run_tmasac_experiment

EXPERIMENT_RUN_NAME = "post_thesis_mjw_bridge_static_from_disconnected_finetune"
ADDITIONAL_TIMESTEPS = 100_000_000
EVALUATION_MILESTONES = (25.0, 50.0, 75.0, 100.0)
SCENARIO_KWARGS = dict(BRIDGE_SCENARIO_KWARGS)


def run_experiment(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="bridge",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        variant="tmasac_baseline",
        variant_name="tmasac_from_disconnected_finetune",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=ADDITIONAL_TIMESTEPS,
        transfer_checkpoint=True,
    )
