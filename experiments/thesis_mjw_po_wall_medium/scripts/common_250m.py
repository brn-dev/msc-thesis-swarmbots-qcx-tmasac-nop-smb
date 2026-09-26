from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_experiment_common import run_thesis_ppo_experiment
from experiments.thesis_mjw_po_wall_medium.scripts.common import SCENARIO_KWARGS

LongHorizonAlgorithmVariant = Literal["mappo", "mat_qcx", "mat_ind", "mat_orig"]

EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_medium_250m"
TOTAL_TIMESTEPS = 250_000_000


def run_experiment(
    *,
    variant: LongHorizonAlgorithmVariant,
    entrypoint_path: Path,
) -> None:
    run_thesis_ppo_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        variant=variant,
        entrypoint_path=entrypoint_path,
        total_timesteps=TOTAL_TIMESTEPS,
    )
