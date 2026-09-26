from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import ContinuousActionDistVariant
from experiments.tmasac_experiment_common import (
    TMASACExperimentVariant,
    run_tmasac_experiment,
)

EXPERIMENT_RUN_NAME = "mjw_vertical_reach_tmasac"
SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": True}


def run_experiment(
        *,
        variant: TMASACExperimentVariant,
        entrypoint_path: Path,
        continuous_action_dist: ContinuousActionDistVariant = "gumbel_softmax_sign_magnitude_beta",
) -> None:
    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="vertical_reach",
        scenario_kwargs=SCENARIO_KWARGS,
        variant=variant,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
    )
