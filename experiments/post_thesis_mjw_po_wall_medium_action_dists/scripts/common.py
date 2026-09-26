from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_action_dist_common import (
    PostThesisActionDistVariant,
    action_dist_group_name,
    run_post_thesis_action_dist_experiment,
)
from experiments.thesis_experiment_common import run_thesis_ppo_experiment
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

EXPERIMENT_RUN_NAME = "post_thesis_mjw_po_wall_medium_action_dists"
SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {"continuous_connector_actions": True},
)


def run_experiment(
        *,
        continuous_action_dist: PostThesisActionDistVariant,
        entrypoint_path: Path,
) -> None:
    run_post_thesis_action_dist_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        continuous_action_dist=continuous_action_dist,
        entrypoint_path=entrypoint_path,
    )


def run_mat_experiment(
        *,
        policy_variant: Literal["mat_ind", "mat_qcx"],
        continuous_action_dist: PostThesisActionDistVariant,
        entrypoint_path: Path,
) -> None:
    run_thesis_ppo_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        variant=policy_variant,
        continuous_action_dist=continuous_action_dist,
        variant_name=action_dist_group_name(
            continuous_action_dist,
            algorithm_variant=policy_variant,
        ),
        entrypoint_path=entrypoint_path,
    )
