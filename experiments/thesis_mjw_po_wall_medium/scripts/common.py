from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_experiment_common import (
    ThesisAlgorithmVariant,
    run_thesis_experiment,
)
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_medium"
SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {"continuous_connector_actions": True},
)


def run_experiment(*, variant: ThesisAlgorithmVariant, entrypoint_path: Path) -> None:
    run_thesis_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        variant=variant,
        entrypoint_path=entrypoint_path,
    )
