import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_multi_payload_goal_easy_tmasac.scripts.common import (
    run_experiment as _run_experiment,
)
from experiments.tmasac_experiment_common import TMASACExperimentVariant


def run_experiment(
        *,
        variant: TMASACExperimentVariant,
        entrypoint_path: Path,
) -> None:
    _run_experiment(
        variant=variant,
        entrypoint_path=entrypoint_path,
        continuous_action_dist="ternary_sign_magnitude_beta",
    )
