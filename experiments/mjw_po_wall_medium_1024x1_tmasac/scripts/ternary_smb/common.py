import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_po_wall_medium_1024x1_tmasac.scripts.common import (
    TMASACActorHeadKind,
)
from experiments.mjw_po_wall_medium_1024x1_tmasac.scripts.common import (
    run_experiment as _run_experiment,
)

__all__ = ["TMASACActorHeadKind", "run_experiment"]


def run_experiment(**kwargs: Any) -> None:
    kwargs["continuous_action_dist"] = "ternary_sign_magnitude_beta"
    _run_experiment(**kwargs)
