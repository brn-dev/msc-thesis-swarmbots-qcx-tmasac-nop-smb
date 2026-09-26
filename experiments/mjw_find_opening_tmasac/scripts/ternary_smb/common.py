import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_find_opening_tmasac.scripts.common import (
    ACTOR_D_MODEL,
    PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM,
    ActorStateCriticInputConfig,
    TMASACActorHeadKind,
)
from experiments.mjw_find_opening_tmasac.scripts.common import (
    run_experiment as _run_experiment,
)

__all__ = [
    "ACTOR_D_MODEL",
    "PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM",
    "ActorStateCriticInputConfig",
    "TMASACActorHeadKind",
    "run_experiment",
]


def run_experiment(**kwargs: Any) -> None:
    kwargs["continuous_action_dist"] = "ternary_sign_magnitude_beta"
    _run_experiment(**kwargs)
