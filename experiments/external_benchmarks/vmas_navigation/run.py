from __future__ import annotations

import importlib
import sys
from pathlib import Path

ENTRYPOINT_PATH = Path(__file__).resolve()
REPO_ROOT = ENTRYPOINT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

scenario_experiment_common = importlib.import_module(
    "experiments.external_benchmarks.scenario_experiment_common"
)


if __name__ == "__main__":
    scenario_experiment_common.run_registered_experiment(
        experiment_name=ENTRYPOINT_PATH.parent.name,
        entrypoint_path=ENTRYPOINT_PATH,
    )
