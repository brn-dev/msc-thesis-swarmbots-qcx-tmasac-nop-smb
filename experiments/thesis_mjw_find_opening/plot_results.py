from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_plot_common import (
    FIND_OPENING_SCENARIO_TITLE,
    plot_thesis_experiment_results,
)

RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = RUNS_DIR / "thesis_mjw_find_opening"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MATCHING_TMASAC_RUN_DIR = RUNS_DIR / "mjw_find_opening_tmasac"
EXTRA_GROUP_SOURCES = {
    "tmasac_baseline": (MATCHING_TMASAC_RUN_DIR / "tmasac_baseline",),
    "slstm_two_small_actor_state_critic": (
        MATCHING_TMASAC_RUN_DIR / "slstm_two_small_actor_state_critic",
    ),
    "lstm_two_small_actor_state_critic": (
        MATCHING_TMASAC_RUN_DIR / "lstm_two_small_actor_state_critic",
    ),
}


def main() -> int:
    result = plot_thesis_experiment_results(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        output_dir=OUTPUT_DIR,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        scenario_title=FIND_OPENING_SCENARIO_TITLE,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
