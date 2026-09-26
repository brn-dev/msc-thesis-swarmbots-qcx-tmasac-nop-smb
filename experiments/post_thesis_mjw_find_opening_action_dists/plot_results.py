from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_action_dist_common import (
    BASELINE_GROUP,
    SLSTM_TMASAC_GROUP,
)
from experiments.post_thesis_action_dist_plot_common import (
    make_thesis_baseline_sources,
    plot_post_thesis_action_dist_results,
)
from experiments.thesis_mjw_find_opening.plot_results import (
    EXPERIMENT_RUN_DIR as THESIS_EXPERIMENT_RUN_DIR,
    EXTRA_GROUP_SOURCES as THESIS_EXTRA_GROUP_SOURCES,
)
from experiments.thesis_plot_common import FIND_OPENING_SCENARIO_TITLE

EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "post_thesis_mjw_find_opening_action_dists"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
ALGORITHM_VARIANTS = (BASELINE_GROUP, SLSTM_TMASAC_GROUP)
BASELINE_SOURCES = {
    algorithm_variant: make_thesis_baseline_sources(
        thesis_experiment_run_dir=THESIS_EXPERIMENT_RUN_DIR,
        thesis_extra_group_sources=THESIS_EXTRA_GROUP_SOURCES,
        algorithm_variant=algorithm_variant,
    )
    for algorithm_variant in ALGORITHM_VARIANTS
}


def main() -> int:
    result = plot_post_thesis_action_dist_results(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        output_dir=OUTPUT_DIR,
        baseline_sources=BASELINE_SOURCES,
        scenario_title=FIND_OPENING_SCENARIO_TITLE,
        algorithm_variants=ALGORITHM_VARIANTS,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
