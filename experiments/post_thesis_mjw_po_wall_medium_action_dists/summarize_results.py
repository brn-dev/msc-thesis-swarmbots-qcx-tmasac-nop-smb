from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_action_dist_plot_common import (
    summarize_post_thesis_action_dist_results,
)
from experiments.post_thesis_mjw_po_wall_medium_action_dists.plot_results import (
    ALGORITHM_VARIANTS,
    BASELINE_SOURCES,
    EXPERIMENT_RUN_DIR,
)
from experiments.thesis_result_summary import parse_summary_args

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "final_metrics.json"


def main() -> int:
    args = parse_summary_args(default_output_path=DEFAULT_OUTPUT_PATH)
    output_path = summarize_post_thesis_action_dist_results(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        baseline_sources=BASELINE_SOURCES,
        output_path=args.output,
        tail_points=args.tail_points,
        algorithm_variants=ALGORITHM_VARIANTS,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
