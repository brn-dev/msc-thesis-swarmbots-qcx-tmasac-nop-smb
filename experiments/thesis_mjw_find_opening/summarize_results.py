from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_mjw_find_opening.plot_results import (
    EXPERIMENT_RUN_DIR,
    EXTRA_GROUP_SOURCES,
)
from experiments.thesis_result_summary import (
    parse_summary_args,
    summarize_thesis_experiment,
)

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "final_metrics.json"


def main() -> int:
    args = parse_summary_args(default_output_path=DEFAULT_OUTPUT_PATH)
    output_path = summarize_thesis_experiment(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        output_path=args.output,
        tail_points=args.tail_points,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
