from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_parallel_env_ablation_po_wall_medium.plot_results import (
    DISPLAY_NAME_OVERRIDES,
    EXPERIMENT_RUN_DIR,
    EXTRA_GROUP_SOURCES,
    GROUP_ORDER,
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
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
