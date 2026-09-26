from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tmasac_experiment_common import EXPERIMENT_TOTAL_TIMESTEPS
from experiments.thesis_result_summary import parse_summary_args, summarize_thesis_groups
from experiments.post_thesis_mjw_po_wall_medium_action_sampling.plot_results import (
    BASELINE_SOURCES,
    DISPLAY_NAME_OVERRIDES,
    EXPERIMENT_RUN_DIR,
    GROUP_ORDER,
)
from plot_logs.experiment_results import load_experiment_groups

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "final_metrics.json"


def main() -> int:
    args = parse_summary_args(default_output_path=DEFAULT_OUTPUT_PATH)
    groups = load_experiment_groups(
        EXPERIMENT_RUN_DIR,
        group_order=GROUP_ORDER,
        group_filter=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=BASELINE_SOURCES,
    )
    output_path = summarize_thesis_groups(
        groups=groups,
        output_path=args.output,
        tail_points=args.tail_points,
        run_length_limit=EXPERIMENT_TOTAL_TIMESTEPS,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
