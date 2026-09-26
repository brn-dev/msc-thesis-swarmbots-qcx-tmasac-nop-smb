from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_mjw_po_wall_disconnected_finetune.plot_hard_wall_results import (
    DISPLAY_NAME_OVERRIDES,
    EXPERIMENT_RUN_DIR,
    GROUP_ORDER,
    OUTPUT_DIR,
    RUN_LENGTH_LIMIT,
)
from experiments.thesis_result_summary import (
    parse_summary_args,
    summarize_thesis_experiment,
)

DEFAULT_OUTPUT_PATH = OUTPUT_DIR / "final_metrics.json"


def main() -> int:
    args = parse_summary_args(default_output_path=DEFAULT_OUTPUT_PATH)
    output_path = summarize_thesis_experiment(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        extra_group_sources={},
        output_path=args.output,
        tail_points=args.tail_points,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        run_length_limit=RUN_LENGTH_LIMIT,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
