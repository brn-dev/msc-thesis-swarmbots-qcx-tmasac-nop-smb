from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_mjw_po_wall_medium.scripts.common_250m import (
    EXPERIMENT_RUN_NAME,
    TOTAL_TIMESTEPS,
)
from experiments.thesis_plot_common import (
    PO_WALL_MEDIUM_SCENARIO_TITLE,
    THESIS_DISPLAY_NAMES,
    THESIS_GROUP_COLOR_OVERRIDES,
    THESIS_GROUP_LINESTYLE_OVERRIDES,
    THESIS_GROUP_MARKER_OVERRIDES,
)
from plot_logs.experiment_results import plot_experiment_results

GROUP_ORDER = ("mappo", "mat_ind", "mat_qcx", "mat_orig")
EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / EXPERIMENT_RUN_NAME
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "250m"


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=THESIS_DISPLAY_NAMES,
        group_color_overrides=THESIS_GROUP_COLOR_OVERRIDES,
        group_linestyle_overrides=THESIS_GROUP_LINESTYLE_OVERRIDES,
        group_marker_overrides=THESIS_GROUP_MARKER_OVERRIDES,
        run_length_limit=TOTAL_TIMESTEPS,
        cut_at_limit=True,
        title_suffix=PO_WALL_MEDIUM_SCENARIO_TITLE,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
