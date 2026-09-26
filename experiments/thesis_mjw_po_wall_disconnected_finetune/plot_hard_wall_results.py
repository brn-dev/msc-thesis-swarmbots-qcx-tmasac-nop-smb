from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_plot_common import (
    THESIS_GROUP_LINESTYLE_OVERRIDES,
    THESIS_GROUP_MARKER_OVERRIDES,
)
from plot_logs.experiment_results import plot_experiment_results

EXPERIMENT_RUN_DIR = (
    REPO_ROOT
    / "runs"
    / "thesis_mjw_po_wall_disconnected_finetune_hard_wall_50m"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "hard_wall_results"
GROUP_ORDER = (
    "tmasac_baseline",
    "tmasac_no_connectors",
)
DISPLAY_NAME_OVERRIDES = {
    "tmasac_baseline": "TMASAC",
    "tmasac_no_connectors": "TMASAC, no connectors",
}
RUN_LENGTH_LIMIT = 150_000_000
TITLE_SUFFIX = "PO-Wall (height 0.4), disconnected-start fine-tuning"


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        group_linestyle_overrides=THESIS_GROUP_LINESTYLE_OVERRIDES,
        group_marker_overrides=THESIS_GROUP_MARKER_OVERRIDES,
        run_length_limit=RUN_LENGTH_LIMIT,
        cut_at_limit=True,
        title_suffix=TITLE_SUFFIX,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
