from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_plot_common import (
    PO_WALL_MEDIUM_SCENARIO_TITLE,
    plot_thesis_experiment_results,
)

RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = RUNS_DIR / "thesis_mjw_po_wall_medium"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
MATCHING_PPO_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x4_250M_cont_conn_act"
MATCHING_TMASAC_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x1_tmasac"
EXTRA_GROUP_SOURCES = {
    "mappo": (MATCHING_PPO_RUN_DIR / "mappo",),
    "mat_qcx": (MATCHING_PPO_RUN_DIR / "mat_qcx",),
    "mat_ind": (MATCHING_PPO_RUN_DIR / "mat_ind",),
    "mat_orig": (MATCHING_PPO_RUN_DIR / "mat_orig",),
    "tmasac_baseline": (MATCHING_TMASAC_RUN_DIR / "tmasac_lr=5e-5_bigger_mlps",),
}


def main() -> int:
    result = plot_thesis_experiment_results(
        experiment_run_dir=EXPERIMENT_RUN_DIR,
        output_dir=OUTPUT_DIR,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        scenario_title=PO_WALL_MEDIUM_SCENARIO_TITLE,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
