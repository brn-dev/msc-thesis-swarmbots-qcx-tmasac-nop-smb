from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_50_50_conn"
BASELINE_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_no_transition_obs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
RUN_LENGTH_LIMIT = 100_000_000
GROUP_ORDER = (
    "mat_ind_baseline",
    "mat_qcx_baseline",
    "mat_ind",
    "mat_qcx",
)
DISPLAY_NAME_OVERRIDES = {
    "mat_ind_baseline": "MAT-Ind 80/20 baseline",
    "mat_qcx_baseline": "MAT-QCX 80/20 baseline",
    "mat_ind": "MAT-Ind 50/50 conn",
    "mat_qcx": "MAT-QCX 50/50 conn",
}
FINAL_GROUP_FILTER = tuple(dict.fromkeys((*GROUP_ORDER, *DISPLAY_NAME_OVERRIDES)))
THEORETICAL_MAXIMUM = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final",
        action="store_true",
        help="Only plot named final variants and write outputs under results/final.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = OUTPUT_DIR / "final" if args.final else OUTPUT_DIR
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        output_dir,
        group_order=GROUP_ORDER,
        group_filter=FINAL_GROUP_FILTER if args.final else None,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources={
            "mat_ind_baseline": (BASELINE_RUN_DIR / "mat_ind",),
            "mat_qcx_baseline": (BASELINE_RUN_DIR / "mat_qcx",),
        },
        run_length_limit=RUN_LENGTH_LIMIT,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
