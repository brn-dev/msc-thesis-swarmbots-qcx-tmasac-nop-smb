from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_no_transition_obs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "mat_qcc",
    "mat_qcs_full_causal",
    "mat_qcs_context_tokens_only",
    "mat_orig",
    "mat_ind",
    "mappo_small",
)
DISPLAY_NAME_OVERRIDES = {
    "mat_qcc": "MAT-QCC",
    "mat_qcs_full_causal": "MAT-QCS full causal",
    "mat_qcs_context_tokens_only": "MAT-QCS context tokens only",
    "mat_orig": "MAT-Orig",
    "mat_ind": "MAT-Ind",
    "mappo_small": "MAPPO",
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
        run_length_limit=250_000_000,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
