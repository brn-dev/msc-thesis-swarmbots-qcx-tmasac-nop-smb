from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR_CANDIDATES = (
    RUNS_DIR / "mjw_po_wall_medium_1024x4_mat_ind_ablations",
)
BASELINE_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x4_250M_no_transition_obs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
BASELINE_GROUP = "mat_ind"
GROUP_ORDER = (
    BASELINE_GROUP,
    "mat_ind_gsde_no_transition_obs",
    "mat_ind_no_nop_no_transition_obs",
)
DISPLAY_NAME_OVERRIDES = {
    BASELINE_GROUP: "MAT-Ind + NOP",
    "mat_ind_gsde_no_transition_obs": "MAT-Ind + NOP, gSDE",
    "mat_ind_no_nop_no_transition_obs": "MAT-Ind, no NOP",
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


def first_existing_dir(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_dir():
            return path

    candidate_paths = "\n".join(f"- {path}" for path in paths)
    raise NotADirectoryError(f"No experiment run dir found. Checked:\n{candidate_paths}")


EXTRA_GROUP_SOURCES = {
    BASELINE_GROUP: (BASELINE_RUN_DIR / BASELINE_GROUP,),
}


def main() -> int:
    args = parse_args()
    experiment_run_dir = first_existing_dir(EXPERIMENT_RUN_DIR_CANDIDATES)
    output_dir = OUTPUT_DIR / "final" if args.final else OUTPUT_DIR
    result = plot_experiment_results(
        experiment_run_dir,
        output_dir,
        group_order=GROUP_ORDER,
        group_filter=FINAL_GROUP_FILTER if args.final else None,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
