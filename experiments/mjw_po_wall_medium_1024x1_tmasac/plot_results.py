from __future__ import annotations

import itertools
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x1_tmasac"
BASELINE_EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x4_250M_cont_conn_act"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000
MAT_IND_BASELINE_GROUP = "mat_ind_baseline"
MAT_QCX_BASELINE_GROUP = "mat_qcx_baseline"
GROUP_ORDER = (
    MAT_IND_BASELINE_GROUP,
    MAT_QCX_BASELINE_GROUP,
    "tmasac_lr=3e-4",
    "tmasac_lr=1e-4",
    "tmasac_lr=1e-4_bigger_mlps",
    "tmasac_lr=5e-5",
)
DISPLAY_NAME_OVERRIDES = {
    MAT_IND_BASELINE_GROUP: "MAT-Ind + NOP baseline",
    MAT_QCX_BASELINE_GROUP: "MAT-QCX + NOP baseline",
    "tmasac_lr=3e-4": "TMASAC Gumbel SMB + NOP, LR 3e-4",
    "tmasac_lr=1e-4": "TMASAC Gumbel SMB + NOP, LR 1e-4",
    "tmasac_lr=1e-4_bigger_mlps": "TMASAC Gumbel SMB + NOP, LR 1e-4, bigger MLPs",
    "tmasac_lr=5e-5": "TMASAC Gumbel SMB + NOP, LR 5e-5",
}
EXTRA_GROUP_SOURCES = {
    MAT_IND_BASELINE_GROUP: (BASELINE_EXPERIMENT_RUN_DIR / "mat_ind",),
    MAT_QCX_BASELINE_GROUP: (BASELINE_EXPERIMENT_RUN_DIR / "mat_qcx",),
}
THEORETICAL_MAXIMUM = None


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        run_length_limit=EXPERIMENT_TOTAL_TIMESTEPS,
        cut_at_limit=True,
    )

    result_250 = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        Path(__file__).resolve().parent / "results/250M",
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        run_length_limit=250_000_000,
        cut_at_limit=True,
    )

    for output_path in itertools.chain(result.output_paths, result_250.output_paths):
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
