from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results

RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x1_ract_tmasac"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
NON_RECURRENT_BASELINE_GROUP = "tmasac_lr=5e-5_bigger_mlps"
RECURRENT_GROUP_ORDER = tuple(
    f"{cell}_{layout}"
    for cell in ("lstm", "slstm", "smlstm")
    for layout in ("small_end", "big_end", "two_small")
)
GROUP_ORDER = (NON_RECURRENT_BASELINE_GROUP, *RECURRENT_GROUP_ORDER)
DISPLAY_NAME_OVERRIDES = {
    NON_RECURRENT_BASELINE_GROUP: "Non-recurrent TMASAC, bigger MLPs, LR 5e-5 baseline",
    **{
        group: group.replace("smlstm", "sLSTM + mLSTM")
        .replace("slstm", "sLSTM")
        .replace("lstm", "LSTM")
        .replace("_", " ")
        for group in RECURRENT_GROUP_ORDER
    },
}
EXTRA_GROUP_SOURCES = {
    NON_RECURRENT_BASELINE_GROUP: (
        RUNS_DIR
        / "mjw_po_wall_medium_1024x1_tmasac"
        / "tmasac_lr=5e-5_bigger_mlps",
    ),
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        run_length_limit=100_000_000,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
