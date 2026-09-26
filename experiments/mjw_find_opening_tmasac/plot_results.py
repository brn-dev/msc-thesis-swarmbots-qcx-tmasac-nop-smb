from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results

EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_find_opening_tmasac"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "tmasac_baseline",
    "lstm_two_small",
    "slstm_two_small",
)
DISPLAY_NAME_OVERRIDES = {
    "tmasac_baseline": "Non-recurrent TMASAC baseline",
    "lstm_two_small": "LSTM TMASAC, two small MLPs",
    "slstm_two_small": "sLSTM TMASAC, two small MLPs",
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        run_length_limit=100_000_000,
        cut_at_limit=True,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
