from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mat_qcs_nop_swarm_bots_wall_batch4096_env_sweep"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "32x128",
    "128x32",
    "512x8",
    "1024x4",
    "2048x2",
)
THEORETICAL_MAXIMUM = 10.5


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
