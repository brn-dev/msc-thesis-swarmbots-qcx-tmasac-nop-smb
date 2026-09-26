from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune.scripts.common import (
    continue_single_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue a completed 100M single-payload TMASAC run for another "
            "200M transitions."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Single-payload final checkpoint.")
    parser.add_argument(
        "--cuda-idx",
        "--cuda_idx",
        "--gpu",
        type=int,
        help="Visible CUDA device index used by the shared experiment runner.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    continue_single_experiment(
        checkpoint_path=args.checkpoint,
        entrypoint_path=Path(__file__).resolve(),
    )
