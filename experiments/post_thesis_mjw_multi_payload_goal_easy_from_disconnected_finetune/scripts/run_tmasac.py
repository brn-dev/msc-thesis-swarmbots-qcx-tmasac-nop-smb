from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_mjw_multi_payload_goal_easy_from_disconnected_finetune.scripts.common import (
    run_experiment,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer a disconnected PO-wall TMASAC final checkpoint to a "
            "multi-payload goal variant and train it for 100M additional transitions."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Source TMASAC final checkpoint.")
    parser.add_argument(
        "--variant",
        choices=("easy", "very_easy", "single"),
        default="easy",
        help="Target task difficulty (default: %(default)s).",
    )
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
    run_experiment(
        checkpoint_path=args.checkpoint,
        entrypoint_path=Path(__file__).resolve(),
        task_variant=args.variant,
    )
