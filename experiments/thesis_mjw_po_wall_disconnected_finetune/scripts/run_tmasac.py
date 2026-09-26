from __future__ import annotations

import argparse
from pathlib import Path

from common import run_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a PO-wall TMASAC checkpoint for 20M additional transitions "
            "from compact, aligned, fully disconnected starts."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Source TMASAC .pt checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_experiment(
        checkpoint_path=args.checkpoint,
        entrypoint_path=Path(__file__).resolve(),
    )
