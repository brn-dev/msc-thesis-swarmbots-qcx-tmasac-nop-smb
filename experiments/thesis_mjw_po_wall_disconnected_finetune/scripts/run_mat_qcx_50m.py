from __future__ import annotations

import argparse
from pathlib import Path

from common import run_mat_qcx_50m


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a PO-wall MAT-QCX checkpoint for 50M additional transitions "
            "from compact, aligned, fully disconnected starts."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="Source MAT-QCX .pt checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_mat_qcx_50m(
        checkpoint_path=args.checkpoint,
        entrypoint_path=Path(__file__).resolve(),
    )
