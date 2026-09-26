from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_thesis_mjw_unseen_morphologies import main as _main

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "experiments"
    / "thesis_mjw_disconnected_pools_no_connections"
    / "results"
    / "evaluation.json"
)


def main(argv: Sequence[str] | None = None) -> int:
    forwarded_argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == "--target" or arg.startswith("--target=") for arg in forwarded_argv):
        forwarded_argv.extend(("--target", "po_wall_tmasac"))
    if "--disable-connector-actions" not in forwarded_argv:
        forwarded_argv.append("--disable-connector-actions")
    return _main(
        forwarded_argv,
        unconnected_prob=1.0,
        default_output_path=DEFAULT_OUTPUT_PATH,
        morphology_description=(
            "unseen, fully disconnected pools with connector actions disabled"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
