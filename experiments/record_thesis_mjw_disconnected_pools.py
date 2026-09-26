from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.record_thesis_mjw_unseen_morphologies import main as _main

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "thesis_mjw_disconnected_pools" / "recordings"


def main(argv: Sequence[str] | None = None) -> int:
    return _main(
        argv,
        unconnected_prob=1.0,
        default_output_root=DEFAULT_OUTPUT_ROOT,
        morphology_description=(
            "unseen, fully disconnected pools generated from pre-connected layouts"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
