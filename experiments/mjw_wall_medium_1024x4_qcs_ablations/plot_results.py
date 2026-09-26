from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR_CANDIDATES = (
    RUNS_DIR / "mjw_wall_medium_1024x4_qcs_ablations",
)
BASELINE_RUN_DIR = RUNS_DIR / "mjw_wall_medium_1024x4"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
BASELINE_GROUP = "mat_qcs_context_tokens_only"
GROUP_ORDER = (
    BASELINE_GROUP,
    "mat_qcs_context_tokens_only_gsde",
    "mat_qcs_context_tokens_only_no_nop",
    "mat_qcs_context_tokens_only_no_transition_obs",
)
DISPLAY_NAME_OVERRIDES = {
    BASELINE_GROUP: "MAT-QCS context tokens only + NOP",
    "mat_qcs_context_tokens_only_gsde": "MAT-QCS context tokens only + NOP, gSDE",
    "mat_qcs_context_tokens_only_no_nop": "MAT-QCS context tokens only, no NOP",
    "mat_qcs_context_tokens_only_no_transition_obs": (
        "MAT-QCS context tokens only + NOP, no transition obs"
    ),
}
THEORETICAL_MAXIMUM = None


def first_existing_dir(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_dir():
            return path

    candidate_paths = "\n".join(f"- {path}" for path in paths)
    raise NotADirectoryError(f"No experiment run dir found. Checked:\n{candidate_paths}")


EXPERIMENT_RUN_DIR = first_existing_dir(EXPERIMENT_RUN_DIR_CANDIDATES)
EXTRA_GROUP_SOURCES = {
    BASELINE_GROUP: (BASELINE_RUN_DIR / BASELINE_GROUP,),
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
