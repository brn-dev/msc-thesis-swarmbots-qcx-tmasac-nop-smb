from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import ExperimentPlotSelection, plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_cont_conn_act"
BASELINE_EXPERIMENT_RUN_DIR = RUNS_DIR / "mjw_po_wall_medium_1024x4_250M_no_transition_obs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
VARIANT_GROUPS = (
    "mat_qcx",
    "mat_ind",
    "mat_qcc",
    "mat_qcs_full_causal",
    "mat_qcs_context_tokens_only",
    "mat_orig",
)
BASE_DISPLAY_NAME_OVERRIDES = {
    "mat_qcx": "MAT-QCX + NOP",
    "mat_ind": "MAT-Ind + NOP",
    "mat_qcc": "MAT-QCC + NOP",
    "mat_qcs_full_causal": "MAT-QCS full causal + NOP",
    "mat_qcs_context_tokens_only": (
        "MAT-QCS context tokens only + NOP"
    ),
    "mat_orig": "MAT-Orig + NOP",
}
THEORETICAL_MAXIMUM = None


def baseline_group_name(group_name: str) -> str:
    return f"{group_name}_non_cont_conn_act"


GROUP_ORDER = tuple(
    group_name
    for variant_group in VARIANT_GROUPS
    for group_name in (baseline_group_name(variant_group), variant_group)
)
DISPLAY_NAME_OVERRIDES = {
    baseline_group_name(group_name): (
        f"{display_name} (discrete connectors)"
    )
    for group_name, display_name in BASE_DISPLAY_NAME_OVERRIDES.items()
} | {
    group_name: f"{display_name} (continuous connectors)"
    for group_name, display_name in BASE_DISPLAY_NAME_OVERRIDES.items()
}
EXTRA_GROUP_SOURCES = {
    baseline_group_name(group_name): (
        BASELINE_EXPERIMENT_RUN_DIR / group_name,
    )
    for group_name in ["mat_qcx", "mat_ind"]
}
EXTRA_PLOT_SELECTIONS = tuple(
    ExperimentPlotSelection(
        name=f"pair_{baseline_group_name(group_name)}_vs_{group_name}",
        group_names=(baseline_group_name(group_name), group_name),
        title_suffix=(
            f"{BASE_DISPLAY_NAME_OVERRIDES[group_name]}: "
            "Non-Continuous vs Continuous Connector Actions"
        ),
        required_group_names=(baseline_group_name(group_name),),
        output_subdir="pairwise",
    )
    for group_name in VARIANT_GROUPS
)


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        extra_plot_selections=EXTRA_PLOT_SELECTIONS,
        run_length_limit=250_000_000,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
