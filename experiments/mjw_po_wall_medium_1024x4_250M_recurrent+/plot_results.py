from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import (
    GROUP_LINE_ALPHA,
    GROUP_LINE_WIDTH,
    ExperimentPlotSelection,
    group_colors,
    load_experiment_groups,
    normalize_dpis,
    plot_experiment_results,
    plot_experiment_selection,
)


EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_recurrent+"
NON_RECURRENT_EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / "mjw_po_wall_medium_1024x4_250M_no_transition_obs"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
GROUP_ORDER = (
    "r_mat_qcc",
    "r_mat_qcx",
    "mlstm_mat_qcx",
    "slstm_mat_qcx",
    "smlstm_mat_qcx",
    "r_mat_qcs_full_causal",
    "r_mat_qcs_context_tokens_only",
    "r_mat_ind",
    "mlstm_mat_ind",
    "slstm_mat_ind",
    "smlstm_mat_ind",
)
DISPLAY_NAME_OVERRIDES = {
    "r_mat_qcc": "R-MAT-QCC + NOP",
    "r_mat_qcx": "R-MAT-QCX + NOP",
    "mlstm_mat_qcx": "R-MAT-QCX (mLSTM) + NOP",
    "slstm_mat_qcx": "R-MAT-QCX (sLSTM) + NOP",
    "smlstm_mat_qcx": "R-MAT-QCX (sLSTM/mLSTM) + NOP",
    "r_mat_qcs_full_causal": "R-MAT-QCS full causal + NOP",
    "r_mat_qcs_context_tokens_only": "R-MAT-QCS context tokens only + NOP",
    "r_mat_ind": "R-MAT-Ind + NOP",
    "mlstm_mat_ind": "R-MAT-Ind (mLSTM) + NOP",
    "slstm_mat_ind": "R-MAT-Ind (sLSTM) + NOP",
    "smlstm_mat_ind": "R-MAT-Ind (sLSTM/mLSTM) + NOP",
}
NON_RECURRENT_DISPLAY_NAME_OVERRIDES = {
    "mat_qcc": "MAT-QCC + NOP",
    "mat_qcx": "MAT-QCX + NOP",
    "mat_qcs_full_causal": "MAT-QCS full causal + NOP",
    "mat_qcs_context_tokens_only": "MAT-QCS context tokens only + NOP",
    "mat_ind": "MAT-Ind + NOP",
}
NON_RECURRENT_GROUP_BY_RECURRENT_GROUP = {
    "r_mat_qcc": "mat_qcc",
    "r_mat_qcx": "mat_qcx",
    "r_mat_qcs_full_causal": "mat_qcs_full_causal",
    "r_mat_qcs_context_tokens_only": "mat_qcs_context_tokens_only",
    "r_mat_ind": "mat_ind",
}
THEORETICAL_MAXIMUM = None
RUN_LENGTH_LIMIT = 250_000_000
FOCUSED_VARIANT_GROUPS_BY_BASELINE = {
    "mat_ind": (
        "r_mat_ind",
        "mlstm_mat_ind",
        "slstm_mat_ind",
        "smlstm_mat_ind",
    ),
    "mat_qcx": (
        "r_mat_qcx",
        "mlstm_mat_qcx",
        "slstm_mat_qcx",
        "smlstm_mat_qcx",
    ),
}


def non_recurrent_pairwise_group_name(non_recurrent_group: str) -> str:
    return f"non_recurrent_{non_recurrent_group}"


def build_pairwise_group_order() -> tuple[str, ...]:
    return (
        *GROUP_ORDER,
        *(
            non_recurrent_pairwise_group_name(non_recurrent_group)
            for non_recurrent_group in NON_RECURRENT_GROUP_BY_RECURRENT_GROUP.values()
        ),
    )


def build_pairwise_display_name_overrides() -> dict[str, str]:
    overrides = dict(DISPLAY_NAME_OVERRIDES)
    overrides.update(
        {
            non_recurrent_pairwise_group_name(non_recurrent_group): (
                f"{NON_RECURRENT_DISPLAY_NAME_OVERRIDES[non_recurrent_group]} (non-recurrent)"
            )
            for non_recurrent_group in NON_RECURRENT_GROUP_BY_RECURRENT_GROUP.values()
        }
    )
    return overrides


def build_pairwise_extra_group_sources() -> dict[str, tuple[Path, ...]]:
    return {
        non_recurrent_pairwise_group_name(non_recurrent_group): (
            NON_RECURRENT_EXPERIMENT_RUN_DIR / non_recurrent_group,
        )
        for non_recurrent_group in NON_RECURRENT_GROUP_BY_RECURRENT_GROUP.values()
    }


def build_pairwise_plot_selections() -> tuple[ExperimentPlotSelection, ...]:
    display_name_overrides = build_pairwise_display_name_overrides()
    return tuple(
        ExperimentPlotSelection(
            name=f"pair_{recurrent_group}_vs_non_recurrent",
            group_names=(
                recurrent_group,
                non_recurrent_pairwise_group_name(non_recurrent_group),
            ),
            title_suffix=(
                f"{display_name_overrides[recurrent_group]} vs "
                f"{display_name_overrides[non_recurrent_pairwise_group_name(non_recurrent_group)]}"
            ),
            required_group_names=(
                recurrent_group,
                non_recurrent_pairwise_group_name(non_recurrent_group),
            ),
            output_subdir="pairwise",
        )
        for recurrent_group, non_recurrent_group in NON_RECURRENT_GROUP_BY_RECURRENT_GROUP.items()
    )


def build_focused_variant_plot_selections() -> tuple[ExperimentPlotSelection, ...]:
    return tuple(
        ExperimentPlotSelection(
            name=f"{baseline_group}_variants_vs_non_recurrent",
            group_names=(
                *variant_groups,
                non_recurrent_pairwise_group_name(baseline_group),
            ),
            title_suffix=(
                f"{NON_RECURRENT_DISPLAY_NAME_OVERRIDES[baseline_group]} recurrent variants vs "
                f"{NON_RECURRENT_DISPLAY_NAME_OVERRIDES[baseline_group]} (non-recurrent)"
            ),
            required_group_names=(non_recurrent_pairwise_group_name(baseline_group),),
            output_subdir=baseline_group,
        )
        for baseline_group, variant_groups in FOCUSED_VARIANT_GROUPS_BY_BASELINE.items()
    )


def plot_pairwise_recurrent_vs_non_recurrent() -> list[Path]:
    groups = load_experiment_groups(
        EXPERIMENT_RUN_DIR,
        group_order=build_pairwise_group_order(),
        display_name_overrides=build_pairwise_display_name_overrides(),
        extra_group_sources=build_pairwise_extra_group_sources(),
    )
    colors = group_colors(groups)
    normalized_dpis = normalize_dpis(None)
    output_paths: list[Path] = []
    # for selection in build_pairwise_plot_selections():
    #     output_paths.extend(
    #         plot_experiment_selection(
    #             selection=selection,
    #             groups=groups,
    #             output_dir=OUTPUT_DIR,
    #             x_column="timesteps",
    #             dpis=normalized_dpis,
    #             theoretical_maximum=THEORETICAL_MAXIMUM,
    #             run_length_limit=RUN_LENGTH_LIMIT,
    #             cut_at_limit=False,
    #             group_line_width=GROUP_LINE_WIDTH,
    #             group_line_alpha=GROUP_LINE_ALPHA,
    #             colors=colors,
    #         )
    #     )
    for selection in build_focused_variant_plot_selections():
        output_paths.extend(
            plot_experiment_selection(
                selection=selection,
                groups=groups,
                output_dir=OUTPUT_DIR,
                x_column="timesteps",
                dpis=normalized_dpis,
                theoretical_maximum=THEORETICAL_MAXIMUM,
                run_length_limit=RUN_LENGTH_LIMIT,
                cut_at_limit=False,
                group_line_width=GROUP_LINE_WIDTH,
                group_line_alpha=GROUP_LINE_ALPHA,
                colors=colors,
            )
        )
    return output_paths


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        run_length_limit=RUN_LENGTH_LIMIT,
    )
    result.output_paths.extend(plot_pairwise_recurrent_vs_non_recurrent())
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
