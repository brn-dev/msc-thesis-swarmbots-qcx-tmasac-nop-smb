from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from plot_logs.experiment_results import (
    ExperimentPlotResult,
    ExperimentPlotSelection,
    plot_experiment_results,
)

SMB_NOP_PLOT_FONT_SIZE = 22
SMB_NOP_LEGEND_FONT_SIZE = 24

THESIS_MAIN_GROUP_ORDER = (
    "mappo",
    "mat_ind",
    "mat_qcx",
    "mat_orig",
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
)
THESIS_GROUP_ORDER = (
    *THESIS_MAIN_GROUP_ORDER,
    "r_mat_ind",
    "mat_qcx_gsde",
    "mat_qcx_no_nop",
    "tmasac_baseline_predicted_std",
    "tmasac_baseline_no_nop",
    "tmasac_shared_encoder",
    "slstm_shared_encoder",
    "slstm_two_small_actor_state_critic_predicted_std",
    "slstm_two_small_actor_state_critic_no_nop",
    "lstm_two_small_actor_state_critic",
    "slstm_two_small_actor_state_critic_recurrent_skip",
    "slstm_two_small_actor_state_critic_no_memory_strength",
)
THESIS_GROUP_COLOR_OVERRIDES = {
    "mappo": "#CC79A7",
    "mat_qcx": "#E69F00",
    "mat_ind": "#009E73",
    "mat_orig": "#D55E00",
    "tmasac_baseline": "#0072B2",
    "slstm_two_small_actor_state_critic": "#56B4E9",
    "r_mat_ind": "#E7298A",
    "mat_qcx_gsde": "#6A3D9A",
    "mat_qcx_no_nop": "#000000",
    "tmasac_baseline_predicted_std": "#A6761D",
    "tmasac_baseline_no_nop": "#000000",
    "tmasac_shared_encoder": "#1B9E77",
    "slstm_shared_encoder": "#1B9E77",
    "slstm_two_small_actor_state_critic_predicted_std": "#A6761D",
    "slstm_two_small_actor_state_critic_no_nop": "#000000",
    "lstm_two_small_actor_state_critic": "#8C564B",
    "slstm_two_small_actor_state_critic_recurrent_skip": "#9467BD",
    "slstm_two_small_actor_state_critic_no_memory_strength": "#666666",
}
THESIS_GROUP_LINESTYLE_OVERRIDES = dict.fromkeys(THESIS_GROUP_ORDER, "-")
THESIS_GROUP_MARKER_OVERRIDES = {
    "mappo": "o",
    "mat_ind": "s",
    "mat_qcx": "^",
    "mat_orig": "v",
    "tmasac_baseline": "P",
    "slstm_two_small_actor_state_critic": "D",
    "slstm_two_small_actor_state_critic_recurrent_skip": "X",
    "tmasac_no_connectors": "o",
}
THESIS_ABLATION_PLOTS = (
    ExperimentPlotSelection(
        name="mat_qcx_no_nop",
        group_names=("mat_qcx", "mat_qcx_no_nop"),
        required_group_names=("mat_qcx_no_nop",),
        output_subdir="no_nop/mat_qcx",
        display_name_overrides={
            "mat_qcx": "MAT-QCX + NOP",
            "mat_qcx_no_nop": "MAT-QCX, no NOP",
        },
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="tmasac_no_nop",
        group_names=("tmasac_baseline", "tmasac_baseline_no_nop"),
        required_group_names=("tmasac_baseline_no_nop",),
        output_subdir="no_nop/tmasac",
        display_name_overrides={
            "tmasac_baseline": "TMASAC + NOP",
            "tmasac_baseline_no_nop": "TMASAC, no NOP",
        },
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_no_nop",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_no_nop",
        ),
        required_group_names=("slstm_two_small_actor_state_critic_no_nop",),
        output_subdir="no_nop/slstm_tmasac",
        display_name_overrides={
            "slstm_two_small_actor_state_critic": "TMASAC + sLSTM + NOP",
            "slstm_two_small_actor_state_critic_no_nop": (
                "TMASAC + sLSTM, no NOP"
            ),
        },
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="tmasac_shared_encoder",
        group_names=("tmasac_baseline", "tmasac_shared_encoder"),
        required_group_names=("tmasac_shared_encoder",),
        output_subdir="shared_encoder/tmasac",
        display_name_overrides={
            "tmasac_baseline": "TMASAC, separate encoders",
            "tmasac_shared_encoder": "TMASAC, shared encoder",
        },
    ),
    ExperimentPlotSelection(
        name="slstm_shared_encoder",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_shared_encoder",
        ),
        required_group_names=("slstm_shared_encoder",),
        output_subdir="shared_encoder/slstm_tmasac",
        display_name_overrides={
            "slstm_two_small_actor_state_critic": "TMASAC + sLSTM, separate encoders",
            "slstm_shared_encoder": "TMASAC + shared sLSTM encoder",
        },
    ),
    ExperimentPlotSelection(
        name="mat_ind_recurrence",
        group_names=("mat_ind", "r_mat_ind"),
        required_group_names=("r_mat_ind",),
        output_subdir="recurrence/mat_ind",
        display_name_overrides={
            "mat_ind": "MAT-IND",
            "r_mat_ind": "R-MAT-IND",
        },
    ),
    ExperimentPlotSelection(
        name="mat_qcx_gsde",
        group_names=("mat_qcx", "mat_qcx_gsde"),
        required_group_names=("mat_qcx_gsde",),
        output_subdir="gaussian_action_distributions/mat_qcx",
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="tmasac_predicted_std",
        group_names=("tmasac_baseline", "tmasac_baseline_predicted_std"),
        required_group_names=("tmasac_baseline_predicted_std",),
        output_subdir="gaussian_action_distributions/tmasac",
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_predicted_std",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_predicted_std",
        ),
        required_group_names=(
            "slstm_two_small_actor_state_critic_predicted_std",
        ),
        output_subdir="gaussian_action_distributions/slstm_tmasac",
        font_size=SMB_NOP_PLOT_FONT_SIZE,
        legend_font_size=SMB_NOP_LEGEND_FONT_SIZE,
    ),
    ExperimentPlotSelection(
        name="tmasac_temporal_model",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "lstm_two_small_actor_state_critic",
        ),
        required_group_names=("lstm_two_small_actor_state_critic",),
        output_subdir="temporal_model/tmasac",
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_recurrent_skip",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_recurrent_skip",
        ),
        required_group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_recurrent_skip",
        ),
        output_subdir="recurrent_skip/slstm_tmasac",
        display_name_overrides={
            "slstm_two_small_actor_state_critic": "TMASAC + sLSTM, no recurrent skip",
            "slstm_two_small_actor_state_critic_recurrent_skip": (
                "TMASAC + sLSTM, recurrent skip"
            ),
        },
    ),
    ExperimentPlotSelection(
        name="slstm_tmasac_memory_strength",
        group_names=(
            "slstm_two_small_actor_state_critic",
            "slstm_two_small_actor_state_critic_no_memory_strength",
        ),
        required_group_names=(
            "slstm_two_small_actor_state_critic_no_memory_strength",
        ),
        output_subdir="memory_strength/slstm_tmasac",
    ),
)
THESIS_DISPLAY_NAMES = {
    "mappo": "MAPPO",
    "mat_qcx": "MAT-QCX",
    "mat_ind": "MAT-Independent",
    "r_mat_ind": "R-MAT-IND",
    "mat_orig": "MAT original decoder",
    "tmasac_baseline": "TMASAC",
    "slstm_two_small_actor_state_critic": "TMASAC + sLSTM",
    "mat_qcx_gsde": "MAT-QCX, gSDE",
    "mat_qcx_no_nop": "MAT-QCX, no NOP",
    "tmasac_baseline_predicted_std": "TMASAC, squashed Gaussian",
    "tmasac_baseline_no_nop": "TMASAC, no NOP",
    "tmasac_shared_encoder": "TMASAC, shared encoder",
    "slstm_shared_encoder": "TMASAC + shared sLSTM encoder",
    "slstm_two_small_actor_state_critic_predicted_std": (
        "TMASAC + sLSTM, squashed Gaussian"
    ),
    "slstm_two_small_actor_state_critic_no_nop": "TMASAC + sLSTM, no NOP",
    "lstm_two_small_actor_state_critic": "TMASAC + LSTM",
    "slstm_two_small_actor_state_critic_recurrent_skip": (
        "TMASAC + sLSTM, recurrent skip"
    ),
    "slstm_two_small_actor_state_critic_no_memory_strength": (
        "TMASAC + sLSTM, no critic memory strength"
    ),
}
THESIS_RUN_LENGTH = 100_000_000
PO_WALL_MEDIUM_SCENARIO_TITLE = "PO-Wall (Medium)"
FIND_OPENING_SCENARIO_TITLE = "Find-Opening"


def plot_thesis_experiment_results(
    *,
    experiment_run_dir: Path,
    output_dir: Path,
    extra_group_sources: Mapping[str, Sequence[Path]],
    scenario_title: str,
) -> ExperimentPlotResult:
    return plot_experiment_results(
        experiment_run_dir,
        output_dir,
        group_order=THESIS_GROUP_ORDER,
        display_name_overrides=THESIS_DISPLAY_NAMES,
        extra_group_sources=extra_group_sources,
        main_group_names=THESIS_MAIN_GROUP_ORDER,
        extra_plot_selections=THESIS_ABLATION_PLOTS,
        group_color_overrides=THESIS_GROUP_COLOR_OVERRIDES,
        group_linestyle_overrides=THESIS_GROUP_LINESTYLE_OVERRIDES,
        group_marker_overrides=THESIS_GROUP_MARKER_OVERRIDES,
        run_length_limit=THESIS_RUN_LENGTH,
        cut_at_limit=True,
        title_suffix=scenario_title,
        include_selection_title_suffix=False,
    )
