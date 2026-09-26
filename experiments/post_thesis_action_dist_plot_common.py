from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.post_thesis_action_dist_common import (
    ACTION_DIST_VARIANTS,
    BASELINE_GROUP,
    PostThesisAlgorithmVariant,
    SLSTM_TMASAC_GROUP,
    action_dist_group_name,
)
from experiments.thesis_plot_common import (
    THESIS_GROUP_COLOR_OVERRIDES,
    THESIS_GROUP_MARKER_OVERRIDES,
    THESIS_RUN_LENGTH,
)
from experiments.thesis_result_summary import summarize_thesis_groups
from plot_logs.experiment_results import (
    ExperimentPlotResult,
    load_experiment_groups,
    plot_experiment_results,
)

DEFAULT_ALGORITHM_VARIANTS: tuple[PostThesisAlgorithmVariant, ...] = (BASELINE_GROUP,)
_ALGORITHM_DISPLAY_NAMES = {
    BASELINE_GROUP: "TMASAC",
    SLSTM_TMASAC_GROUP: "TMASAC + sLSTM",
    "mat_ind": "MAT-IND",
    "mat_qcx": "MAT-QCX",
}
_ALGORITHM_LINESTYLES = {
    BASELINE_GROUP: "-",
    SLSTM_TMASAC_GROUP: "--",
    "mat_ind": "-.",
    "mat_qcx": ":",
}
_ACTION_DIST_DISPLAY_NAMES = {
    "bernstein_6": "Bernstein (degree 6)",
    "bernstein_8": "Bernstein (degree 8)",
    "bernstein_12": "Bernstein (degree 12)",
    "bernstein_18": "Bernstein (degree 18)",
    "rqs_4": "RQS (4 bins)",
    "rqs_6": "RQS (6 bins)",
}
_ACTION_DIST_COLORS = {
    "bernstein_6": "#E69F00",
    "bernstein_8": "#D55E00",
    "bernstein_12": "#56B4E9",
    "bernstein_18": "#0072B2",
    "rqs_4": "#009E73",
    "rqs_6": "#CC79A7",
}
_ACTION_DIST_MARKERS = {
    "bernstein_6": "o",
    "bernstein_8": "s",
    "bernstein_12": "v",
    "bernstein_18": "P",
    "rqs_4": "^",
    "rqs_6": "D",
}


@dataclass(frozen=True)
class ActionDistGroupSpec:
    name: str
    display_name: str
    color: str
    linestyle: str
    marker: str


def make_group_specs(
        algorithm_variants: Sequence[PostThesisAlgorithmVariant],
) -> tuple[ActionDistGroupSpec, ...]:
    specs: list[ActionDistGroupSpec] = []
    for algorithm_variant in algorithm_variants:
        architecture_name = _ALGORITHM_DISPLAY_NAMES[algorithm_variant]
        linestyle = _ALGORITHM_LINESTYLES[algorithm_variant]
        baseline_distribution = (
            "sign-magnitude Beta"
            if algorithm_variant in ("mat_ind", "mat_qcx")
            else "Gumbel sign-magnitude Beta"
        )
        specs.append(ActionDistGroupSpec(
            name=algorithm_variant,
            display_name=f"{architecture_name}, {baseline_distribution}",
            color=THESIS_GROUP_COLOR_OVERRIDES[algorithm_variant],
            linestyle=linestyle,
            marker=THESIS_GROUP_MARKER_OVERRIDES[algorithm_variant],
        ))
        specs.extend(
            ActionDistGroupSpec(
                name=action_dist_group_name(
                    action_dist,
                    algorithm_variant=algorithm_variant,
                ),
                display_name=(
                    f"{architecture_name}, {_ACTION_DIST_DISPLAY_NAMES[action_dist]}"
                ),
                color=_ACTION_DIST_COLORS[action_dist],
                linestyle=linestyle,
                marker=_ACTION_DIST_MARKERS[action_dist],
            )
            for action_dist in ACTION_DIST_VARIANTS
        )
    return tuple(specs)


def make_group_order(
        algorithm_variants: Sequence[PostThesisAlgorithmVariant],
) -> tuple[str, ...]:
    return tuple(spec.name for spec in make_group_specs(algorithm_variants))


_DEFAULT_GROUP_SPECS = make_group_specs(DEFAULT_ALGORITHM_VARIANTS)
GROUP_ORDER = tuple(spec.name for spec in _DEFAULT_GROUP_SPECS)
DISPLAY_NAME_OVERRIDES = {
    spec.name: spec.display_name for spec in _DEFAULT_GROUP_SPECS
}
GROUP_COLOR_OVERRIDES = {spec.name: spec.color for spec in _DEFAULT_GROUP_SPECS}
GROUP_LINESTYLE_OVERRIDES = {
    spec.name: spec.linestyle for spec in _DEFAULT_GROUP_SPECS
}
GROUP_MARKER_OVERRIDES = {spec.name: spec.marker for spec in _DEFAULT_GROUP_SPECS}


def make_thesis_baseline_sources(
        *,
        thesis_experiment_run_dir: Path,
        thesis_extra_group_sources: Mapping[str, Sequence[Path]],
        algorithm_variant: PostThesisAlgorithmVariant = BASELINE_GROUP,
) -> tuple[Path, ...]:
    return (
        thesis_experiment_run_dir / algorithm_variant,
        *thesis_extra_group_sources[algorithm_variant],
    )


def plot_post_thesis_action_dist_results(
        *,
        experiment_run_dir: Path,
        output_dir: Path,
        baseline_sources: Mapping[PostThesisAlgorithmVariant, Sequence[Path]],
        scenario_title: str,
        algorithm_variants: Sequence[PostThesisAlgorithmVariant] = DEFAULT_ALGORITHM_VARIANTS,
) -> ExperimentPlotResult:
    group_specs = make_group_specs(algorithm_variants)
    group_order = tuple(spec.name for spec in group_specs)
    return plot_experiment_results(
        experiment_run_dir,
        output_dir,
        group_order=group_order,
        group_filter=group_order,
        display_name_overrides={spec.name: spec.display_name for spec in group_specs},
        extra_group_sources=baseline_sources,
        group_color_overrides={spec.name: spec.color for spec in group_specs},
        group_linestyle_overrides={spec.name: spec.linestyle for spec in group_specs},
        group_marker_overrides={spec.name: spec.marker for spec in group_specs},
        run_length_limit=THESIS_RUN_LENGTH,
        cut_at_limit=True,
        title_suffix=scenario_title,
    )


def summarize_post_thesis_action_dist_results(
        *,
        experiment_run_dir: Path,
        baseline_sources: Mapping[PostThesisAlgorithmVariant, Sequence[Path]],
        output_path: Path,
        tail_points: int,
        algorithm_variants: Sequence[PostThesisAlgorithmVariant] = DEFAULT_ALGORITHM_VARIANTS,
) -> Path:
    group_specs = make_group_specs(algorithm_variants)
    group_order = tuple(spec.name for spec in group_specs)
    groups = load_experiment_groups(
        experiment_run_dir,
        group_order=group_order,
        group_filter=group_order,
        display_name_overrides={spec.name: spec.display_name for spec in group_specs},
        extra_group_sources=baseline_sources,
    )
    return summarize_thesis_groups(
        groups=groups,
        output_path=output_path,
        tail_points=tail_points,
    )
