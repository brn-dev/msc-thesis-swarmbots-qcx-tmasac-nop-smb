from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.thesis_plot_common import (
    THESIS_DISPLAY_NAMES,
    THESIS_GROUP_ORDER,
    THESIS_RUN_LENGTH,
)
from plot_logs.experiment_results import ExperimentGroup, load_experiment_groups
from plot_logs.experiment_summary import (
    DEFAULT_TAIL_POINTS,
    summarize_experiment_groups,
    write_experiment_summary,
)


def parse_summary_args(*, default_output_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize each variant's final reward and success EMAs across runs, "
            "including the distribution of per-run timesteps at which the success "
            "EMA first reaches each configured threshold. Each run contributes the "
            "mean of its final finite metric values."
        )
    )
    parser.add_argument(
        "--tail-points",
        type=int,
        default=DEFAULT_TAIL_POINTS,
        help=(
            "Number of final finite values to average per run before aggregating "
            f"runs. Defaults to {DEFAULT_TAIL_POINTS}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path,
        help=f"Output JSON path. Defaults to {default_output_path}.",
    )
    return parser.parse_args()


def summarize_thesis_experiment(
    *,
    experiment_run_dir: Path,
    extra_group_sources: Mapping[str, Sequence[Path]],
    output_path: Path,
    tail_points: int,
    group_order: Sequence[str] = THESIS_GROUP_ORDER,
    display_name_overrides: Mapping[str, str] = THESIS_DISPLAY_NAMES,
    run_length_limit: int = THESIS_RUN_LENGTH,
    run_length_limit_overrides: Mapping[str, int] | None = None,
) -> Path:
    groups = load_experiment_groups(
        experiment_run_dir,
        group_order=group_order,
        display_name_overrides=display_name_overrides,
        extra_group_sources=extra_group_sources,
    )
    return summarize_thesis_groups(
        groups=groups,
        output_path=output_path,
        tail_points=tail_points,
        run_length_limit=run_length_limit,
        run_length_limit_overrides=run_length_limit_overrides,
    )


def summarize_thesis_groups(
    *,
    groups: Sequence[ExperimentGroup],
    output_path: Path,
    tail_points: int,
    run_length_limit: int = THESIS_RUN_LENGTH,
    run_length_limit_overrides: Mapping[str, int] | None = None,
) -> Path:
    summary = summarize_experiment_groups(
        groups,
        tail_points=tail_points,
        run_length_limit=run_length_limit,
        run_length_limit_overrides=run_length_limit_overrides,
        cut_at_limit=True,
    )
    return write_experiment_summary(summary, output_path)
