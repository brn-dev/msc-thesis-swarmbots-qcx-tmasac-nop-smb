from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.post_thesis_mjw_po_wall_medium_action_sampling.scripts.common import (
    ACTOR_ACTION_SAMPLE_OPTIONS,
    DISTRIBUTIONS,
    EXPERIMENT_RUN_NAME,
    TARGET_ACTION_SAMPLES,
    action_sampling_group_name,
)
from experiments.post_thesis_action_dist_plot_common import make_thesis_baseline_sources
from experiments.thesis_mjw_po_wall_medium.plot_results import (
    EXTRA_GROUP_SOURCES as THESIS_EXTRA_GROUP_SOURCES,
    EXPERIMENT_RUN_DIR as THESIS_EXPERIMENT_RUN_DIR,
)
from experiments.thesis_plot_common import PO_WALL_MEDIUM_SCENARIO_TITLE
from experiments.tmasac_experiment_common import EXPERIMENT_TOTAL_TIMESTEPS
from plot_logs.experiment_results import plot_experiment_results

EXPERIMENT_RUN_DIR = REPO_ROOT / "runs" / EXPERIMENT_RUN_NAME
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
BASELINE_GROUP = "tmasac_baseline"


@dataclass(frozen=True)
class ActionSamplingGroupSpec:
    name: str
    display_name: str
    color: str
    linestyle: str


_DISTRIBUTION_STYLES = {
    "signmag_beta": ("Sign-mag Beta (Gumbel ST)", "#0072B2"),
    "rqs_6": ("RQS-6", "#CC79A7"),
    "bernstein_8": ("Bernstein-8", "#D55E00"),
}
_ACTOR_SAMPLE_LINESTYLES = {6: "-", 8: "--"}
MULTISAMPLE_GROUP_SPECS = tuple(
    ActionSamplingGroupSpec(
        name=action_sampling_group_name(
            distribution=distribution,
            actor_action_samples=actor_action_samples,
        ),
        display_name=(
            f"{_DISTRIBUTION_STYLES[distribution][0]}, "
            f"actor={actor_action_samples}, target={TARGET_ACTION_SAMPLES}"
        ),
        color=_DISTRIBUTION_STYLES[distribution][1],
        linestyle=_ACTOR_SAMPLE_LINESTYLES[actor_action_samples],
    )
    for distribution in DISTRIBUTIONS
    for actor_action_samples in ACTOR_ACTION_SAMPLE_OPTIONS
)
GROUP_SPECS = (
    ActionSamplingGroupSpec(
        name=BASELINE_GROUP,
        display_name="TMASAC baseline (actor=1, target=1)",
        color="#666666",
        linestyle=":",
    ),
    *MULTISAMPLE_GROUP_SPECS,
)
MULTISAMPLE_GROUP_ORDER = tuple(spec.name for spec in MULTISAMPLE_GROUP_SPECS)
GROUP_ORDER = tuple(spec.name for spec in GROUP_SPECS)
DISPLAY_NAME_OVERRIDES = {spec.name: spec.display_name for spec in GROUP_SPECS}
GROUP_COLOR_OVERRIDES = {spec.name: spec.color for spec in GROUP_SPECS}
GROUP_LINESTYLE_OVERRIDES = {spec.name: spec.linestyle for spec in GROUP_SPECS}
BASELINE_SOURCES = {
    BASELINE_GROUP: make_thesis_baseline_sources(
        thesis_experiment_run_dir=THESIS_EXPERIMENT_RUN_DIR,
        thesis_extra_group_sources=THESIS_EXTRA_GROUP_SOURCES,
        algorithm_variant=BASELINE_GROUP,
    )
}


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        group_filter=GROUP_ORDER,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=BASELINE_SOURCES,
        group_color_overrides=GROUP_COLOR_OVERRIDES,
        group_linestyle_overrides=GROUP_LINESTYLE_OVERRIDES,
        run_length_limit=EXPERIMENT_TOTAL_TIMESTEPS,
        cut_at_limit=True,
        title_suffix=(
            f"{PO_WALL_MEDIUM_SCENARIO_TITLE}, "
            f"target samples={TARGET_ACTION_SAMPLES}"
        ),
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
