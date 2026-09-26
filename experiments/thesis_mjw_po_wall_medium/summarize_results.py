from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.thesis_mjw_po_wall_medium.plot_results import (
    EXPERIMENT_RUN_DIR,
    EXTRA_GROUP_SOURCES,
)
from experiments.thesis_mjw_po_wall_medium.plot_250m_results import (
    EXPERIMENT_RUN_DIR as EXPERIMENT_RUN_DIR_250M,
    GROUP_ORDER as PPO_GROUP_NAMES,
)
from experiments.thesis_mjw_po_wall_medium.scripts.common_250m import (
    TOTAL_TIMESTEPS as TOTAL_TIMESTEPS_250M,
)
from experiments.thesis_plot_common import (
    THESIS_DISPLAY_NAMES,
    THESIS_GROUP_ORDER,
)
from experiments.thesis_result_summary import (
    parse_summary_args,
    summarize_thesis_groups,
)
from plot_logs.experiment_results import ExperimentGroup, load_experiment_groups

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "results" / "final_metrics.json"
NON_PPO_EXTRA_GROUP_SOURCES = {
    group_name: sources
    for group_name, sources in EXTRA_GROUP_SOURCES.items()
    if group_name not in PPO_GROUP_NAMES
}
RUN_LENGTH_LIMIT_OVERRIDES = {
    f"{group_name}_250m": TOTAL_TIMESTEPS_250M for group_name in PPO_GROUP_NAMES
}


def main() -> int:
    args = parse_summary_args(default_output_path=DEFAULT_OUTPUT_PATH)
    groups = load_experiment_groups(
        EXPERIMENT_RUN_DIR,
        group_order=THESIS_GROUP_ORDER,
        display_name_overrides=THESIS_DISPLAY_NAMES,
        extra_group_sources=NON_PPO_EXTRA_GROUP_SOURCES,
    )
    long_horizon_groups = load_experiment_groups(
        EXPERIMENT_RUN_DIR_250M,
        group_order=PPO_GROUP_NAMES,
        group_filter=PPO_GROUP_NAMES,
        display_name_overrides=THESIS_DISPLAY_NAMES,
    )
    groups = [
        ExperimentGroup(
            name=f"{group.name}_100m",
            display_name=f"{group.display_name} (100M)",
            runs=group.runs,
        )
        if group.name in PPO_GROUP_NAMES
        else group
        for group in groups
    ]
    groups.extend(
        ExperimentGroup(
            name=f"{group.name}_250m",
            display_name=f"{group.display_name} (250M)",
            runs=group.runs,
        )
        for group in long_horizon_groups
    )
    output_path = summarize_thesis_groups(
        groups=groups,
        output_path=args.output,
        tail_points=args.tail_points,
        run_length_limit_overrides=RUN_LENGTH_LIMIT_OVERRIDES,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
