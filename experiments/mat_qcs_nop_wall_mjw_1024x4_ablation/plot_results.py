from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_logs.experiment_results import ExperimentPlotSelection, plot_experiment_results


RUNS_DIR = REPO_ROOT / "runs"
EXPERIMENT_RUN_DIR_CANDIDATES = (
    RUNS_DIR / "mat_qcs_nop_swarm_bots_wall_mjw_1024x4_ablation",
    RUNS_DIR / "mat_nop_swarm_bots_wall_mjw_1024x4_ablation",
)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
BASELINE_GROUP = "sign_magnitude_beta_nop"
GROUP_ORDER = (
    BASELINE_GROUP,
    "gsde_nop",
    "beta_nop",
    "squashed_diag_gaussian_nop",
    "sign_magnitude_beta_nop_wall_pass_skew_2",
    "sign_magnitude_beta_nop_shuffle_agents",
    "sign_magnitude_beta_nop_shuffle_agents_preserve_prefix",
    "sign_magnitude_beta_no_nop",
    "sticky_sign_magnitude_beta_nop_squared_relu",
    "sign_magnitude_beta_nop_init_gain_0_01",
    "sign_magnitude_beta_nop_init_gain_0_01_projections_1",
    "sign_magnitude_beta_nop_init_gain_0_1",
    "sign_magnitude_beta_nop_init_gain_0_1_projections_1",
    "sign_magnitude_beta_nop_value_regressor_init_gain_0_1",
    "sign_magnitude_beta_nop_context_tokens_only",
    "sign_magnitude_beta_nop_no_agent_embeddings",
    "sign_magnitude_beta_nop_context_tokens_only_no_agent_embeddings",
)
DISPLAY_NAME_OVERRIDES = {
    BASELINE_GROUP: "Sign-Magnitude Beta + NOP",
    "gsde_nop": "gSDE + NOP",
    "beta_nop": "Beta + NOP",
    "squashed_diag_gaussian_nop": "Squashed Diag Gaussian + NOP",
    "sign_magnitude_beta_nop_wall_pass_skew_2": "Sign-Magnitude Beta + NOP, wall-pass skew 2",
    "sign_magnitude_beta_nop_shuffle_agents": "Sign-Magnitude Beta + NOP, shuffled agents",
    "sign_magnitude_beta_nop_shuffle_agents_preserve_prefix": (
        "Sign-Magnitude Beta + NOP, shuffled active-prefix agents"
    ),
    "sign_magnitude_beta_no_nop": "Sign-Magnitude Beta, no NOP",
    "sticky_sign_magnitude_beta_nop_squared_relu": "Sticky Sign-Magnitude Beta + NOP, Squared ReLU",
    "sign_magnitude_beta_nop_init_gain_0_01": "Sign-Magnitude Beta + NOP, hidden init gain 0.01",
    "sign_magnitude_beta_nop_init_gain_0_01_projections_1": (
        "Sign-Magnitude Beta + NOP, hidden init gain 0.01, projections 1.0"
    ),
    "sign_magnitude_beta_nop_init_gain_0_1": "Sign-Magnitude Beta + NOP, hidden init gain 0.1",
    "sign_magnitude_beta_nop_init_gain_0_1_projections_1": (
        "Sign-Magnitude Beta + NOP, hidden init gain 0.1, projections 1.0"
    ),
    "sign_magnitude_beta_nop_value_regressor_init_gain_0_1": (
        "Sign-Magnitude Beta + NOP, value regressor init gain 0.1"
    ),
    "sign_magnitude_beta_nop_context_tokens_only": "Sign-Magnitude Beta + NOP, context tokens only",
    "sign_magnitude_beta_nop_no_agent_embeddings": "Sign-Magnitude Beta + NOP, no agent embeddings",
    "sign_magnitude_beta_nop_context_tokens_only_no_agent_embeddings": (
        "Sign-Magnitude Beta + NOP, context tokens only, no agent embeddings"
    ),
}
PAIRWISE_VARIANT_GROUPS = tuple(group_name for group_name in GROUP_ORDER if group_name != BASELINE_GROUP)
VARIANT_FAMILY_SELECTIONS = (
    ExperimentPlotSelection(
        name="action_distribution_variants",
        group_names=(
            BASELINE_GROUP,
            "gsde_nop",
            "beta_nop",
            "squashed_diag_gaussian_nop",
        ),
        title_suffix="Action Distribution Variants",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="sign_magnitude_beta_ablation_variants",
        group_names=(
            BASELINE_GROUP,
            "sign_magnitude_beta_nop_wall_pass_skew_2",
            "sign_magnitude_beta_nop_shuffle_agents",
            "sign_magnitude_beta_nop_shuffle_agents_preserve_prefix",
            "sign_magnitude_beta_no_nop",
            "sticky_sign_magnitude_beta_nop_squared_relu",
        ),
        title_suffix="Sign-Magnitude Beta Ablation Variants",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="shuffle_variants",
        group_names=(
            BASELINE_GROUP,
            "sign_magnitude_beta_nop_shuffle_agents",
            "sign_magnitude_beta_nop_shuffle_agents_preserve_prefix",
        ),
        title_suffix="Shuffle Variants",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="nop_ablation",
        group_names=(
            BASELINE_GROUP,
            "sign_magnitude_beta_no_nop",
        ),
        title_suffix="NOP Ablation",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="initialization_variants",
        group_names=(
            BASELINE_GROUP,
            "sign_magnitude_beta_nop_init_gain_0_01",
            "sign_magnitude_beta_nop_init_gain_0_01_projections_1",
            "sign_magnitude_beta_nop_init_gain_0_1",
            "sign_magnitude_beta_nop_init_gain_0_1_projections_1",
            "sign_magnitude_beta_nop_value_regressor_init_gain_0_1",
        ),
        title_suffix="Initialization Variants",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
    ExperimentPlotSelection(
        name="context_agent_embedding_variants",
        group_names=(
            BASELINE_GROUP,
            "sign_magnitude_beta_nop_context_tokens_only",
            "sign_magnitude_beta_nop_no_agent_embeddings",
            "sign_magnitude_beta_nop_context_tokens_only_no_agent_embeddings",
        ),
        title_suffix="Context-Only And Agent Embedding Variants",
        required_group_names=(BASELINE_GROUP,),
        output_subdir="selections",
    ),
)
EXTRA_PLOT_SELECTIONS = tuple(
    ExperimentPlotSelection(
        name=f"pair_{BASELINE_GROUP}_vs_{variant_group}",
        group_names=(BASELINE_GROUP, variant_group),
        title_suffix=(
            f"{DISPLAY_NAME_OVERRIDES[BASELINE_GROUP]} vs "
            f"{DISPLAY_NAME_OVERRIDES.get(variant_group, variant_group)}"
        ),
        required_group_names=(BASELINE_GROUP,),
        output_subdir="pairwise",
    )
    for variant_group in PAIRWISE_VARIANT_GROUPS
) + VARIANT_FAMILY_SELECTIONS
THEORETICAL_MAXIMUM = 10.5


def first_existing_dir(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_dir():
            return path

    candidate_paths = "\n".join(f"- {path}" for path in paths)
    raise NotADirectoryError(f"No experiment run dir found. Checked:\n{candidate_paths}")


def legacy_group_name(group_name: str) -> str:
    return (
        group_name
        .replace("sticky_sign_magnitude_beta", "sticky_lr_beta")
        .replace("sign_magnitude_beta", "lr_beta")
    )


def group_source_names(group_name: str) -> tuple[str, ...]:
    legacy_name = legacy_group_name(group_name)
    if legacy_name == group_name:
        return (group_name,)
    return (group_name, legacy_name)


def build_extra_group_sources() -> dict[str, tuple[Path, ...]]:
    sources = {
        group_name: tuple(
            run_dir / source_name
            for run_dir in EXPERIMENT_RUN_DIR_CANDIDATES
            for source_name in group_source_names(group_name)
        )
        for group_name in GROUP_ORDER
    }
    sources[BASELINE_GROUP] += (
        RUNS_DIR / "mat_qcs_nop_swarm_bots_wall_mjw_batch_env_sweep" / "1024x4",
        RUNS_DIR / "mat_nop_swarm_bots_wall_mjw",
    )
    return sources


EXPERIMENT_RUN_DIR = first_existing_dir(EXPERIMENT_RUN_DIR_CANDIDATES)
EXTRA_GROUP_SOURCES = build_extra_group_sources()


def main() -> int:
    result = plot_experiment_results(
        EXPERIMENT_RUN_DIR,
        OUTPUT_DIR,
        group_order=GROUP_ORDER,
        theoretical_maximum=THEORETICAL_MAXIMUM,
        display_name_overrides=DISPLAY_NAME_OVERRIDES,
        extra_group_sources=EXTRA_GROUP_SOURCES,
        extra_plot_selections=EXTRA_PLOT_SELECTIONS,
    )
    for output_path in result.output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
