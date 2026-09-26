from __future__ import annotations

from pathlib import Path
from typing import Literal

from experiments.mjw_experiment_common import MJWScenarioName
from experiments.tmasac_experiment_common import run_tmasac_experiment

PostThesisActionDistVariant = Literal[
    "bernstein_6",
    "bernstein_8",
    "bernstein_12",
    "bernstein_18",
    "rqs_4",
    "rqs_6",
]
PostThesisTMASACVariant = Literal[
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
]
PostThesisAlgorithmVariant = PostThesisTMASACVariant | Literal["mat_ind", "mat_qcx"]

ACTION_DIST_VARIANTS: tuple[PostThesisActionDistVariant, ...] = (
    "bernstein_6",
    "bernstein_8",
    "bernstein_12",
    "bernstein_18",
    "rqs_4",
    "rqs_6",
)
BASELINE_GROUP: PostThesisTMASACVariant = "tmasac_baseline"
SLSTM_TMASAC_GROUP: PostThesisTMASACVariant = "slstm_two_small_actor_state_critic"


def action_dist_group_name(
        variant: PostThesisActionDistVariant,
        *,
        algorithm_variant: PostThesisAlgorithmVariant = BASELINE_GROUP,
) -> str:
    return f"{algorithm_variant}_{variant}"


def run_post_thesis_action_dist_experiment(
        *,
        experiment_run_name: str,
        scenario_name: MJWScenarioName,
        scenario_kwargs: dict[str, object],
        continuous_action_dist: PostThesisActionDistVariant,
        entrypoint_path: Path,
        tmasac_variant: PostThesisTMASACVariant = BASELINE_GROUP,
) -> None:
    run_tmasac_experiment(
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        variant=tmasac_variant,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        use_nop=True,
        variant_name=action_dist_group_name(
            continuous_action_dist,
            algorithm_variant=tmasac_variant,
        ),
    )
