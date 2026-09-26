from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import ContinuousActionDistVariant
from experiments.tmasac_experiment_common import run_tmasac_experiment
from swarmbots.learn.action_dists.action_sampling import ActionSampleStrategy
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

EXPERIMENT_RUN_NAME = "post_thesis_mjw_po_wall_medium_action_sampling"
SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {"continuous_connector_actions": True},
)
TARGET_ACTION_SAMPLES = 4
ACTION_SAMPLE_STRATEGY: ActionSampleStrategy = "stratified"
SamplingDistribution = Literal["signmag_beta", "rqs_6", "bernstein_8"]
ActorActionSamples = Literal[6, 8]
DISTRIBUTIONS: tuple[SamplingDistribution, ...] = (
    "signmag_beta",
    "rqs_6",
    "bernstein_8",
)
ACTOR_ACTION_SAMPLE_OPTIONS: tuple[ActorActionSamples, ...] = (6, 8)

_DISTRIBUTION_CONFIG: dict[SamplingDistribution, ContinuousActionDistVariant] = {
    # Plain sign_magnitude_beta has no reparameterized gradient and is rejected by SAC.
    "signmag_beta": "gumbel_softmax_sign_magnitude_beta",
    "rqs_6": "rqs_6",
    "bernstein_8": "bernstein_8",
}


def action_sampling_group_name(
        *,
        distribution: SamplingDistribution,
        actor_action_samples: ActorActionSamples,
) -> str:
    return f"{distribution}_actor{actor_action_samples}_target{TARGET_ACTION_SAMPLES}"


def run_experiment(
        *,
        distribution: SamplingDistribution,
        actor_action_samples: ActorActionSamples,
        entrypoint_path: Path,
) -> None:
    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        variant="tmasac_baseline",
        entrypoint_path=entrypoint_path,
        continuous_action_dist=_DISTRIBUTION_CONFIG[distribution],
        actor_action_samples=actor_action_samples,
        target_action_samples=TARGET_ACTION_SAMPLES,
        action_sample_strategy=ACTION_SAMPLE_STRATEGY,
        use_nop=True,
        variant_name=action_sampling_group_name(
            distribution=distribution,
            actor_action_samples=actor_action_samples,
        ),
    )
