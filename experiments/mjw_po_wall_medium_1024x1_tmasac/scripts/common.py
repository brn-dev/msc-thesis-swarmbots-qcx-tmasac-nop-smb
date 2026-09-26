from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    run_experiment as run_mjw_wall_experiment,
)
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadKind
from swarmbots.learn.nn_components.feed_forward import FeedForwardConfig
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS

EXPERIMENT_RUN_NAME = "mjw_po_wall_medium_1024x1_tmasac"
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000
SCENARIO_KWARGS: dict[str, object] = dict(PO_WALL_MEDIUM_SCENARIO_KWARGS)
SCENARIO_KWARGS["continuous_connector_actions"] = True


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        continuous_action_dist: ContinuousActionDistVariant,
        use_nop: bool = True,
        sac_learning_rate: float = 3e-4,
        sac_ent_coef_learning_rate: float | None = None,
        sac_ent_coef: float | str = "auto_0.05",
        sac_target_entropy: float | str = "auto_0.1",
        mat_encoder_transformer_ff_config: FeedForwardConfig | None = None,
        actor_head_kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
) -> None:
    run_mjw_wall_experiment(
        num_envs=1024,
        rollout_steps_per_env=1,
        variant_name=(
            variant_name
            if continuous_action_dist == "gumbel_softmax_sign_magnitude_beta"
            else f"{variant_name}_{continuous_action_dist}"
        ),
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant="tmasac",
        mat_add_agent_embeddings=False,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=use_nop,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        sac_learning_rate=sac_learning_rate,
        sac_ent_coef_learning_rate=sac_ent_coef_learning_rate,
        sac_ent_coef=sac_ent_coef,
        sac_target_entropy=sac_target_entropy,
        mat_encoder_transformer_ff_config=mat_encoder_transformer_ff_config,
        tmasac_actor_head_kind=actor_head_kind,
    )
