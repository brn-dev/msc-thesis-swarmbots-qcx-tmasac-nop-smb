from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    PolicyVariant,
    run_experiment as run_mjw_wall_experiment,
)
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS

EXPERIMENT_RUN_NAME = "mjw_po_wall_medium_1024x4_250M_bigger_mlps"
EXPERIMENT_TOTAL_TIMESTEPS = 250_000_000
ENCODER_TRANSFORMER_FF_CONFIG = MLPConfig(hidden_dims=[512, 512])
SCENARIO_KWARGS: dict[str, object] = dict(PO_WALL_MEDIUM_SCENARIO_KWARGS)
SCENARIO_KWARGS["continuous_connector_actions"] = True


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        policy_variant: PolicyVariant = "mat_qcc",
        continuous_action_dist: ContinuousActionDistVariant = "sign_magnitude_beta",
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        mat_decoder_lr_multiplier: float = 0.25,
        include_actor_head_lr_multiplier: bool = False,
        use_nop: bool = True,
        sac_ent_coef: float | str = "auto_0.01",
        sac_target_entropy: float | str = "auto_0.1",
) -> None:
    run_mjw_wall_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=use_nop,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        mat_decoder_lr_multiplier=mat_decoder_lr_multiplier,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
        mat_encoder_transformer_ff_config=ENCODER_TRANSFORMER_FF_CONFIG,
        sac_ent_coef=sac_ent_coef,
        sac_target_entropy=sac_target_entropy,
    )
