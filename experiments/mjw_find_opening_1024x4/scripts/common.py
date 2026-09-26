from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import PolicyVariant, run_experiment as run_mjw_find_opening_experiment
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode

EXPERIMENT_RUN_NAME = "mjw_find_opening_1024x4"
SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": False}


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        policy_variant: PolicyVariant = "mat_qcc",
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        mat_decoder_lr_multiplier: float = 0.25,
        include_actor_head_lr_multiplier: bool = False,
) -> None:
    run_mjw_find_opening_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        nop_add_agent_embeddings_transition_model=False,
        use_transition_obs=False,
        use_nop=True,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="find_opening",
        scenario_kwargs=SCENARIO_KWARGS,
        mat_decoder_lr_multiplier=mat_decoder_lr_multiplier,
        include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
    )
