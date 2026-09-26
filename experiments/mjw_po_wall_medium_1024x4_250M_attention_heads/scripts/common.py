from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import PolicyVariant, run_experiment as run_mjw_wall_experiment
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS


EXPERIMENT_RUN_NAME = "mjw_po_wall_medium_1024x4_250M_attention_heads"
EXPERIMENT_TOTAL_TIMESTEPS = 250_000_000
SCENARIO_KWARGS: dict[str, object] = dict(PO_WALL_MEDIUM_SCENARIO_KWARGS)
SCENARIO_KWARGS["continuous_connector_actions"] = True


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        policy_variant: PolicyVariant,
        encoder_attention_heads: int,
        mat_decoder_lr_multiplier: float = 0.25,
        include_actor_head_lr_multiplier: bool = False,
) -> None:
    run_mjw_wall_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        mat_decoder_lr_multiplier=mat_decoder_lr_multiplier,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
        enc_nhead=encoder_attention_heads,
        dec_nhead=2,
    )
