from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import PolicyVariant, run_experiment as run_mjw_wall_experiment
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.xlstm.mlstm import MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS

EXPERIMENT_RUN_NAME = "mjw_po_wall_medium_1024x4_250M_recurrent"
EXPERIMENT_TOTAL_TIMESTEPS = 250_000_000
SCENARIO_KWARGS: dict[str, object] = dict(PO_WALL_MEDIUM_SCENARIO_KWARGS)
SCENARIO_KWARGS["continuous_connector_actions"] = False
TemporalModelVariant = Literal["lstm", "mlstm", "slstm", "smlstm"]


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        policy_variant: PolicyVariant = "r_mat_qcc",
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        mat_decoder_lr_multiplier: float = 0.25,
        include_actor_head_lr_multiplier: bool = False,
        temporal_model_variant: TemporalModelVariant = "lstm",
        rmat_temporal_residual: bool = False,
        rmat_temporal_layer_norm: bool = False,
) -> None:
    temporal_model_cls, temporal_model_config = _make_temporal_model_specs(temporal_model_variant)
    run_mjw_wall_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=False,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        mat_decoder_lr_multiplier=mat_decoder_lr_multiplier,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
        rmat_temporal_model_cls=temporal_model_cls,
        rmat_temporal_model_config=temporal_model_config,
        rmat_temporal_residual=rmat_temporal_residual,
        rmat_temporal_layer_norm=rmat_temporal_layer_norm,
    )


def _make_temporal_model_specs(variant: TemporalModelVariant) -> tuple[object | None, object | None]:
    if variant == "lstm":
        return None, None
    if variant == "mlstm":
        return MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=4)
    if variant == "slstm":
        return SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=4)
    if variant == "smlstm":
        return (
            [SLSTMTemporalSequenceModel, MLSTMTemporalSequenceModel],
            [
                SLSTMTemporalSequenceModelConfig(num_heads=4),
                MLSTMTemporalSequenceModelConfig(num_heads=4),
            ],
        )
    raise ValueError(f"Unknown temporal_model_variant={variant!r}")
