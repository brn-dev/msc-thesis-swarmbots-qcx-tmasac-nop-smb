from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import run_experiment as run_mjw_wall_experiment
from swarmbots.learn.algos.xlstm.mlstm import (
    MLSTMTemporalSequenceModel,
    MLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS

EXPERIMENT_RUN_NAME = "mjw_po_wall_medium_1024x1_ract_tmasac"
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000
ACTOR_D_MODEL = 256
MAT_TRANSFORMER_FF_CONFIG = MLPConfig(hidden_dims=[512, 512])
SCENARIO_KWARGS: dict[str, object] = dict(PO_WALL_MEDIUM_SCENARIO_KWARGS)
SCENARIO_KWARGS["continuous_connector_actions"] = True
TemporalModelVariant = Literal["mat", "lstm", "slstm", "smlstm"]
MLPLayout = Literal["small_end", "big_end", "two_small"]


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        temporal_model_variant: TemporalModelVariant,
        mlp_layout: MLPLayout | None,
) -> None:
    temporal_model_cls, temporal_model_config = _make_temporal_model_specs(temporal_model_variant)
    transformer_ff_config, inter_module_mlp = _make_mlp_layout(mlp_layout)
    run_mjw_wall_experiment(
        num_envs=1024,
        rollout_steps_per_env=1,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        policy_variant="segment_tmasac" if temporal_model_variant == "mat" else "r_tmasac",
        mat_add_agent_embeddings=False,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=SCENARIO_KWARGS,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        sac_batch_size=16,
        sac_buffer_capacity_per_env=1024,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=16,
        sac_temporal_state_storage_dtype=torch.float16,
        mat_encoder_transformer_ff_config=(
            MAT_TRANSFORMER_FF_CONFIG
            if temporal_model_variant == "mat"
            else None
        ),
        rmat_actor_d_model=ACTOR_D_MODEL,
        rmat_actor_transformer_ff_config=transformer_ff_config,
        rmat_actor_inter_module_mlp=inter_module_mlp,
        rmat_temporal_model_cls=temporal_model_cls,
        rmat_temporal_model_config=temporal_model_config,
        rmat_temporal_residual=False,
        rmat_temporal_layer_norm=False,
        rmat_use_temporal_output_projection=False,
    )


def _make_temporal_model_specs(variant: TemporalModelVariant) -> tuple[object | None, object | None]:
    if variant == "mat":
        return None, None
    if variant == "lstm":
        return None, None
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


def _make_mlp_layout(layout: MLPLayout | None) -> tuple[MLPConfig | None, bool]:
    if layout is None:
        return None, False
    if layout == "small_end":
        return MLPConfig(hidden_dims=[ACTOR_D_MODEL * 2]), False
    if layout == "big_end":
        return MLPConfig(hidden_dims=[ACTOR_D_MODEL * 2, ACTOR_D_MODEL * 2]), False
    if layout == "two_small":
        return MLPConfig(hidden_dims=[ACTOR_D_MODEL * 2]), True
    raise ValueError(f"Unknown mlp_layout={layout!r}")
