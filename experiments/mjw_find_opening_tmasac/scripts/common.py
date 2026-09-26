from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    run_experiment as run_mjw_find_opening_experiment,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import ActorStateCriticInputConfig
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadKind
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import FeedForwardConfig, MLPConfig

EXPERIMENT_RUN_NAME = "mjw_find_opening_tmasac"
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000
ACTOR_D_MODEL = 256
PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM = 344
SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": True}
TemporalModelVariant = Literal["baseline", "lstm", "slstm"]


def run_experiment(
        *,
        variant_name: str,
        entrypoint_path: Path,
        temporal_model_variant: TemporalModelVariant,
        actor_state_critic_input_config: ActorStateCriticInputConfig | None = None,
        separate_observation_action_encoders: bool = False,
        actor_head_kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
        mat_encoder_transformer_ff_config: FeedForwardConfig | None = None,
        rmat_actor_transformer_ff_config: FeedForwardConfig | None = None,
        continuous_action_dist: ContinuousActionDistVariant = "gumbel_softmax_sign_magnitude_beta",
) -> None:
    is_recurrent = temporal_model_variant != "baseline"
    temporal_model_cls, temporal_model_config = _make_temporal_model_specs(temporal_model_variant)
    resolved_mat_encoder_transformer_ff_config = mat_encoder_transformer_ff_config
    if resolved_mat_encoder_transformer_ff_config is None and not is_recurrent:
        resolved_mat_encoder_transformer_ff_config = MLPConfig(hidden_dims=[512, 512])
    resolved_rmat_actor_transformer_ff_config = (
        MLPConfig(hidden_dims=[ACTOR_D_MODEL * 2])
        if rmat_actor_transformer_ff_config is None
        else rmat_actor_transformer_ff_config
    )

    run_mjw_find_opening_experiment(
        num_envs=1024,
        rollout_steps_per_env=1,
        variant_name=(
            variant_name
            if continuous_action_dist == "gumbel_softmax_sign_magnitude_beta"
            else f"{variant_name}_{continuous_action_dist}"
        ),
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant="r_tmasac" if is_recurrent else "tmasac",
        mat_add_agent_embeddings=False,
        mat_encoder_transformer_ff_config=resolved_mat_encoder_transformer_ff_config,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="find_opening",
        scenario_kwargs=SCENARIO_KWARGS,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        sac_batch_size=16 if is_recurrent else None,
        sac_buffer_capacity_per_env=1024 if is_recurrent else None,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=16,
        sac_temporal_state_storage_dtype=torch.float16 if is_recurrent else None,
        rmat_actor_d_model=ACTOR_D_MODEL if is_recurrent else None,
        rmat_actor_transformer_ff_config=(
            resolved_rmat_actor_transformer_ff_config if is_recurrent else None
        ),
        rmat_actor_inter_module_mlp=is_recurrent,
        r_tmasac_actor_state_critic_input_config=actor_state_critic_input_config,
        tmasac_actor_head_kind=actor_head_kind,
        tmasac_separate_observation_action_encoders=separate_observation_action_encoders,
        rmat_temporal_model_cls=temporal_model_cls,
        rmat_temporal_model_config=temporal_model_config,
        rmat_temporal_residual=False,
        rmat_temporal_layer_norm=False,
        rmat_use_temporal_output_projection=not is_recurrent,
        rmat_experimental_compile_lstm=temporal_model_variant == "lstm",
    )


def _make_temporal_model_specs(variant: TemporalModelVariant) -> tuple[object | None, object | None]:
    if variant in {"baseline", "lstm"}:
        return None, None
    if variant == "slstm":
        return SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=4)
    raise ValueError(f"Unknown temporal_model_variant={variant!r}")
