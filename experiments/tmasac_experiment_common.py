from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    MJWScenarioName,
)
from experiments.mjw_experiment_common import run_experiment as run_mjw_experiment
from swarmbots.learn.action_dists.action_sampling import ActionSampleStrategy
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
)
from swarmbots.learn.algos.sac.sac_nop import SACNOPLatentSource
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import (
    GLUStackConfig,
    MLPConfig,
    SwiGLUConfig,
)

ACTOR_D_MODEL = 256
PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM = 344
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000

TMASACExperimentVariant = Literal[
    "tmasac_baseline",
    "tmasac_shared_encoder",
    "tmasac_swiglu",
    "slstm_two_small_actor_state_critic",
    "slstm_two_small_actor_state_critic_recurrent_skip",
    "slstm_shared_encoder",
    "lstm_two_small_actor_state_critic",
    "slstm_two_small_swiglu_actor_state_critic",
]


def run_tmasac_experiment(
    *,
    experiment_run_name: str,
    scenario_name: MJWScenarioName,
    scenario_kwargs: dict[str, object],
    variant: TMASACExperimentVariant,
    entrypoint_path: Path,
    continuous_action_dist: ContinuousActionDistVariant = "gumbel_softmax_sign_magnitude_beta",
    actor_action_samples: int = 1,
    target_action_samples: int = 1,
    action_sample_strategy: ActionSampleStrategy = "iid",
    use_nop: bool = True,
    include_slstm_memory_strength: bool = False,
    variant_name: str | None = None,
    load_path: str | Path | None = None,
    additional_timesteps: int | None = None,
    evaluation_scenario_kwargs: dict[str, object] | None = None,
    evaluation_milestones: Sequence[float] | None = None,
    evaluation_recording_episodes: int | None = None,
    live_recording_schedule: Mapping[float, int] | None = None,
    disable_connector_actions: bool = False,
    migrate_removed_connector_actions: bool = False,
    transfer_checkpoint: bool = False,
) -> None:
    temporal_model_cls, temporal_model_config = _make_temporal_model_spec(
        variant=variant,
    )
    is_recurrent = temporal_model_cls is not None
    uses_recurrent_shared_encoder = variant == "slstm_shared_encoder"
    uses_shared_encoder = variant == "tmasac_shared_encoder" or uses_recurrent_shared_encoder
    mat_transformer_ff_config, actor_transformer_ff_config = _make_feedforward_configs(
        variant=variant,
    )

    continuation_kwargs: dict[str, object] = {}
    if load_path is not None:
        continuation_kwargs["load_path"] = load_path
    if additional_timesteps is not None:
        continuation_kwargs["additional_timesteps"] = additional_timesteps
    if evaluation_scenario_kwargs is not None:
        continuation_kwargs["evaluation_scenario_kwargs"] = evaluation_scenario_kwargs
    if evaluation_milestones is not None:
        continuation_kwargs["evaluation_milestones"] = evaluation_milestones
    if evaluation_recording_episodes is not None:
        continuation_kwargs["evaluation_recording_episodes"] = evaluation_recording_episodes
    if live_recording_schedule is not None:
        continuation_kwargs["live_recording_schedule"] = live_recording_schedule
    if disable_connector_actions:
        continuation_kwargs["disable_connector_actions"] = True
    if migrate_removed_connector_actions:
        continuation_kwargs["migrate_removed_connector_actions"] = True
    if transfer_checkpoint:
        continuation_kwargs["transfer_checkpoint"] = True

    run_mjw_experiment(
        num_envs=1024,
        rollout_steps_per_env=1,
        variant_name=(
            variant_name
            if variant_name is not None
            else (
                variant
                if continuous_action_dist == "gumbel_softmax_sign_magnitude_beta"
                else f"{variant}_{continuous_action_dist}"
            )
        ),
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant="r_tmasac" if is_recurrent else "tmasac",
        mat_add_agent_embeddings=False,
        mat_encoder_transformer_ff_config=mat_transformer_ff_config,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=use_nop,
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        sac_actor_action_samples=actor_action_samples,
        sac_target_action_samples=target_action_samples,
        sac_action_sample_strategy=action_sample_strategy,
        sac_batch_size=16 if is_recurrent else None,
        sac_buffer_capacity_per_env=1024 if is_recurrent else None,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=16,
        sac_temporal_state_storage_dtype=torch.float16 if is_recurrent else None,
        rmat_actor_d_model=ACTOR_D_MODEL if is_recurrent else None,
        rmat_actor_transformer_ff_config=(
            actor_transformer_ff_config if is_recurrent else None
        ),
        rmat_actor_inter_module_mlp=is_recurrent and not uses_recurrent_shared_encoder,
        r_tmasac_actor_state_critic_input_config=(
            ActorStateCriticInputConfig(
                projection_dim=ACTOR_D_MODEL,
                projection_hidden_dims=(ACTOR_D_MODEL,),
                include_slstm_memory_strength=include_slstm_memory_strength,
            )
            if is_recurrent and not uses_recurrent_shared_encoder
            else None
        ),
        tmasac_shared_encoder_num_layers=2 if uses_shared_encoder else None,
        tmasac_recurrent_shared_encoder=uses_recurrent_shared_encoder,
        tmasac_actor_encoder_num_layers=1 if uses_shared_encoder else 2,
        tmasac_critic_encoder_num_layers=1 if uses_shared_encoder else 2,
        tmasac_nop_latent_source=(
            SACNOPLatentSource.SHARED_ENCODER
            if uses_shared_encoder
            else SACNOPLatentSource.CRITIC
        ),
        nop_skip_first_transition_for_critic=True,
        rmat_temporal_model_cls=temporal_model_cls,
        rmat_temporal_model_config=temporal_model_config,
        rmat_temporal_residual=variant == "slstm_two_small_actor_state_critic_recurrent_skip",
        rmat_temporal_layer_norm=False,
        rmat_use_temporal_output_projection=not is_recurrent,
        rmat_experimental_compile_lstm=variant.startswith("lstm_"),
        **continuation_kwargs,
    )


def _make_feedforward_configs(
    *,
    variant: TMASACExperimentVariant,
) -> tuple[MLPConfig | SwiGLUConfig, MLPConfig | SwiGLUConfig | None]:
    if variant in {
        "tmasac_baseline",
        "tmasac_shared_encoder",
        "slstm_shared_encoder",
    }:
        if variant == "slstm_shared_encoder":
            config = MLPConfig(hidden_dims=[512, 512])
            return config, config
        return MLPConfig(hidden_dims=[512, 512]), None
    if variant == "tmasac_swiglu":
        return _make_stacked_swiglu_config(), None
    if variant in {
        "slstm_two_small_actor_state_critic",
        "slstm_two_small_actor_state_critic_recurrent_skip",
        "lstm_two_small_actor_state_critic",
    }:
        return MLPConfig(hidden_dims=[512]), MLPConfig(hidden_dims=[512])
    if variant == "slstm_two_small_swiglu_actor_state_critic":
        swiglu_config = SwiGLUConfig(hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM)
        return swiglu_config, swiglu_config
    raise ValueError(f"Unknown TMASAC experiment variant: {variant!r}")


def _make_temporal_model_spec(
    *,
    variant: TMASACExperimentVariant,
) -> tuple[type[nn.Module] | None, object | None]:
    if variant.startswith("slstm_"):
        return SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=4)
    if variant.startswith("lstm_"):
        return LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()
    return None, None


def _make_stacked_swiglu_config() -> SwiGLUConfig:
    return SwiGLUConfig(
        hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM,
        stacked=GLUStackConfig(
            n_layers=2,
            pre_norm=nn.LayerNorm,
            norm_first_layer=False,
            residual=True,
            residual_first_layer=False,
        ),
    )
