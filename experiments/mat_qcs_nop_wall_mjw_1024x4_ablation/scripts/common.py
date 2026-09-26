from __future__ import annotations

import sys
from pathlib import Path

from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
from swarmbots.learn.nn_components.activations import ActivationFactory

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    MATInitGains,
    MATNormalizationConfig,
    NOPInitGains,
    PolicyVariant,
    run_experiment,
)

EXPERIMENT_RUN_NAME = "mat_qcs_nop_swarm_bots_wall_mjw_1024x4_ablation"
DEFAULT_SCENARIO_KWARGS: dict[str, object] = {"continuous_connector_actions": False}


def make_mat_hidden_init_gains(
        gain: float,
        *,
        projection_gain: float | None = None,
) -> MATInitGains:
    effective_projection_gain = gain if projection_gain is None else projection_gain
    return MATInitGains(
        obs_encoder=gain,
        obs_encoder_projection=effective_projection_gain,
        encoder_transformer_ff=gain,
        decoder_token_encoder=gain,
        decoder_token_encoder_projection=effective_projection_gain,
        decoder_transformer_ff=gain,
        actor_head=gain,
        critic_local_projection=effective_projection_gain,
        critic_value_regressor=gain,
    )


def make_nop_hidden_init_gains(
        gain: float,
        *,
        projection_gain: float | None = None,
) -> NOPInitGains:
    effective_projection_gain = gain if projection_gain is None else projection_gain
    return NOPInitGains(
        pre_transition=gain,
        transition_coembed=effective_projection_gain,
        transition_transformer_ff=gain,
        pre_predictors=gain,
    )


def run_ablation(
        *,
        variant_name: str,
        entrypoint_path: Path,
        continuous_action_dist: ContinuousActionDistVariant = "sign_magnitude_beta",
        policy_variant: PolicyVariant = "mat_qcs",
        mat_add_agent_embeddings: bool = True,
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        act_fn_cls: ActivationFactory = nn.GELU,
        mat_init_gains: MATInitGains = MATInitGains(),
        nop_init_gains: NOPInitGains = NOPInitGains(),
        mat_normalization: MATNormalizationConfig = MATNormalizationConfig(),
        use_nop: bool = True,
        shuffle_agents: bool = False,
        preserve_inactive_prefix_structure: bool = False,
        include_actor_head_lr_multiplier: bool = False,
        scenario_kwargs: dict[str, object] | None = None,
) -> None:
    effective_scenario_kwargs = {
        **DEFAULT_SCENARIO_KWARGS,
        **({} if scenario_kwargs is None else scenario_kwargs),
    }
    run_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant_name,
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant=policy_variant,
        mat_add_agent_embeddings=mat_add_agent_embeddings,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        act_fn_cls=act_fn_cls,
        mat_init_gains=mat_init_gains,
        nop_init_gains=nop_init_gains,
        mat_normalization=mat_normalization,
        use_nop=use_nop,
        nop_add_agent_embeddings_transition_model=True,
        shuffle_agents=shuffle_agents,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_kwargs=effective_scenario_kwargs,
    )
