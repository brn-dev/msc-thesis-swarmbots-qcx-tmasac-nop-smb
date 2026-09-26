from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.entropy_utils import (
    AgentActionsReduction,
    EntropyLossConfig,
)
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import (
    MATQCXPolicy,
    MATQCXPolicyConfig,
)
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.sac.sac_nop import SACNOPConfig
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadConfig
from swarmbots.learn.algos.sac.tmasac_policy import (
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.obs_indices import ObsIndices

FeedForwardTransformerPolicyVariant = Literal["mat_qcx", "mat_dec", "mat_ind", "tmasac"]


@dataclass(frozen=True)
class MATInitGains:
    obs_encoder: float = 1.0
    obs_encoder_projection: float | None = 1.0
    encoder_transformer_ff: float | None = 1.0
    decoder_token_encoder: float = 1.0
    decoder_token_encoder_projection: float | None = 1.0
    decoder_transformer_ff: float | None = 1.0
    actor_head: float = 1.0
    action_net: float = 0.01
    critic_local_projection: float = 1.0
    critic_value_regressor: float = 1.0
    critic_value_head: float = 0.01


@dataclass(frozen=True)
class NOPInitGains:
    pre_transition: float = 1.0
    transition_coembed: float = 1.0
    transition_transformer_ff: float | None = 1.0
    transition_head: float = 0.01
    pre_predictors: float = 1.0
    predictors: float = 0.01


@dataclass(frozen=True)
class MATNormalizationConfig:
    normalize_obs_inputs: bool = False
    normalize_encoder_tokens: bool = False
    normalize_query_input: bool = False
    normalize_context_input: bool = False
    normalize_action_input: bool = False
    normalize_memory_input: bool = False
    normalize_query_tokens: bool = False
    normalize_context_tokens: bool = False
    normalize_action_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    normalize_prev_binary_actions: bool = False


def make_benchmark_transformer_policy(
    *,
    env: Any,
    policy_variant: FeedForwardTransformerPolicyVariant,
    use_nop: bool,
    obs_indices: ObsIndices,
    compile_modules: bool,
    mat_init_gains: MATInitGains,
    nop_init_gains: NOPInitGains,
    mat_normalization: MATNormalizationConfig,
    enc_d_model: int = 256,
    enc_nhead: int = 4,
    dec_d_model: int = 128,
    dec_nhead: int = 2,
    world_model_loss_coef: float = 0.1,
    world_model_num_next_steps: int = 3,
    transition_model_d_model: int = 128,
) -> MATQCXPolicy | MATDecPolicy | MATIndPolicy | TMASACPolicy:
    is_sac = policy_variant == "tmasac"
    entropy_config = EntropyLossConfig(
        entropy_floor=0.35,
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )
    magnitude_entropy_config = EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )
    continuous_config = (
        GumbelSoftmaxSignMagnitudeBetaConfig(
            ent_loss_coef=0.0,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=entropy_config,
            beta_ent_loss_config=magnitude_entropy_config,
        )
        if is_sac
        else SignMagnitudeBetaConfig(
            ent_loss_coef=1e-3,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=entropy_config,
            beta_ent_loss_config=magnitude_entropy_config,
        )
    )
    bernoulli_config = BernoulliConfig(initial_prob=0.8, ent_loss_coef=1e-3)
    popart_config = PopArtConfig(beta=5e-4, init_sigma=0.65)
    encoder_config = MATEncoderConfig(
        d_model=enc_d_model,
        nhead=enc_nhead,
        num_layers=2,
        dim_feedforward=enc_d_model * 2,
        act_fn_cls=nn.GELU,
        add_agent_embeddings=False,
        linear_init_gain=mat_init_gains.obs_encoder,
        linear_projection_init_gain=mat_init_gains.obs_encoder_projection,
        transformer_ff_init_gain=mat_init_gains.encoder_transformer_ff,
        transformer_ff_config=MLPConfig(hidden_dims=[512, 512]),
        local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
    )
    ppo_critic_config = MATQCSCriticConfig(
        n_local_projection_hidden_layers=2,
        n_value_regressor_hidden_layers=1,
        use_popart=not is_sac,
        popart_config=popart_config,
        local_projection_init_gain=mat_init_gains.critic_local_projection,
        value_regressor_init_gain=mat_init_gains.critic_value_regressor,
        value_head_init_gain=mat_init_gains.critic_value_head,
    )
    nop_config = make_sac_nop_config(
        use_nop=use_nop,
        obs_indices=obs_indices,
        source_latent_dim=enc_d_model,
        nop_init_gains=nop_init_gains,
        compile_modules=compile_modules,
        world_model_loss_coef=world_model_loss_coef,
        num_next_steps=world_model_num_next_steps,
        transition_model_d_model=transition_model_d_model,
        transition_model_nhead=2,
        act_fn_cls=nn.GELU,
    )
    return make_feedforward_transformer_policy(
        env=env,
        policy_variant=policy_variant,
        encoder_config=encoder_config,
        ppo_critic_config=ppo_critic_config,
        tmasac_critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=2,
            n_value_regressor_hidden_layers=1,
            use_popart=False,
            popart_config=popart_config,
            local_projection_init_gain=mat_init_gains.critic_local_projection,
            value_regressor_init_gain=mat_init_gains.critic_value_regressor,
            value_head_init_gain=mat_init_gains.critic_value_head,
        ),
        continuous_config=continuous_config,
        bernoulli_config=bernoulli_config,
        nop_config=nop_config,
        dec_d_model=dec_d_model,
        dec_nhead=dec_nhead,
        mat_init_gains=mat_init_gains,
        mat_normalization=mat_normalization,
        act_fn_cls=nn.GELU,
        compile_modules=compile_modules,
        compile_mode="default",
    )


def make_feedforward_transformer_policy(
    *,
    env: Any,
    policy_variant: FeedForwardTransformerPolicyVariant,
    encoder_config: MATEncoderConfig,
    ppo_critic_config: MATQCSCriticConfig,
    tmasac_critic_config: TMASACCriticConfig,
    continuous_config: Any,
    bernoulli_config: BernoulliConfig,
    nop_config: SACNOPConfig,
    dec_d_model: int,
    dec_nhead: int,
    mat_init_gains: MATInitGains,
    mat_normalization: MATNormalizationConfig,
    act_fn_cls: Any,
    compile_modules: bool,
    compile_mode: str,
    max_agents: int = 20,
) -> MATQCXPolicy | MATDecPolicy | MATIndPolicy | TMASACPolicy:
    if policy_variant == "tmasac":
        return TMASACPolicy(
            env=env,
            config=TMASACPolicyConfig(
                actor_encoder_config=encoder_config,
                critic_encoder_config=encoder_config,
                actor_head_config=TMASACActorHeadConfig(
                    hidden_dims=[dec_d_model],
                    normalize_input=mat_normalization.normalize_actor_head_input,
                    init_gain=mat_init_gains.actor_head,
                ),
                critic_config=tmasac_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                max_agents=max_agents,
                nop_config=nop_config,
                compile_modules=compile_modules,
                compile_mode=compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )
    if policy_variant == "mat_dec":
        return MATDecPolicy(
            env=env,
            config=MATDecPolicyConfig(
                encoder_config=encoder_config,
                critic_config=ppo_critic_config,
                actor_head_hidden_dims=[dec_d_model, dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=max_agents,
                compile_modules=compile_modules,
                compile_mode=compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )
    if policy_variant == "mat_ind":
        return MATIndPolicy(
            env=env,
            config=MATIndPolicyConfig(
                encoder_config=encoder_config,
                critic_config=ppo_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=max_agents,
                compile_modules=compile_modules,
                compile_mode=compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )
    if policy_variant != "mat_qcx":
        raise ValueError(
            f"Unsupported feed-forward transformer policy variant: {policy_variant}"
        )
    return MATQCXPolicy(
        env=env,
        config=MATQCXPolicyConfig(
            encoder_config=encoder_config,
            decoder_config=MATQCXDecoderConfig(
                d_model=dec_d_model,
                nhead=dec_nhead,
                num_layers=2,
                dim_feedforward=dec_d_model * 2,
                add_agent_embeddings=False,
                token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                actor_head_init_gain=mat_init_gains.actor_head,
                context_encoder_hidden_dims=[dec_d_model],
                action_encoder_dims=[dec_d_model, dec_d_model],
                memory_dims=None,
                normalize_context_input=mat_normalization.normalize_context_input,
                normalize_action_input=mat_normalization.normalize_action_input,
                normalize_memory_input=mat_normalization.normalize_memory_input,
                normalize_action_tokens=mat_normalization.normalize_action_tokens,
                normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
            ),
            critic_config=ppo_critic_config,
            dropout=0.0,
            act_fn_cls=act_fn_cls,
            continuous_config=continuous_config,
            bernoulli_config=bernoulli_config,
            max_agents=max_agents,
            compile_modules=compile_modules,
            compile_mode=compile_mode,
            action_net_init_gain=mat_init_gains.action_net,
        ),
    )


def make_sac_nop_config(
    *,
    use_nop: bool,
    obs_indices: ObsIndices | None,
    source_latent_dim: int,
    nop_init_gains: NOPInitGains,
    compile_modules: bool,
    world_model_loss_coef: float,
    num_next_steps: int,
    transition_model_d_model: int,
    transition_model_nhead: int,
    act_fn_cls: Any,
    add_agent_embeddings_transition_model: bool = False,
    skip_first_transition_for_critic: bool = True,
    compile_mode: str = "default",
) -> SACNOPConfig:
    if not use_nop:
        return SACNOPConfig(enabled=False, num_next_steps=num_next_steps)
    if obs_indices is None:
        raise ValueError(
            "obs_indices is required when building TMASAC with NOP enabled."
        )
    return SACNOPConfig(
        enabled=True,
        num_next_steps=num_next_steps,
        skip_first_transition_for_critic=skip_first_transition_for_critic,
        nop_loss_coef=world_model_loss_coef,
        compile_modules=compile_modules,
        compile_mode=compile_mode,
        act_fn_cls=act_fn_cls,
        latent_projection_hidden_dims=[source_latent_dim, source_latent_dim],
        pre_predictors_hidden_dims=[transition_model_d_model, transition_model_d_model],
        scalar_predictor_hidden_dims=[],
        angle_predictor_hidden_dims=[],
        rot6d_predictor_hidden_dims=[],
        binary_predictor_hidden_dims=[],
        latent_projection_init_gain=nop_init_gains.pre_transition,
        pre_predictors_init_gain=nop_init_gains.pre_predictors,
        predictor_init_gain=nop_init_gains.predictors,
        transition_model_dropout=0.0,
        transition_model_d_model=transition_model_d_model,
        transition_model_nhead=transition_model_nhead,
        transition_model_num_layers=2,
        transition_model_dim_feedforward=transition_model_d_model * 2,
        transition_model_add_agent_embeddings=add_agent_embeddings_transition_model,
        transition_model_coembed_hidden_dims=[transition_model_d_model],
        transition_model_coembed_init_gain=nop_init_gains.transition_coembed,
        transition_model_head_init_gain=nop_init_gains.transition_head,
        transition_model_transformer_ff_init_gain=nop_init_gains.transition_transformer_ff,
        scalar_loss_fn="smooth_l1",
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=obs_indices.local_scalar_indices,
            local_angle_target_indices=obs_indices.local_angle_indices,
            local_rot6d_target_indices=obs_indices.local_rot6d_indices,
            local_binary_target_indices=obs_indices.local_binary_indices,
        ),
    )


def make_mat_parameter_lr_multipliers(
    *,
    policy_variant: FeedForwardTransformerPolicyVariant,
    decoder_lr_multiplier: float,
) -> dict[str, float]:
    if decoder_lr_multiplier == 1.0 or policy_variant != "mat_qcx":
        return {}
    return {
        prefix: decoder_lr_multiplier
        for prefix in (
            "decoder",
            "action_input_norm",
            "action_encoder",
            "action_token_norm",
            "memory_input_norm",
            "memory_encoder",
            "memory_token_norm",
        )
    }
