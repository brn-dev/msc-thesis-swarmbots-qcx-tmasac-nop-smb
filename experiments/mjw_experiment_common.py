from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from torch import nn
import warp as wp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import swarmbots.mjw_env.scenarios.mjw_scenario_presets as mjw_scenario_presets
from experiments.transformer_policy_common import (
    MATInitGains,
    MATNormalizationConfig,
    NOPInitGains,
)
from swarmbots.learn.action_dists.action_sampling import (
    ActionSampleStrategy,
    validate_action_sampling,
)
from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.bernstein_quantile_action_dist import BernsteinQuantileConfig
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.entropy_utils import AgentActionsReduction, EntropyLossConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.rational_quadratic_spline_quantile_action_dist import (
    RationalQuadraticSplineQuantileConfig,
)
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import StickySignMagnitudeBetaConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.action_dists.ternary_sign_magnitude_beta_action_dist import (
    TernarySignMagnitudeBetaConfig,
)
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoderConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_policy import MATQCCPolicy, MATQCCPolicyConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOCriticConfig, MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat import FeedForwardConfig, MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigCriticConfig, MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOCriticConfig, PPOPolicy, PPOPolicyConfig, PopArtConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_qcc_policy import RMATQCCPolicy, RMATQCCPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcx_policy import RMATQCXPolicy, RMATQCXPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcs_policy import RMATQCSPolicy, RMATQCSPolicyConfig
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamplerConfig
from swarmbots.learn.algos.sac.recurrent_sac import RecurrentSAC
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
    RecurrentTMASACPolicy,
    RecurrentTMASACPolicyConfig,
)
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.sac.sac_nop import SACNOPConfig, SACNOPLatentSource
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_actor_heads import (
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
)
from swarmbots.learn.algos.sac.tmasac_policy import (
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.checkpointing import (
    migrate_tmasac_removed_connector_action_dims,
)
from swarmbots.learn.discord_notifications import run_with_discord_notification
from swarmbots.learn.evaluation import (
    DEFAULT_EVALUATION_MILESTONES,
    EvaluationRecordingConfig,
    FrozenEvaluationRunner,
    ScheduledEvaluationHook,
)
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.metrics_logger import MetricsLogger
from swarmbots.learn.nn_components.activations import ActivationFactory, activation_factory_name
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager, ScheduleUnit
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.learn.nn_components.deep_set import DeepSetCriticConfig
from swarmbots.mjw_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import (
    default_bridge,
    default_climb,
    default_dual_payload_plane,
    default_find_opening,
    default_multi_payload_goal,
    default_payload_step,
    default_vertical_reach,
    default_wall,
)
from swarmbots.utils.recording_schedule import DEFAULT_LIVE_RECORDING_SCHEDULE, install_scheduled_recordings
from swarmbots.utils.run_paths import generate_run_id, get_run_id_from_checkpoint_path

ContinuousActionDistVariant = Literal[
    "sticky_sign_magnitude_beta",
    "sign_magnitude_beta",
    "gumbel_softmax_sign_magnitude_beta",
    "ternary_sign_magnitude_beta",
    "gumbel_softmax_sign_magnitude_kumaraswamy",
    "reparameterized_sign_magnitude_kumaraswamy",
    "reparameterized_squashed_gaussian_mixture",
    "bernstein_6",
    "bernstein_8",
    "bernstein_12",
    "bernstein_18",
    "rqs_4",
    "rqs_6",
    "beta",
    "predicted_std",
    "gsde",
    "squashed_diag_gaussian",
]
PolicyVariant = Literal[
    "mat_qcs",
    "mat_qcc",
    "mat_qcx",
    "mat_dec",
    "mat_ind",
    "mat_orig",
    "r_mat_qcs",
    "r_mat_qcc",
    "r_mat_qcx",
    "r_mat_ind",
    "ppo",
    "ppo_small",
    "mappo",
    "mappo_small",
    "tmasac",
    "r_tmasac",
    "segment_tmasac",
]
MJWScenarioName = Literal[
    "wall",
    "bridge",
    "find_opening",
    "climb",
    "vertical_reach",
    "dual_payload",
    "payload_step",
    "multi_payload_goal",
]


def retain_shape_compatible_policy_state(
        state_dict: Mapping[str, Any],
        target_state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep compatible task-independent state and reset global-observation encoders."""
    compatible_state: dict[str, Any] = {}
    skipped_keys: list[str] = []
    for key, value in state_dict.items():
        target_value = target_state_dict.get(key)
        if (
            _is_global_observation_encoder_key(key)
            or target_value is None
            or not hasattr(value, "shape")
            or not hasattr(target_value, "shape")
            or value.shape != target_value.shape
        ):
            skipped_keys.append(key)
            continue
        compatible_state[key] = value

    if skipped_keys:
        logger.warning(
            "Skipped checkpoint entries that are absent or shape-incompatible in "
            f"the transfer target: {skipped_keys}"
        )
    return compatible_state


def _is_global_observation_encoder_key(key: str) -> bool:
    canonical_key = ".".join(
        part for part in key.split(".") if part != "_orig_mod"
    )
    qualified_key = f".{canonical_key}."
    return any(
        component in qualified_key
        for component in (
            ".global_obs_input_norm.",
            ".global_obs_encoder.",
            ".global_input_norm.",
            ".global_encoder.",
        )
    )


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def _get_requested_cuda_idx(argv: Sequence[str] | None = None) -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_idx", "--cuda-idx", "--gpu", type=int, dest="cuda_idx")
    parsed_args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    return parsed_args.cuda_idx


def _configure_cuda_device(*, cuda_idx: int | None) -> None:
    if cuda_idx is None:
        return
    if cuda_idx < 0:
        raise ValueError(f"cuda_idx must be non-negative, got {cuda_idx}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested via --cuda_idx, but torch.cuda.is_available() is False.")

    visible_device_count = torch.cuda.device_count()
    if cuda_idx >= visible_device_count:
        raise RuntimeError(
            f"Requested CUDA device {cuda_idx}, but only {visible_device_count} CUDA device(s) are visible."
        )

    torch.cuda.set_device(cuda_idx)
    wp.set_device(f'cuda:{cuda_idx}')


def _make_scenario(*, scenario_name: MJWScenarioName, scenario_kwargs: dict[str, object] | None) -> Any:
    scenario_factory = {
        "wall": default_wall,
        "bridge": default_bridge,
        "find_opening": default_find_opening,
        "climb": default_climb,
        "vertical_reach": default_vertical_reach,
        "dual_payload": default_dual_payload_plane,
        "payload_step": default_payload_step,
        "multi_payload_goal": default_multi_payload_goal,
    }[scenario_name]
    return scenario_factory(**({} if scenario_kwargs is None else scenario_kwargs))


def _with_default_scenario_kwargs(scenario_kwargs: dict[str, object] | None) -> dict[str, object]:
    return {
        "continuous_connector_actions": True,
        **({} if scenario_kwargs is None else scenario_kwargs),
    }


def _scenario_display_name(*, scenario_name: MJWScenarioName) -> str:
    return {
        "wall": "wall",
        "bridge": "bridge",
        "find_opening": "find-opening",
        "climb": "climb",
        "vertical_reach": "vertical-reach",
        "dual_payload": "dual-payload",
        "payload_step": "payload-step",
        "multi_payload_goal": "multi-payload-goal",
    }[scenario_name]


def _default_experiment_run_name(*, scenario_name: MJWScenarioName) -> str:
    return f"mat_nop_swarm_bots_{scenario_name}_mjw"


def _default_ccd_iterations(*, scenario_name: MJWScenarioName) -> int | None:
    return 4096 if scenario_name in {"dual_payload", "payload_step", "multi_payload_goal"} else None


def _metadata_name(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
            "values": asdict(value),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_metadata_name(item) for item in value]
    return value


def _is_recurrent_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"r_mat_qcs", "r_mat_qcc", "r_mat_qcx", "r_mat_ind", "r_tmasac"}


def _is_sac_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"tmasac", "r_tmasac", "segment_tmasac"}


def _make_staggered_first_episode_lengths(*, episode_length: int, num_envs: int) -> list[int]:
    return [math.ceil((env_idx + 1) * episode_length / num_envs) for env_idx in range(num_envs)]


def make_vector_env(
    *,
    episode_length: int,
    num_envs: int,
    first_episode_lengths: list[int] | None,
    settle_initial_reset: bool,
    device: torch.device,
    scenario_name: MJWScenarioName = "wall",
    ccd_iterations: int | None = None,
    scenario_kwargs: dict[str, object] | None = None,
    compile_env_tensor_operations: bool | None = None,
    env_tensor_operations_compile_mode: str = "default",
) -> MJWSwarmBotsVectorEnv:
    return MJWSwarmBotsVectorEnv(
        scenario=_make_scenario(scenario_name=scenario_name, scenario_kwargs=scenario_kwargs),
        num_envs=num_envs,
        episode_length=episode_length,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=settle_initial_reset,
        device=device,
        ccd_iterations=ccd_iterations,
        compile_tensor_operations=compile_env_tensor_operations,
        tensor_operations_compile_mode=env_tensor_operations_compile_mode,
    )


def wrap_vec_env(
    *,
    vector_env: Any,
    obs_indices: ObsIndices,
    gamma: float,
    use_popart: bool,
    rollout_device: torch.device,
    normalize_prev_binary_actions: bool = False,
    use_transition_obs: bool = False,
    shuffle_agents: bool = False,
    preserve_inactive_prefix_structure: bool = False,
    disable_connector_actions: bool = False,
) -> Any:
    from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
    from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
    from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
    from swarmbots.learn.env_wrappers.torch_progress_guidance_ep_stats_wrapper import (
        TorchProgressGuidanceEpisodeStatsWrapper,
    )
    from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
    from swarmbots.learn.env_wrappers.torch_shuffle_agents_wrapper import TorchShuffleAgentsWrapper
    from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper

    env = SwarmBotsLearnEnvWrapper(
        vector_env,
        device=rollout_device,
        disable_connector_actions=disable_connector_actions,
    )
    if shuffle_agents:
        env = TorchShuffleAgentsWrapper(
            env,
            preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        )
    env = TorchRecordEpisodeStatisticsWrapper(env)
    env = TorchProgressGuidanceEpisodeStatsWrapper(env)
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="local_obs",
        scalar_feature_indices=obs_indices.local_scalar_indices,
        quaternion_indices=obs_indices.local_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="global_obs",
        scalar_feature_indices=obs_indices.global_scalar_indices,
        quaternion_indices=obs_indices.global_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_local_vars",
        scalar_feature_indices=obs_indices.hidden_local_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_local_vars_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_global_vars",
        scalar_feature_indices=obs_indices.hidden_global_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_global_vars_quaternion_indices,
    )
    if use_transition_obs:
        env = TorchTransitionObsWrapper(env, normalize_prev_binary_actions=normalize_prev_binary_actions)
    if not use_popart:
        env = TorchNormalizeRewardWrapper(env, gamma=gamma)
    return env


def split_actuator_joints(actions: Any, actuators_per_limb: int) -> dict[str, Any]:
    return {f"j{i}": actions[..., i::actuators_per_limb] for i in range(actuators_per_limb)}


def set_actuator_gsde_init_joint_stds(
    *,
    policy: Any,
    actuators_per_limb: int,
    joint_stds: list[float],
) -> None:
    from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist

    gsde_dist = next(
        (dist for dist in policy.action_dist.distributions if isinstance(dist, GSDEActionDist)),
        None,
    )

    if gsde_dist is None:
        return

    if len(joint_stds) != actuators_per_limb:
        raise ValueError(
            f"Expected one gSDE initial std per actuator joint, got "
            f"{len(joint_stds)=} {actuators_per_limb=}"
        )

    with torch.no_grad():
        for i, joint_std in enumerate(joint_stds):
            gsde_dist.log_stds[:, i::actuators_per_limb] = math.log(joint_std)


def make_sign_magnitude_categorical_entropy_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        entropy_floor=0.35,
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def make_sign_magnitude_magnitude_entropy_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def make_continuous_config(
        *,
        variant: ContinuousActionDistVariant,
        initial_stickiness: float,
        gsde_init_stds: list[float],
        action_net_init_gain: float = 0.01,
        rsmk_kumaraswamy_ent_scale: float = 0.75,
        ent_loss_coef: float = 1e-3,
) -> (
        StickySignMagnitudeBetaConfig
        | SignMagnitudeBetaConfig
        | GumbelSoftmaxSignMagnitudeBetaConfig
        | TernarySignMagnitudeBetaConfig
        | GumbelSoftmaxSignMagnitudeKumaraswamyConfig
        | ReparameterizedSignMagnitudeKumaraswamyConfig
        | ReparameterizedSquashedGaussianMixtureConfig
        | BernsteinQuantileConfig
        | RationalQuadraticSplineQuantileConfig
        | BetaConfig
        | PredictedStdConfig
        | GSDEConfig
        | SquashedDiagGaussianConfig
):
    if variant == "sticky_sign_magnitude_beta":
        return StickySignMagnitudeBetaConfig(
            stickiness=initial_stickiness,
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "sign_magnitude_beta":
        return SignMagnitudeBetaConfig(
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "gumbel_softmax_sign_magnitude_beta":
        return GumbelSoftmaxSignMagnitudeBetaConfig(
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "ternary_sign_magnitude_beta":
        return TernarySignMagnitudeBetaConfig(
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "gumbel_softmax_sign_magnitude_kumaraswamy":
        return GumbelSoftmaxSignMagnitudeKumaraswamyConfig(
            ent_loss_coef=ent_loss_coef,
            kumaraswamy_ent_scale=rsmk_kumaraswamy_ent_scale,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            kumaraswamy_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "reparameterized_sign_magnitude_kumaraswamy":
        return ReparameterizedSignMagnitudeKumaraswamyConfig(
            ent_loss_coef=ent_loss_coef,
            kumaraswamy_ent_scale=rsmk_kumaraswamy_ent_scale,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            kumaraswamy_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "reparameterized_squashed_gaussian_mixture":
        return ReparameterizedSquashedGaussianMixtureConfig(
            ent_loss_coef=ent_loss_coef,
            gaussian_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            gaussian_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant in ("bernstein_6", "bernstein_8", "bernstein_12", "bernstein_18"):
        return BernsteinQuantileConfig(
            degree=int(variant.removeprefix("bernstein_")),
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "rqs_4":
        return RationalQuadraticSplineQuantileConfig(
            num_bins=4,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "rqs_6":
        return RationalQuadraticSplineQuantileConfig(
            num_bins=6,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "beta":
        return BetaConfig(
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "predicted_std":
        return PredictedStdConfig(
            base_std=gsde_init_stds[0],
            log_std_net_initialization=make_init_linear_orthogonal(action_net_init_gain),
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    if variant == "gsde":
        return GSDEConfig(
            base_std=gsde_init_stds[0],
            latent_sde_dim=None,
            std_learnable=True,
            full_std=True,
            sde_learn_features=False,
            normalize_latent_sde_by_dim=True,
            log_std_clamp_range=(-20.0, 2.0),
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    if variant == "squashed_diag_gaussian":
        return SquashedDiagGaussianConfig(
            std=gsde_init_stds[0],
            std_learnable=True,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    raise ValueError(f"Unknown continuous action dist variant: {variant}")


def _run_training_with_notification_and_close(
        *,
        env: Any,
        run_name: str,
        run_dir: str | Path,
        total_timesteps: int,
        algorithm: Any,
        run: Callable[[], None],
) -> None:
    try:
        run_with_discord_notification(
            run_name=run_name,
            run_dir=run_dir,
            total_timesteps=total_timesteps,
            algorithm=algorithm,
            run=run,
        )
    finally:
        env.close()


def run_experiment(
        *,
        num_envs: int,
        rollout_steps_per_env: int,
        variant_name: str,
        entrypoint_path: Path,
        virtual_mini_batches: int = 1,
        n_epochs: int = 8,
        continuous_action_dist: ContinuousActionDistVariant = "sign_magnitude_beta",
        policy_variant: PolicyVariant = "mat_qcs",
        mat_add_agent_embeddings: bool = False,
        mat_use_agent_attention: bool = True,
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        mat_qcc_tie_query_context_and_context_self_attention: bool = True,
        act_fn_cls: ActivationFactory = nn.GELU,
        mat_init_gains: MATInitGains = MATInitGains(),
        nop_init_gains: NOPInitGains = NOPInitGains(),
        mat_normalization: MATNormalizationConfig = MATNormalizationConfig(),
        enc_nhead: int = 4,
        dec_nhead: int = 2,
        mat_encoder_transformer_ff_config: FeedForwardConfig | None = None,
        rmat_actor_d_model: int | None = None,
        rmat_actor_transformer_ff_config: FeedForwardConfig | None = None,
        rmat_actor_inter_module_mlp: bool = False,
        r_tmasac_actor_state_critic_input_config: ActorStateCriticInputConfig | None = None,
        tmasac_shared_encoder_num_layers: int | None = None,
        tmasac_recurrent_shared_encoder: bool = False,
        tmasac_actor_encoder_num_layers: int = 2,
        tmasac_critic_encoder_num_layers: int = 2,
        tmasac_nop_latent_source: SACNOPLatentSource | str | None = None,
        tmasac_actor_head_kind: TMASACActorHeadKind | str = TMASACActorHeadKind.INDEPENDENT,
        tmasac_separate_observation_action_encoders: bool = False,
        use_nop: bool = True,
        nop_add_agent_embeddings_transition_model: bool = False,
        nop_skip_first_transition_for_critic: bool = True,
        use_transition_obs: bool = False,
        shuffle_agents: bool = False,
        preserve_inactive_prefix_structure: bool = False,
        mat_decoder_lr_multiplier: float = 0.25,
        include_actor_head_lr_multiplier: bool = False,
        experiment_run_name: str | None = None,
        scenario_name: MJWScenarioName = "wall",
        ccd_iterations: int | None = None,
        scenario_kwargs: dict[str, object] | None = None,
        evaluation_scenario_kwargs: dict[str, object] | None = None,
        compile_env_tensor_operations: bool | None = None,
        env_tensor_operations_compile_mode: str = "default",
        rmat_temporal_model_cls: Any = None,
        rmat_temporal_model_config: Any = None,
        rmat_temporal_residual: bool = False,
        rmat_temporal_layer_norm: bool = False,
        rmat_use_temporal_output_projection: bool = True,
        rmat_experimental_compile_lstm: bool = False,
        total_timesteps: int = 100_000_000,
        load_path: str | Path | None = None,
        additional_timesteps: int | None = None,
        sac_learning_rate: float = 3e-4,
        sac_ent_coef_learning_rate: float | None = 1e-3,
        sac_ent_coef: float | str = "auto_0.05",
        sac_target_entropy: float | str = "auto_0.5",
        sac_actor_action_samples: int = 1,
        sac_target_action_samples: int = 1,
        sac_action_sample_strategy: ActionSampleStrategy = "iid",
        sac_independent_nop_sampling: bool = False,
        sac_batch_size: int | None = None,
        sac_buffer_capacity_per_env: int | None = None,
        sac_recurrent_burn_in_steps: int = 32,
        sac_recurrent_learning_steps: int = 64,
        sac_temporal_state_store_interval: int = 32,
        sac_temporal_state_storage_dtype: torch.dtype | None = None,
        sac_max_truncations_per_segment: int = 1,
        bernoulli_initial_prob: float = 0.8,
        evaluation_num_envs: int = 256,
        evaluation_episodes_per_env: int = 1,
        evaluation_milestones: Sequence[float] = DEFAULT_EVALUATION_MILESTONES,
        evaluation_seed: int = 1_000_000,
        evaluation_recording_episodes: int = 0,
        live_recording_schedule: Mapping[float, int] | None = None,
        disable_connector_actions: bool = False,
        migrate_removed_connector_actions: bool = False,
        transfer_checkpoint: bool = False,
) -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    cuda_idx = _get_requested_cuda_idx()
    _configure_cuda_device(cuda_idx=cuda_idx)

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    if not torch.cuda.is_available():
        raise RuntimeError("MJW requires CUDA.")

    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")
    if rollout_steps_per_env <= 0:
        raise ValueError(f"rollout_steps_per_env must be > 0, got {rollout_steps_per_env}")
    if virtual_mini_batches <= 0:
        raise ValueError(f"virtual_mini_batches must be > 0, got {virtual_mini_batches}")
    if n_epochs <= 0:
        raise ValueError(f"n_epochs must be > 0, got {n_epochs}")
    if total_timesteps <= 0:
        raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
    if additional_timesteps is not None:
        if additional_timesteps <= 0:
            raise ValueError(
                f"additional_timesteps must be > 0, got {additional_timesteps}"
            )
        if load_path is None:
            raise ValueError("additional_timesteps requires load_path")
    if transfer_checkpoint and load_path is None:
        raise ValueError("transfer_checkpoint requires load_path")
    if transfer_checkpoint and migrate_removed_connector_actions:
        raise ValueError(
            "transfer_checkpoint and migrate_removed_connector_actions cannot be combined"
        )
    if evaluation_num_envs <= 0:
        raise ValueError(f"evaluation_num_envs must be > 0, got {evaluation_num_envs}")
    if evaluation_episodes_per_env <= 0:
        raise ValueError(
            f"evaluation_episodes_per_env must be > 0, got {evaluation_episodes_per_env}"
        )
    if evaluation_recording_episodes < 0:
        raise ValueError(
            f"evaluation_recording_episodes must be >= 0, got {evaluation_recording_episodes}"
        )
    evaluation_milestones = tuple(float(milestone) for milestone in evaluation_milestones)
    if any(milestone < 0 or milestone > 100 for milestone in evaluation_milestones):
        raise ValueError(
            f"evaluation_milestones must contain percentages in [0, 100], got {evaluation_milestones}"
        )
    if rmat_actor_d_model is not None and rmat_actor_d_model <= 0:
        raise ValueError(f"rmat_actor_d_model must be > 0, got {rmat_actor_d_model}")
    if tmasac_shared_encoder_num_layers is not None and tmasac_shared_encoder_num_layers <= 0:
        raise ValueError(
            "tmasac_shared_encoder_num_layers must be > 0, got "
            f"{tmasac_shared_encoder_num_layers}"
        )
    if tmasac_actor_encoder_num_layers <= 0:
        raise ValueError(
            "tmasac_actor_encoder_num_layers must be > 0, got "
            f"{tmasac_actor_encoder_num_layers}"
        )
    if tmasac_critic_encoder_num_layers <= 0:
        raise ValueError(
            "tmasac_critic_encoder_num_layers must be > 0, got "
            f"{tmasac_critic_encoder_num_layers}"
        )
    if tmasac_recurrent_shared_encoder:
        if policy_variant != "r_tmasac":
            raise ValueError(
                "tmasac_recurrent_shared_encoder=True requires policy_variant='r_tmasac'."
            )
        if tmasac_shared_encoder_num_layers is None:
            raise ValueError(
                "tmasac_recurrent_shared_encoder=True requires "
                "tmasac_shared_encoder_num_layers."
            )
    resolved_tmasac_nop_latent_source = _resolve_tmasac_nop_latent_source(
        shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
        latent_source=tmasac_nop_latent_source,
    )
    if sac_buffer_capacity_per_env is not None and sac_buffer_capacity_per_env <= 0:
        raise ValueError(
            f"sac_buffer_capacity_per_env must be > 0, got {sac_buffer_capacity_per_env}"
        )
    if sac_batch_size is not None and sac_batch_size <= 0:
        raise ValueError(f"sac_batch_size must be > 0, got {sac_batch_size}")
    validate_action_sampling(
        actor_action_samples=sac_actor_action_samples,
        target_action_samples=sac_target_action_samples,
        action_sample_strategy=sac_action_sample_strategy,
    )

    rollout_samples = num_envs * rollout_steps_per_env
    if rollout_samples % virtual_mini_batches != 0:
        raise ValueError(
            f"Expected rollout_samples divisible by virtual_mini_batches, got "
            f"{rollout_samples=} {virtual_mini_batches=}"
        )
    recurrent_policy = _is_recurrent_policy_variant(policy_variant)
    sac_policy = _is_sac_policy_variant(policy_variant)
    recurrent_sac_policy = policy_variant in {"r_tmasac", "segment_tmasac"}
    sampler_batch_size = num_envs if recurrent_policy else rollout_samples
    if sampler_batch_size % virtual_mini_batches != 0:
        raise ValueError(
            f"Expected sampler_batch_size divisible by virtual_mini_batches, got "
            f"{sampler_batch_size=} {virtual_mini_batches=}"
        )

    episode_length = 512
    rollout_warmup_steps_per_env = episode_length
    save_interval = None

    use_popart = not sac_policy
    popart_beta = 5e-4
    popart_init_sigma = 0.65

    vf_coef = 2.0 if use_popart else 0.5
    world_model_loss_coef = 0.1
    world_model_num_next_steps = 3

    initial_stickiness = 0.25
    final_stickiness = 0.0
    stickiness_anneal_steps = 15_000_000
    gsde_init_stds = [0.25, 0.30]

    compile_policy_modules = True
    policy_compile_mode = "default"
    compile_world_model_modules = True

    sac_learning_starts = max(10_000, rollout_samples * 4)
    resolved_sac_batch_size = rollout_samples if sac_batch_size is None else sac_batch_size
    resolved_sac_buffer_capacity_per_env = (
        max(
            episode_length * 2,
            math.ceil(max(sac_learning_starts, resolved_sac_batch_size) / num_envs),
        )
        if sac_buffer_capacity_per_env is None
        else sac_buffer_capacity_per_env
    )
    sac_gradient_steps = 8
    logging_buffer_size = 20 if sac_policy else 5

    run_id = generate_run_id()

    rollout_device = torch.device("cuda")
    train_device = torch.device("cuda")
    record_device = torch.device("cuda")
    scenario_kwargs = _with_default_scenario_kwargs(scenario_kwargs)
    evaluation_scenario_kwargs = (
        scenario_kwargs
        if evaluation_scenario_kwargs is None
        else _with_default_scenario_kwargs(evaluation_scenario_kwargs)
    )

    logger.info(f"{rollout_device = }")
    logger.info(f"{train_device = }")
    if cuda_idx is not None:
        logger.info(f"CUDA device index requested via --cuda_idx={cuda_idx}")
    mat_decoder_self_attention_mode_metadata = (
        mat_decoder_self_attention_mode.name
        if policy_variant in {"mat_qcs", "r_mat_qcs"}
        else None
    )
    variant_log_message = (
        f"MJW {_scenario_display_name(scenario_name=scenario_name)} - {variant_name}: "
        f"{num_envs} envs x {rollout_steps_per_env} steps/env = {rollout_samples}, "
        f"algorithm={'sac' if sac_policy else 'ppo'}, "
        f"virtual_mini_batches={virtual_mini_batches}, n_epochs={n_epochs}, "
        f"continuous_action_dist={continuous_action_dist}, use_nop={use_nop}, "
        f"use_transition_obs={use_transition_obs}, "
        f"compile_policy_modules={compile_policy_modules}, policy_compile_mode={policy_compile_mode}, "
        f"nop_add_agent_embeddings_transition_model={nop_add_agent_embeddings_transition_model}, "
        f"nop_skip_first_transition_for_critic={nop_skip_first_transition_for_critic}, "
        f"act_fn_cls={activation_factory_name(act_fn_cls)}, "
        f"enc_nhead={enc_nhead}, dec_nhead={dec_nhead}, "
        f"mat_init_gains={mat_init_gains}, nop_init_gains={nop_init_gains}, "
        f"mat_normalization={mat_normalization}, "
        f"mat_add_agent_embeddings={mat_add_agent_embeddings}, "
        f"mat_use_agent_attention={mat_use_agent_attention}, "
        f"mat_decoder_self_attention_mode={mat_decoder_self_attention_mode_metadata}, "
        f"mat_qcc_tie_query_context_and_context_self_attention="
        f"{mat_qcc_tie_query_context_and_context_self_attention}, "
        f"bernoulli_initial_prob={bernoulli_initial_prob}, "
        f"shuffle_agents={shuffle_agents}, "
        f"preserve_inactive_prefix_structure={preserve_inactive_prefix_structure}, "
        f"compile_env_tensor_operations={compile_env_tensor_operations}, "
        f"env_tensor_operations_compile_mode={env_tensor_operations_compile_mode}, "
        f"ccd_iterations={ccd_iterations}, "
        f"scenario_kwargs={scenario_kwargs}, "
        f"rmat_temporal_residual={rmat_temporal_residual}, "
        f"rmat_temporal_layer_norm={rmat_temporal_layer_norm}, "
        f"rmat_use_temporal_output_projection={rmat_use_temporal_output_projection}, "
        f"rmat_actor_d_model={rmat_actor_d_model}, "
        f"rmat_actor_transformer_ff_config={rmat_actor_transformer_ff_config}, "
        f"rmat_actor_inter_module_mlp={rmat_actor_inter_module_mlp}"
    )
    if sac_policy:
        variant_log_message = (
            f"{variant_log_message}, sac_learning_rate={sac_learning_rate}, "
            f"sac_buffer_capacity_per_env={resolved_sac_buffer_capacity_per_env}, "
            f"sac_learning_starts={sac_learning_starts}, sac_batch_size={resolved_sac_batch_size}, "
            f"sac_gradient_steps={sac_gradient_steps}, sac_ent_coef={sac_ent_coef}, "
            f"sac_ent_coef_learning_rate={sac_ent_coef_learning_rate}, "
            f"sac_target_entropy={sac_target_entropy}, "
            f"sac_actor_action_samples={sac_actor_action_samples}, "
            f"sac_target_action_samples={sac_target_action_samples}, "
            f"sac_action_sample_strategy={sac_action_sample_strategy}, "
            f"sac_independent_nop_sampling={sac_independent_nop_sampling}"
        )
        if recurrent_sac_policy:
            variant_log_message = (
                f"{variant_log_message}, sac_recurrent_burn_in_steps={sac_recurrent_burn_in_steps}, "
                f"sac_recurrent_learning_steps={sac_recurrent_learning_steps}, "
                f"sac_temporal_state_store_interval={sac_temporal_state_store_interval}, "
                f"sac_temporal_state_storage_dtype={sac_temporal_state_storage_dtype}"
            )
    if policy_variant != "mat_qcs":
        variant_log_message = f"{variant_log_message}, policy_variant={policy_variant}"
    logger.info(variant_log_message)
    scenario_display_name_text = _scenario_display_name(scenario_name=scenario_name)
    logger.info(
        f"MJW {scenario_display_name_text} training uses one batched GPU env directly; worker-pool vectorization is disabled."
    )
    logger.info(
        f"MJW {scenario_display_name_text} training supports live exact-state recording via the `record` command "
        "(for example: record:{\"episodes\":8,\"parallel\":4})."
    )
    logger.info(
        f"MJW {scenario_display_name_text} env uses per-env first-episode staggering so episode ends are spread across time from startup."
    )
    logger.info("MJW env also settles all worlds once on the initial reset, which increases startup latency.")

    if load_path is not None:
        if Path(load_path).suffix != ".pt":
            logger.error("load_path is missing .pt")
            raise ValueError()
        logger.info(f"{load_path = }")
        run_id = get_run_id_from_checkpoint_path(load_path)
    logger.info(f"{run_id = }")

    if experiment_run_name is None:
        experiment_run_name = _default_experiment_run_name(scenario_name=scenario_name)
    if ccd_iterations is None:
        ccd_iterations = _default_ccd_iterations(scenario_name=scenario_name)

    run_dir = REPO_ROOT / "runs" / experiment_run_name / variant_name / run_id
    save_optimizer = True

    first_episode_lengths = _make_staggered_first_episode_lengths(
        episode_length=episode_length,
        num_envs=num_envs,
    )

    print("Creating MJW vector env...")
    vector_env = make_vector_env(
        episode_length=episode_length,
        num_envs=num_envs,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=True,
        device=rollout_device,
        scenario_name=scenario_name,
        ccd_iterations=ccd_iterations,
        scenario_kwargs=scenario_kwargs,
        compile_env_tensor_operations=compile_env_tensor_operations,
        env_tensor_operations_compile_mode=env_tensor_operations_compile_mode,
    )
    print(f"Created {type(vector_env)} with {num_envs} environments.")

    env_settings = vector_env.get_settings()
    obs_indices: ObsIndices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=int(vector_env.single_observation_space["local_obs"].shape[-1]),
        global_obs_dim=int(vector_env.single_observation_space["global_obs"].shape[-1]),
        hidden_local_vars_dim=int(vector_env.single_observation_space["hidden_local_vars"].shape[-1]),
        hidden_global_vars_dim=int(vector_env.single_observation_space["hidden_global_vars"].shape[-1]),
    )

    gamma = 0.99

    print("Wrapping...")
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        use_popart=use_popart,
        rollout_device=rollout_device,
        normalize_prev_binary_actions=mat_normalization.normalize_prev_binary_actions,
        use_transition_obs=use_transition_obs,
        shuffle_agents=shuffle_agents,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        disable_connector_actions=disable_connector_actions,
    )

    print("Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")
    if env.connectors_dim <= 0 or env.actuators_dim % env.connectors_dim != 0:
        raise ValueError(
            f"Expected actuators_dim divisible by connectors_dim, got {env.actuators_dim=} {env.connectors_dim=}"
        )
    actuators_per_limb = env.actuators_dim // env.connectors_dim
    print(f"actuators_per_limb: {actuators_per_limb}")
    connector_action_histogram_bins = 5 if env.continuous_connector_actions else 2

    enc_d_model = 256
    dec_d_model = 128
    transition_model_d_model = 128

    transition_model_nhead = 2

    print("Initializing Policy...")
    base_policy = _make_base_policy(
        env=env,
        policy_variant=policy_variant,
        enc_d_model=enc_d_model,
        enc_nhead=enc_nhead,
        dec_d_model=dec_d_model,
        dec_nhead=dec_nhead,
        use_popart=use_popart,
        popart_beta=popart_beta,
        popart_init_sigma=popart_init_sigma,
        compile_policy_modules=compile_policy_modules,
        policy_compile_mode=policy_compile_mode,
        continuous_action_dist=continuous_action_dist,
        initial_stickiness=initial_stickiness,
        gsde_init_stds=gsde_init_stds,
        mat_add_agent_embeddings=mat_add_agent_embeddings,
        mat_use_agent_attention=mat_use_agent_attention,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        mat_qcc_tie_query_context_and_context_self_attention=(
            mat_qcc_tie_query_context_and_context_self_attention
        ),
        act_fn_cls=act_fn_cls,
        mat_init_gains=mat_init_gains,
        nop_init_gains=nop_init_gains,
        mat_normalization=mat_normalization,
        use_nop=use_nop,
        nop_add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
        nop_skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
        compile_world_model_modules=compile_world_model_modules,
        world_model_loss_coef=world_model_loss_coef,
        world_model_num_next_steps=world_model_num_next_steps,
        transition_model_d_model=transition_model_d_model,
        transition_model_nhead=transition_model_nhead,
        mat_encoder_transformer_ff_config=mat_encoder_transformer_ff_config,
        rmat_actor_d_model=rmat_actor_d_model,
        rmat_actor_transformer_ff_config=rmat_actor_transformer_ff_config,
        rmat_actor_inter_module_mlp=rmat_actor_inter_module_mlp,
        r_tmasac_actor_state_critic_input_config=r_tmasac_actor_state_critic_input_config,
        tmasac_shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
        tmasac_recurrent_shared_encoder=tmasac_recurrent_shared_encoder,
        tmasac_actor_encoder_num_layers=tmasac_actor_encoder_num_layers,
        tmasac_critic_encoder_num_layers=tmasac_critic_encoder_num_layers,
        tmasac_nop_latent_source=resolved_tmasac_nop_latent_source,
        tmasac_actor_head_kind=tmasac_actor_head_kind,
        tmasac_separate_observation_action_encoders=tmasac_separate_observation_action_encoders,
        obs_indices=obs_indices,
        rmat_temporal_model_cls=rmat_temporal_model_cls,
        rmat_temporal_model_config=rmat_temporal_model_config,
        rmat_temporal_residual=rmat_temporal_residual,
        rmat_temporal_layer_norm=rmat_temporal_layer_norm,
        rmat_use_temporal_output_projection=rmat_use_temporal_output_projection,
        rmat_experimental_compile_lstm=rmat_experimental_compile_lstm,
        assume_agent_mask_is_active_prefix=not shuffle_agents or preserve_inactive_prefix_structure,
        bernoulli_initial_prob=bernoulli_initial_prob,
    )
    policy_local_latent_dim = int(getattr(base_policy, "local_latent_dim", enc_d_model))
    policy = base_policy
    if use_nop and not sac_policy:
        policy = NextObsPredWrapper(
            policy=base_policy,
            world_model_config=NOPWorldModelConfig(
                n_agents=env.n_agents,
                local_latent_dim=policy_local_latent_dim,
                action_dim=env.action_space.total_agent_action_dim,
                world_model_loss_coef=world_model_loss_coef,
                compile_modules=compile_world_model_modules,
                compile_mode=policy_compile_mode,
                act_fn_cls=act_fn_cls,
                wm_pre_transition_init_gain=nop_init_gains.pre_transition,
                transition_model_coembed_init_gain=nop_init_gains.transition_coembed,
                transition_model_transformer_ff_init_gain=nop_init_gains.transition_transformer_ff,
                transition_model_head_init_gain=nop_init_gains.transition_head,
                wm_pre_predictors_init_gain=nop_init_gains.pre_predictors,
                wm_predictor_init_gain=nop_init_gains.predictors,
                transition_model_dropout=0.0,
                wm_pre_transition_dims=[policy_local_latent_dim],
                d_model_transition_model=transition_model_d_model,
                nhead_transition_model=transition_model_nhead,
                num_layers_transition_model=2,
                dim_feedforward_transition_model=transition_model_d_model * 2,
                add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
                transition_model_coembed_hidden_dims=[transition_model_d_model],
                wm_pre_predictors_dims=[transition_model_d_model, transition_model_d_model],
                wm_scalar_predictor_hidden_dims=[],
                wm_angle_predictor_hidden_dims=[],
                wm_rot6d_predictor_hidden_dims=[],
                wm_binary_predictor_hidden_dims=[],
                scalar_loss_fn="smooth_l1",
                next_obs_pred_config=NextObsPredConfig(
                    local_scalar_target_indices=obs_indices.local_scalar_indices,
                    local_angle_target_indices=obs_indices.local_angle_indices,
                    local_rot6d_target_indices=obs_indices.local_rot6d_indices,
                    local_binary_target_indices=obs_indices.local_binary_indices,
                    scalar_loss_weight=1.0,
                    angle_loss_weight=1.0,
                    rot6d_loss_weight=1.0,
                    binary_loss_weight=1.0,
                ),
            ),
        )
    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=actuators_per_limb,
        joint_stds=gsde_init_stds,
    )
    print(policy)
    print(f"learnable_params: {policy.num_parameters():,}")

    print(f"Initializing {'SAC' if sac_policy else 'PPO'} Algorithm...")

    parameter_lr_multipliers: dict[str, float] = {}
    algorithm: PPO | SAC | RecurrentSAC
    if sac_policy:
        recurrent_sac_kwargs: dict[str, object] = {}
        if recurrent_sac_policy:
            recurrent_sac_kwargs = {
                "burn_in_steps": sac_recurrent_burn_in_steps,
                "learning_steps": sac_recurrent_learning_steps,
                "temporal_state_store_interval": sac_temporal_state_store_interval,
                "temporal_state_storage_dtype": sac_temporal_state_storage_dtype,
                "max_truncations_per_segment": sac_max_truncations_per_segment,
            }
        sac_algorithm_cls = RecurrentSAC if recurrent_sac_policy else SAC
        algorithm = sac_algorithm_cls(
            policy=policy,
            env=env,
            learning_rate=sac_learning_rate,
            buffer_capacity_per_env=resolved_sac_buffer_capacity_per_env,
            learning_starts=sac_learning_starts,
            batch_size=resolved_sac_batch_size,
            rollout_steps_per_iteration=rollout_samples,
            rollout_warmup_steps_per_env=rollout_warmup_steps_per_env,
            gradient_steps=sac_gradient_steps,
            gamma=gamma,
            tau=0.005,
            ent_coef=sac_ent_coef,
            ent_coef_learning_rate=sac_ent_coef_learning_rate,
            target_entropy=sac_target_entropy,
            actor_action_samples=sac_actor_action_samples,
            target_action_samples=sac_target_action_samples,
            action_sample_strategy=sac_action_sample_strategy,
            target_update_interval=1,
            max_grad_norm=2.0,
            independent_nop_sampling=sac_independent_nop_sampling,
            gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
            train_device=train_device,
            rollout_device=rollout_device,
            record_device=record_device,
            replay_storage_device="cuda",
            metrics_action_splitters=[lambda actions: split_actuator_joints(actions, actuators_per_limb)]
            + ([] if disable_connector_actions else [None]),
            **recurrent_sac_kwargs,
        )
    else:
        warm_lr = 1e-4
        warmup_iterations = 200
        cold_lr = warm_lr * 5e-3 if warmup_iterations > 0 else warm_lr
        clip_range = 0.05
        target_kl = 0.002

        auto_lr = AutomaticLearningRate(
            initial_lr=cold_lr,
            max_lr=8e-4,
            updater=make_auto_lr_updater(
                early_stop_epoch_decay_limit=math.ceil(n_epochs * 0.75),
                max_kl_div=target_kl * 1.55,
                warm_scheduler_config=CosineSchedulerConfig(
                    unit=ScheduleUnit.ITERATIONS,
                    duration=warmup_iterations,
                    start_value=cold_lr,
                    final_value=warm_lr,
                )
                if warmup_iterations > 0
                else None,
            ),
        )

        scheduler_manager: SchedulerManager | None = None
        continuous_dist = policy.action_dist.distributions[0]
        if continuous_action_dist == "sticky_sign_magnitude_beta" and isinstance(continuous_dist, StickyActionDist):
            sticky_dist: StickyActionDist = continuous_dist
            scheduler_manager = SchedulerManager(
                [
                    ScheduledHyperParameter(
                        name="act0_stickiness",
                        scheduler=LinearScheduler(
                            unit=ScheduleUnit.TIMESTEPS,
                            duration=stickiness_anneal_steps,
                            start_value=initial_stickiness,
                            final_value=final_stickiness,
                            name="act0_stickiness",
                        ),
                        get_value=lambda: sticky_dist.get_stickiness(),
                        apply=lambda new_value: sticky_dist.set_stickiness(new_value),
                    )
                ]
            )
        elif continuous_action_dist == "sticky_sign_magnitude_beta":
            act0_dist_type = type(continuous_dist) if policy.action_dist.distributions else None
            logger.warning(f"Skipping act0_stickiness scheduler: action dist[0] is {act0_dist_type}")

        parameter_lr_multipliers = _make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            mat_decoder_lr_multiplier=mat_decoder_lr_multiplier,
            include_actor_head_lr_multiplier=include_actor_head_lr_multiplier,
        )
        if recurrent_policy:
            sampler_config = RPPOWMSamplerConfig(
                batch_size=sampler_batch_size,
                sequence_length=rollout_steps_per_env,
                num_next_steps=world_model_num_next_steps,
                compile_wm_window_helper=use_nop,
            )
        elif use_nop:
            sampler_config = PPOWMSamplerConfig(
                batch_size=sampler_batch_size,
                num_next_steps=world_model_num_next_steps,
                compile_wm_window_helper=True,
            )
        else:
            sampler_config = PPOSamplerConfig(batch_size=sampler_batch_size)

        algorithm = PPO(
            policy=policy,
            env=env,
            learning_rate=auto_lr,
            rollout_mode=StepsRolloutMode(rollout_samples),
            rollout_warmup_steps_per_env=rollout_warmup_steps_per_env,
            max_episode_length=episode_length,
            sampler_config=sampler_config,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=clip_range,
            target_kl=target_kl,
            max_grad_norm=2.0,
            gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
            mc_ent_coef=0e-5,
            vf_coef=vf_coef,
            value_loss_fn=nn.MSELoss(reduction="none"),
            train_device=train_device,
            rollout_device=rollout_device,
            record_device=record_device,
            use_popart=use_popart,
            agent_logprob_reduction="sum" if policy_variant == "ppo" else None,
            metrics_action_splitters=[lambda actions: split_actuator_joints(actions, actuators_per_limb)]
            + ([] if disable_connector_actions else [None]),
            scheduler_manager=scheduler_manager,
            virtual_mini_batches=virtual_mini_batches,
            parameter_lr_multipliers=parameter_lr_multipliers,
        )

    if load_path:
        logger.info(f"Loading model from {load_path}")
        checkpoint_load_kwargs: dict[str, object] = {
            "strict_load_state_dict": not (
                migrate_removed_connector_actions or transfer_checkpoint
            ),
        }
        if migrate_removed_connector_actions:
            checkpoint_load_kwargs["policy_state_dict_transform"] = (
                migrate_tmasac_removed_connector_action_dims
            )
        elif transfer_checkpoint:
            checkpoint_load_kwargs.update(
                policy_state_dict_transform=retain_shape_compatible_policy_state,
                restore_optimizer_state=False,
                restore_env_state=False,
            )
        algorithm.load(
            load_path,
            recover_best_return_ema=False,
            **checkpoint_load_kwargs,
        )

    training_start_timesteps = int(algorithm.n_total_timesteps)
    if additional_timesteps is not None:
        total_timesteps = training_start_timesteps + additional_timesteps
        logger.info(
            "Continuing training for "
            f"{additional_timesteps} transitions ({training_start_timesteps} -> {total_timesteps})."
        )

    scheduled_recording_hook = install_scheduled_recordings(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        start_timesteps=training_start_timesteps,
        schedule=(
            DEFAULT_LIVE_RECORDING_SCHEDULE
            if live_recording_schedule is None
            else live_recording_schedule
        ),
    )

    def make_evaluation_env() -> Any:
        evaluation_vector_env = make_vector_env(
            episode_length=episode_length,
            num_envs=evaluation_num_envs,
            first_episode_lengths=None,
            settle_initial_reset=True,
            device=rollout_device,
            scenario_name=scenario_name,
            ccd_iterations=ccd_iterations,
            scenario_kwargs=evaluation_scenario_kwargs,
            compile_env_tensor_operations=compile_env_tensor_operations,
            env_tensor_operations_compile_mode=env_tensor_operations_compile_mode,
        )
        return wrap_vec_env(
            vector_env=evaluation_vector_env,
            obs_indices=obs_indices,
            gamma=gamma,
            use_popart=use_popart,
            rollout_device=rollout_device,
            normalize_prev_binary_actions=mat_normalization.normalize_prev_binary_actions,
            use_transition_obs=use_transition_obs,
            shuffle_agents=shuffle_agents,
            preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
            disable_connector_actions=disable_connector_actions,
        )

    evaluation_runner = FrozenEvaluationRunner(
        make_env=make_evaluation_env,
        training_env=env,
        policy=policy,
        episodes_per_env=evaluation_episodes_per_env,
        seed=evaluation_seed,
        recording_config=EvaluationRecordingConfig(num_episodes=evaluation_recording_episodes),
        video_folder=run_dir / "videos" / "eval",
    )
    evaluation_metrics_logger = MetricsLogger(
        log_dir=run_dir,
        filename="eval_log.csv",
        console_keys=[
            ("timesteps", None),
            ("eval_milestone_pct", ".1f", "eval_pct"),
            ("eval_stochastic_ep_rew", SummaryStatisticsFormat(mean=" .2f", std=".2f", n="1")),
            ("eval_deterministic_ep_rew", SummaryStatisticsFormat(mean=" .2f", std=".2f", n="1")),
            ("eval_stochastic_success_rate", "5.2f", "eval_stochastic_success_pct"),
            ("eval_deterministic_success_rate", "5.2f", "eval_deterministic_success_pct"),
            ("eval_duration", ".1f", "eval_seconds"),
        ],
        buffer_size=1,
    )
    scheduled_evaluation_hook = ScheduledEvaluationHook(
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        start_timesteps=training_start_timesteps,
        milestones=evaluation_milestones,
        runner=evaluation_runner,
        metrics_logger=evaluation_metrics_logger,
    )

    print("Starting training...")
    logging_console_keys: list[
        tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]
    ] = [
        ("iteration", "5", "it"),
        ("timesteps", "8", "steps"),
        ("total_updates", "6", "tot_upd"),
    ]
    if sac_policy:
        logging_console_keys.extend(
            [
                ("rollout_act0_j0", SummaryStatisticsFormat(histogram=11), "roll0_j0"),
                ("rollout_act0_j1", SummaryStatisticsFormat(histogram=11), "roll0_j1"),
                ("replay_act0_j0", SummaryStatisticsFormat(histogram=11), "rep0_j0"),
                ("replay_act0_j1", SummaryStatisticsFormat(histogram=11), "rep0_j1"),
                ("updates", "3", "upd"),
                ("critic_loss", SummaryStatisticsFormat(mean=".3f")),
                ("actor_loss", SummaryStatisticsFormat(mean=".3f")),
                ("ent_coef", SummaryStatisticsFormat(mean=".3f")),
                ("entropy", SummaryStatisticsFormat(mean=".3f"), "ent"),
                ("target_entropy", SummaryStatisticsFormat(mean=".3f"), "target_ent"),
                ("log_prob", SummaryStatisticsFormat(mean=".3f")),
                ("q_pi", SummaryStatisticsFormat(mean=".3f")),
                ("target_q", SummaryStatisticsFormat(mean=".3f")),
                ("replay_size", "8"),
                ("random_actions", None, "rnd"),
            ]
        )
        if not disable_connector_actions:
            logging_console_keys.extend(
                [
                    ("rollout_act1", SummaryStatisticsFormat(histogram=connector_action_histogram_bins), "roll1"),
                    ("replay_act1", SummaryStatisticsFormat(histogram=connector_action_histogram_bins), "rep1"),
                ]
            )
        if use_nop:
            logging_console_keys.append(("critic_nop_loss_scaled", None, "critic_nop"))
    else:
        logging_console_keys.extend(
            (f"act0_j{i}", SummaryStatisticsFormat(histogram=11)) for i in range(actuators_per_limb)
        )
        if continuous_action_dist in {"predicted_std", "gsde", "squashed_diag_gaussian"}:
            logging_console_keys.extend(
                (
                    f"std0_j{i}",
                    SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".3f", max_value=".3f"),
                )
                for i in range(actuators_per_limb)
            )
        if not disable_connector_actions:
            logging_console_keys.append(
                ("act1", SummaryStatisticsFormat(histogram=connector_action_histogram_bins))
            )
        logging_console_keys.extend(
            [
                ("updates", "3", "upd"),
                ("approx_kl", SummaryStatisticsFormat(mean=".2e", std=".2e")),
                ("clip_frac", None),
                ("ratio", SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".1e", max_value=".3f")),
            ]
        )
        if use_nop:
            logging_console_keys.append(("wm_loss_scaled", None, "wm_loss"))
        logging_console_keys.extend(
            [
                ("val_loss_scaled", None, "val_loss"),
                ("expl_var", SummaryStatisticsFormat(mean=".3f", std=".3f")),
                ("popart_mu", ".3f", "pa_mu"),
                ("popart_sigma", ".3f", "pa_sigma"),
            ]
        )
    logging_console_keys.extend(
        [
            ("ep_rew", SummaryStatisticsFormat(mean=" .2f", std=".2f", max_value=" .2f", n="1")),
            ("ep_rew_ema", " .3f"),
            ("best_ep_rew_ema", " .3f", "best_ema"),
            ("ep_success_rate_ema", "5.2f", "success_pct_ema"),
            ("fps", None),
        ]
    )

    extra_run_metadata = {
        "load_path": load_path,
        "env_settings": env_settings,
        "script": entrypoint_path.read_text(encoding="utf-8"),
        "shared_experiment_script": Path(__file__).read_text(encoding="utf-8"),
        "script_scenario_presets": Path(mjw_scenario_presets.__file__).read_text(encoding="utf-8"),
        "backend": "mjw_env",
        "algorithm_variant": "sac" if sac_policy else "ppo",
        "policy_variant": policy_variant,
        "recording_enabled": "live_mjw_exact_state",
        "evaluation_num_envs": evaluation_num_envs,
        "evaluation_episodes_per_env": evaluation_episodes_per_env,
        "evaluation_milestones": list(evaluation_milestones),
        "evaluation_seed": evaluation_seed,
        "evaluation_action_modes": ["stochastic", "deterministic"],
        "evaluation_recording_episodes": evaluation_recording_episodes,
        "rollout_samples": rollout_samples,
        "rollout_steps_per_env": rollout_steps_per_env,
        "total_timesteps": total_timesteps,
        "training_start_timesteps": training_start_timesteps,
        "additional_timesteps": additional_timesteps,
        "transfer_checkpoint": transfer_checkpoint,
        "disable_connector_actions": disable_connector_actions,
        "migrate_removed_connector_actions": migrate_removed_connector_actions,
        "evaluation_scenario_kwargs": evaluation_scenario_kwargs,
        "logging_buffer_size": logging_buffer_size,
        "sampler_batch_size": sampler_batch_size,
        "recurrent_policy": recurrent_policy,
        "rollout_warmup_steps_per_env": rollout_warmup_steps_per_env,
        "virtual_mini_batches": virtual_mini_batches,
        "n_epochs": n_epochs,
        "num_envs": num_envs,
        "variant_name": variant_name,
        "continuous_action_dist": continuous_action_dist,
        "use_nop": use_nop,
        "world_model_num_next_steps": world_model_num_next_steps,
        "sac_independent_nop_sampling": sac_independent_nop_sampling,
        "use_transition_obs": use_transition_obs,
        "nop_add_agent_embeddings_transition_model": nop_add_agent_embeddings_transition_model,
        "nop_skip_first_transition_for_critic": nop_skip_first_transition_for_critic,
        "act_fn_cls": activation_factory_name(act_fn_cls),
        "enc_nhead": enc_nhead,
        "dec_nhead": dec_nhead,
        "mat_init_gains": asdict(mat_init_gains),
        "nop_init_gains": asdict(nop_init_gains),
        "mat_normalization": asdict(mat_normalization),
        "mat_encoder_transformer_ff_config": (
            None
            if mat_encoder_transformer_ff_config is None
            else serialize_dataclass(mat_encoder_transformer_ff_config)
        ),
        "rmat_actor_d_model": rmat_actor_d_model,
        "rmat_actor_transformer_ff_config": (
            None
            if rmat_actor_transformer_ff_config is None
            else serialize_dataclass(rmat_actor_transformer_ff_config)
        ),
        "rmat_actor_inter_module_mlp": rmat_actor_inter_module_mlp,
        "tmasac_shared_encoder_num_layers": tmasac_shared_encoder_num_layers,
        "tmasac_recurrent_shared_encoder": tmasac_recurrent_shared_encoder,
        "tmasac_actor_encoder_num_layers": tmasac_actor_encoder_num_layers,
        "tmasac_critic_encoder_num_layers": tmasac_critic_encoder_num_layers,
        "tmasac_nop_latent_source": (
            resolved_tmasac_nop_latent_source.value
            if isinstance(resolved_tmasac_nop_latent_source, SACNOPLatentSource)
            else resolved_tmasac_nop_latent_source
        ),
        "tmasac_actor_head_kind": (
            tmasac_actor_head_kind.value
            if isinstance(tmasac_actor_head_kind, TMASACActorHeadKind)
            else tmasac_actor_head_kind
        ),
        "mat_add_agent_embeddings": mat_add_agent_embeddings,
        "mat_use_agent_attention": mat_use_agent_attention,
        "mat_decoder_self_attention_mode": mat_decoder_self_attention_mode_metadata,
        "mat_qcc_tie_query_context_and_context_self_attention": (
            mat_qcc_tie_query_context_and_context_self_attention
        ),
        "bernoulli_initial_prob": bernoulli_initial_prob,
        "mat_decoder_lr_multiplier": mat_decoder_lr_multiplier,
        "include_actor_head_lr_multiplier": include_actor_head_lr_multiplier,
        "parameter_lr_multipliers": parameter_lr_multipliers,
        "shuffle_agents": shuffle_agents,
        "preserve_inactive_prefix_structure": preserve_inactive_prefix_structure,
        "experiment_run_name": experiment_run_name,
        "scenario_name": scenario_name,
        "ccd_iterations": ccd_iterations,
        "settle_initial_reset": True,
        "scenario_kwargs": scenario_kwargs,
        "rmat_temporal_model_cls": _metadata_name(rmat_temporal_model_cls),
        "rmat_temporal_model_config": _metadata_name(rmat_temporal_model_config),
        "rmat_temporal_residual": rmat_temporal_residual,
        "rmat_temporal_layer_norm": rmat_temporal_layer_norm,
        "rmat_use_temporal_output_projection": rmat_use_temporal_output_projection,
    }
    if sac_policy:
        extra_run_metadata.update(
            {
                "sac_learning_rate": sac_learning_rate,
                "sac_buffer_capacity_per_env": resolved_sac_buffer_capacity_per_env,
                "sac_learning_starts": sac_learning_starts,
                "sac_batch_size": resolved_sac_batch_size,
                "sac_gradient_steps": sac_gradient_steps,
                "sac_ent_coef": sac_ent_coef,
                "sac_ent_coef_learning_rate": sac_ent_coef_learning_rate,
                "sac_target_entropy": sac_target_entropy,
                "sac_actor_action_samples": sac_actor_action_samples,
                "sac_target_action_samples": sac_target_action_samples,
                "sac_action_sample_strategy": sac_action_sample_strategy,
            }
        )
        if recurrent_sac_policy:
            extra_run_metadata.update(
                {
                    "sac_recurrent_burn_in_steps": sac_recurrent_burn_in_steps,
                    "sac_recurrent_learning_steps": sac_recurrent_learning_steps,
                    "sac_max_truncations_per_segment": sac_max_truncations_per_segment,
                    "sac_temporal_state_store_interval": sac_temporal_state_store_interval,
                    "sac_temporal_state_storage_dtype": str(sac_temporal_state_storage_dtype),
                }
            )
    try:
        _run_training_with_notification_and_close(
            env=env,
            run_name=f"{experiment_run_name}/{variant_name}/{run_id}",
            run_dir=str(run_dir),
            total_timesteps=total_timesteps,
            algorithm=algorithm,
            run=lambda: algorithm.learn(
                max_total_timesteps=total_timesteps,
                run_dir=str(run_dir),
                log_interval=1,
                logging_buffer_size=logging_buffer_size,
                save_interval=save_interval,
                save_optimizer=save_optimizer,
                best_rotation_n=1,
                extra_run_metadata=extra_run_metadata,
                logging_console_keys=logging_console_keys,
                post_iteration_hooks=[scheduled_recording_hook, scheduled_evaluation_hook],
            ),
        )
    finally:
        scheduled_evaluation_hook.close()

    print("Training Finished.")


def _make_mat_parameter_lr_multipliers(
        *,
        policy_variant: PolicyVariant,
        mat_decoder_lr_multiplier: float,
        include_actor_head_lr_multiplier: bool = False,
) -> dict[str, float]:
    if mat_decoder_lr_multiplier == 1.0:
        return {}

    def make_prefix_multipliers(*prefixes: str, include_actor_head_input_norm: bool = False) -> dict[str, float]:
        effective_prefixes = list(prefixes)
        if include_actor_head_lr_multiplier:
            if include_actor_head_input_norm:
                effective_prefixes.append("actor_head_input_norm")
            effective_prefixes.append("actor_head")
        return {prefix: mat_decoder_lr_multiplier for prefix in effective_prefixes}

    if policy_variant == "mat_orig":
        return {
            "decoder": mat_decoder_lr_multiplier,
            "encoder_decoder_projection": mat_decoder_lr_multiplier,
        }

    if policy_variant in {"mat_qcs", "mat_qcc", "r_mat_qcs", "r_mat_qcc"}:
        return make_prefix_multipliers(
            "decoder",
            "query_input_norm",
            "query_encoder",
            "query_token_norm",
            "context_input_norm",
            "context_encoder",
            "context_token_norm",
            "memory_input_norm",
            "memory_encoder",
            "memory_token_norm",
            "agent_embeddings_decoder",
            include_actor_head_input_norm=True,
        )

    if policy_variant == "mat_qcx":
        return make_prefix_multipliers(
            "decoder",
            "action_input_norm",
            "action_encoder",
            "action_token_norm",
            "memory_input_norm",
            "memory_encoder",
            "memory_token_norm",
            include_actor_head_input_norm=True,
        )

    if policy_variant == "r_mat_qcx":
        return make_prefix_multipliers(
            "decoder",
            "action_input_norm",
            "action_encoder",
            "action_token_norm",
            "memory_input_norm",
            "memory_encoder",
            "memory_token_norm",
            include_actor_head_input_norm=True,
        )

    if policy_variant in {"mat_dec", "mat_ind", "r_mat_ind"} and include_actor_head_lr_multiplier:
        return make_prefix_multipliers()

    logger.warning(f"Ignoring decoder LR multiplier for policy_variant={policy_variant!r}")
    return {}


def _make_sac_nop_config(
        *,
        use_nop: bool,
        obs_indices: ObsIndices | None,
        source_latent_dim: int,
        nop_init_gains: NOPInitGains,
        nop_add_agent_embeddings_transition_model: bool,
        nop_skip_first_transition_for_critic: bool,
        latent_source: SACNOPLatentSource | str,
        compile_world_model_modules: bool,
        policy_compile_mode: str,
        world_model_loss_coef: float,
        num_next_steps: int,
        transition_model_d_model: int,
        transition_model_nhead: int,
        act_fn_cls: ActivationFactory,
) -> SACNOPConfig:
    if not use_nop:
        return SACNOPConfig(enabled=False, num_next_steps=num_next_steps)
    if obs_indices is None:
        raise ValueError("obs_indices is required when building TMASAC with NOP enabled.")

    return SACNOPConfig(
        enabled=True,
        num_next_steps=num_next_steps,
        latent_source=latent_source,
        skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
        nop_loss_coef=world_model_loss_coef,
        compile_modules=compile_world_model_modules,
        compile_mode=policy_compile_mode,
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
        transition_model_add_agent_embeddings=nop_add_agent_embeddings_transition_model,
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
            scalar_loss_weight=1.0,
            angle_loss_weight=1.0,
            rot6d_loss_weight=1.0,
            binary_loss_weight=1.0,
        ),
    )


def _resolve_tmasac_nop_latent_source(
        *,
        shared_encoder_num_layers: int | None,
        latent_source: SACNOPLatentSource | str | None,
) -> SACNOPLatentSource | str:
    if latent_source is not None:
        return latent_source
    if shared_encoder_num_layers is not None:
        return SACNOPLatentSource.SHARED_ENCODER
    return SACNOPLatentSource.CRITIC


def _make_base_policy(
        *,
        env: Any,
        policy_variant: PolicyVariant,
        enc_d_model: int,
        enc_nhead: int,
        dec_d_model: int,
        dec_nhead: int,
        use_popart: bool,
        popart_beta: float,
        popart_init_sigma: float,
        compile_policy_modules: bool,
        policy_compile_mode: str,
        continuous_action_dist: ContinuousActionDistVariant,
        initial_stickiness: float,
        gsde_init_stds: list[float],
        mat_add_agent_embeddings: bool,
        mat_use_agent_attention: bool,
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode,
        act_fn_cls: ActivationFactory,
        mat_init_gains: MATInitGains,
        mat_normalization: MATNormalizationConfig,
        nop_init_gains: NOPInitGains = NOPInitGains(),
        use_nop: bool = False,
        nop_add_agent_embeddings_transition_model: bool = False,
        nop_skip_first_transition_for_critic: bool = True,
        compile_world_model_modules: bool = False,
        world_model_loss_coef: float = 0.1,
        world_model_num_next_steps: int = 4,
        transition_model_d_model: int = 128,
        transition_model_nhead: int = 2,
        mat_encoder_transformer_ff_config: FeedForwardConfig | None = None,
        rmat_actor_d_model: int | None = None,
        rmat_actor_transformer_ff_config: FeedForwardConfig | None = None,
        rmat_actor_inter_module_mlp: bool = False,
        r_tmasac_actor_state_critic_input_config: ActorStateCriticInputConfig | None = None,
        tmasac_shared_encoder_num_layers: int | None = None,
        tmasac_recurrent_shared_encoder: bool = False,
        tmasac_actor_encoder_num_layers: int = 2,
        tmasac_critic_encoder_num_layers: int = 2,
        tmasac_nop_latent_source: SACNOPLatentSource | str | None = None,
        tmasac_actor_head_kind: TMASACActorHeadKind | str = TMASACActorHeadKind.INDEPENDENT,
        tmasac_separate_observation_action_encoders: bool = False,
        obs_indices: ObsIndices | None = None,
        mat_qcc_tie_query_context_and_context_self_attention: bool = True,
        rmat_temporal_model_cls: Any = None,
        rmat_temporal_model_config: Any = None,
        rmat_temporal_residual: bool = False,
        rmat_temporal_layer_norm: bool = False,
        rmat_use_temporal_output_projection: bool = True,
        rmat_experimental_compile_lstm: bool = False,
        assume_agent_mask_is_active_prefix: bool = False,
        bernoulli_initial_prob: float = 0.8,
) -> (
        PPOPolicy
        | MAPPOPolicy
        | MATQCSPolicy
        | MATQCCPolicy
        | MATQCXPolicy
        | MATDecPolicy
        | MATIndPolicy
        | MATOrigPolicy
        | RMATQCSPolicy
        | RMATQCCPolicy
        | RMATQCXPolicy
        | RMATIndPolicy
        | TMASACPolicy
        | RecurrentTMASACPolicy
        | SegmentTMASACPolicy
):
    sac_policy_variant = _is_sac_policy_variant(policy_variant)
    continuous_config = make_continuous_config(
        variant=continuous_action_dist,
        initial_stickiness=initial_stickiness,
        gsde_init_stds=gsde_init_stds,
        action_net_init_gain=mat_init_gains.action_net,
        ent_loss_coef=0.0 if sac_policy_variant else 1e-3,
        rsmk_kumaraswamy_ent_scale=(
            0.0
            if sac_policy_variant and continuous_action_dist in (
                "reparameterized_sign_magnitude_kumaraswamy",
                "gumbel_softmax_sign_magnitude_kumaraswamy",
            )
            else 0.75
        ),
    )
    bernoulli_config = BernoulliConfig(
        initial_prob=bernoulli_initial_prob,
        ent_loss_coef=1e-3,
        ent_loss_config=EntropyLossConfig(
            agent_actions_reduction=AgentActionsReduction.SUM,
            metrics_reduction=AgentActionsReduction.MEAN,
        ),
    )
    popart_config = PopArtConfig(
        beta=popart_beta,
        init_sigma=popart_init_sigma,
    )
    mat_encoder_config = MATEncoderConfig(
        d_model=enc_d_model,
        nhead=enc_nhead,
        num_layers=2,
        dim_feedforward=enc_d_model * 2,
        act_fn_cls=act_fn_cls,
        add_agent_embeddings=mat_add_agent_embeddings,
        linear_init_gain=mat_init_gains.obs_encoder,
        linear_projection_init_gain=mat_init_gains.obs_encoder_projection,
        transformer_ff_init_gain=mat_init_gains.encoder_transformer_ff,
        transformer_ff_config=mat_encoder_transformer_ff_config,
        local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
        use_agent_attention=mat_use_agent_attention,
    )
    rmat_encoder_config = RMATEncoderConfig(
        d_model=enc_d_model,
        nhead=enc_nhead,
        num_layers=2,
        dim_feedforward=enc_d_model * 2,
        act_fn_cls=act_fn_cls,
        add_agent_embeddings=mat_add_agent_embeddings,
        linear_init_gain=mat_init_gains.obs_encoder,
        linear_projection_init_gain=mat_init_gains.obs_encoder_projection,
        transformer_ff_init_gain=mat_init_gains.encoder_transformer_ff,
        transformer_ff_config=mat_encoder_transformer_ff_config,
        local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
        use_agent_attention=mat_use_agent_attention,
        temporal_residual=rmat_temporal_residual,
        temporal_layer_norm=rmat_temporal_layer_norm,
        use_temporal_output_projection=rmat_use_temporal_output_projection,
        **({} if rmat_temporal_model_cls is None else {"temporal_model_cls": rmat_temporal_model_cls}),
        **({} if rmat_temporal_model_config is None else {"temporal_model_config": rmat_temporal_model_config}),
    )
    resolved_rmat_actor_d_model = enc_d_model if rmat_actor_d_model is None else rmat_actor_d_model
    resolved_rmat_actor_ff_config = (
        mat_encoder_transformer_ff_config
        if rmat_actor_transformer_ff_config is None
        else rmat_actor_transformer_ff_config
    )
    rmat_actor_encoder_config = replace(
        rmat_encoder_config,
        d_model=resolved_rmat_actor_d_model,
        dim_feedforward=resolved_rmat_actor_d_model * 2,
        transformer_ff_config=resolved_rmat_actor_ff_config,
        local_obs_encoder_config=MLPConfig(
            hidden_dims=[resolved_rmat_actor_d_model, resolved_rmat_actor_d_model]
        ),
        global_obs_encoder_config=MLPConfig(hidden_dims=[resolved_rmat_actor_d_model]),
        inter_module_mlp=rmat_actor_inter_module_mlp,
    )

    if sac_policy_variant:
        if use_popart:
            raise ValueError("TMASACPolicy does not support PopArt; call with use_popart=False.")
        resolved_tmasac_nop_latent_source = _resolve_tmasac_nop_latent_source(
            shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
            latent_source=tmasac_nop_latent_source,
        )
        tmasac_config_kwargs = {
            "actor_encoder_config": (
                replace(
                    (
                        rmat_actor_encoder_config
                        if policy_variant == "r_tmasac" and not tmasac_recurrent_shared_encoder
                        else mat_encoder_config
                    ),
                    num_layers=tmasac_actor_encoder_num_layers,
                )
            ),
            "critic_encoder_config": replace(
                mat_encoder_config,
                num_layers=tmasac_critic_encoder_num_layers,
            ),
            "shared_encoder_config": (
                None
                if tmasac_shared_encoder_num_layers is None
                else replace(
                    (
                        rmat_actor_encoder_config
                        if tmasac_recurrent_shared_encoder
                        else mat_encoder_config
                    ),
                    num_layers=tmasac_shared_encoder_num_layers,
                )
            ),
            "actor_head_config": TMASACActorHeadConfig(
                kind=tmasac_actor_head_kind,
                hidden_dims=[dec_d_model],
                normalize_input=mat_normalization.normalize_actor_head_input,
                init_gain=mat_init_gains.actor_head,
                qcx_decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    context_encoder_hidden_dims=[dec_d_model],
                    action_encoder_dims=[dec_d_model, dec_d_model],
                    memory_dims=None,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_action_input=mat_normalization.normalize_action_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_action_tokens=mat_normalization.normalize_action_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
            ),
            "critic_config": TMASACCriticConfig(
                n_local_projection_hidden_layers=2,
                n_value_regressor_hidden_layers=1,
                separate_observation_action_encoders=tmasac_separate_observation_action_encoders,
                use_popart=False,
                popart_config=popart_config,
                local_projection_init_gain=mat_init_gains.critic_local_projection,
                value_regressor_init_gain=mat_init_gains.critic_value_regressor,
                value_head_init_gain=mat_init_gains.critic_value_head,
            ),
            "dropout": 0.0,
            "act_fn_cls": act_fn_cls,
            "continuous_config": continuous_config,
            "max_agents": 20,
            "nop_config": _make_sac_nop_config(
                use_nop=use_nop,
                obs_indices=obs_indices,
                source_latent_dim=enc_d_model,
                nop_init_gains=nop_init_gains,
                nop_add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
                nop_skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
                latent_source=resolved_tmasac_nop_latent_source,
                compile_world_model_modules=compile_world_model_modules,
                policy_compile_mode=policy_compile_mode,
                world_model_loss_coef=world_model_loss_coef,
                num_next_steps=world_model_num_next_steps,
                transition_model_d_model=transition_model_d_model,
                transition_model_nhead=transition_model_nhead,
                act_fn_cls=act_fn_cls,
            ),
            "compile_modules": compile_policy_modules,
            "compile_mode": policy_compile_mode,
            "action_net_init_gain": mat_init_gains.action_net,
        }
        if policy_variant == "r_tmasac":
            return RecurrentTMASACPolicy(
                env=env,
                config=RecurrentTMASACPolicyConfig(
                    recurrent_critic=False,
                    recurrent_shared_encoder=tmasac_recurrent_shared_encoder,
                    actor_state_critic_input_config=r_tmasac_actor_state_critic_input_config,
                    experimental_compile_lstm=rmat_experimental_compile_lstm,
                    **tmasac_config_kwargs,
                ),
            )
        if policy_variant == "segment_tmasac":
            return SegmentTMASACPolicy(
                env=env,
                config=TMASACPolicyConfig(**tmasac_config_kwargs),
            )
        return TMASACPolicy(
            env=env,
            config=TMASACPolicyConfig(**tmasac_config_kwargs),
        )

    if policy_variant == "ppo":
        return PPOPolicy(
            env=env,
            config=PPOPolicyConfig(
                actor_config=PPOActorConfig(
                    hidden_dims=[512, 384, 256, 256],
                    shared_encoder_latent_dim_per_agent=192,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim_per_agent=96,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=PPOCriticConfig(
                    hidden_dims=[256, 256],
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    if policy_variant == "ppo_small":
        return PPOPolicy(
            env=env,
            config=PPOPolicyConfig(
                actor_config=PPOActorConfig(
                    hidden_dims=[384, 320, 256, 192],
                    shared_encoder_latent_dim_per_agent=160,
                    actor_head_hidden_dims=[128, 96],
                    latent_pi_dim_per_agent=64,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=PPOCriticConfig(
                    hidden_dims=[160, 160],
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    if policy_variant == "mappo":
        return MAPPOPolicy(
            env=env,
            config=MAPPOPolicyConfig(
                actor_config=MAPPOActorConfig(
                    hidden_dims=[768, 512, 512, 384],
                    shared_encoder_latent_dim=256,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim=128,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=MAPPOCriticConfig(
                    deep_set_config=DeepSetCriticConfig(
                        local_projection_hidden_dims=[256, 128],
                        value_regressor_hidden_dims=[256, 128],
                    ),
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    if policy_variant == "mappo_small":
        return MAPPOPolicy(
            env=env,
            config=MAPPOPolicyConfig(
                actor_config=MAPPOActorConfig(
                    hidden_dims=[512, 512, 384, 256],
                    shared_encoder_latent_dim=256,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim=112,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=MAPPOCriticConfig(
                    deep_set_config=DeepSetCriticConfig(
                        local_projection_hidden_dims=[224, 112],
                        value_regressor_hidden_dims=[192, 128],
                    ),
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    mat_qcs_critic_config = MATQCSCriticConfig(
        n_local_projection_hidden_layers=2,
        n_value_regressor_hidden_layers=1,
        use_popart=use_popart,
        popart_config=popart_config,
        local_projection_init_gain=mat_init_gains.critic_local_projection,
        value_regressor_init_gain=mat_init_gains.critic_value_regressor,
        value_head_init_gain=mat_init_gains.critic_value_head,
    )

    if policy_variant == "mat_qcs":
        return MATQCSPolicy(
            env=env,
            config=MATQCSPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCSDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    self_attention_mode=mat_decoder_self_attention_mode,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_qcc":
        return MATQCCPolicy(
            env=env,
            config=MATQCCPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCCDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                    tie_query_context_and_context_self_attention=(
                        mat_qcc_tie_query_context_and_context_self_attention
                    ),
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "r_mat_qcs":
        return RMATQCSPolicy(
            env=env,
            config=RMATQCSPolicyConfig(
                encoder_config=rmat_encoder_config,
                decoder_config=MATQCSDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    self_attention_mode=mat_decoder_self_attention_mode,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "r_mat_qcc":
        return RMATQCCPolicy(
            env=env,
            config=RMATQCCPolicyConfig(
                encoder_config=rmat_encoder_config,
                decoder_config=MATQCCDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                    tie_query_context_and_context_self_attention=(
                        mat_qcc_tie_query_context_and_context_self_attention
                    ),
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_qcx":
        return MATQCXPolicy(
            env=env,
            config=MATQCXPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
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
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "r_mat_qcx":
        return RMATQCXPolicy(
            env=env,
            config=RMATQCXPolicyConfig(
                encoder_config=rmat_encoder_config,
                decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
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
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_dec":
        return MATDecPolicy(
            env=env,
            config=MATDecPolicyConfig(
                encoder_config=mat_encoder_config,
                critic_config=mat_qcs_critic_config,
                actor_head_hidden_dims=[dec_d_model, dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_ind":
        return MATIndPolicy(
            env=env,
            config=MATIndPolicyConfig(
                encoder_config=mat_encoder_config,
                critic_config=mat_qcs_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "r_mat_ind":
        return RMATIndPolicy(
            env=env,
            config=RMATIndPolicyConfig(
                encoder_config=rmat_encoder_config,
                critic_config=mat_qcs_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_orig":
        return MATOrigPolicy(
            env=env,
            config=MATOrigPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATOrigDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    latent_pi_dim=dec_d_model,
                ),
                critic_config=MATOrigCriticConfig(
                    use_popart=use_popart,
                    popart_config=popart_config,
                    pool_mode="mean",
                    value_head_hidden_dims=[enc_d_model, enc_d_model],
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                encoder_decoder_projection_init_gain=(
                    mat_init_gains.decoder_token_encoder
                    if mat_init_gains.decoder_token_encoder_projection is None
                    else mat_init_gains.decoder_token_encoder_projection
                ),
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    raise ValueError(f"Unknown policy_variant: {policy_variant}")
