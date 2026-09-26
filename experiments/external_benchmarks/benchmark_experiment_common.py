from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from loguru import logger
from torch import nn

from experiments.transformer_policy_common import (
    MATInitGains,
    MATNormalizationConfig,
    NOPInitGains,
    make_benchmark_transformer_policy,
    make_mat_parameter_lr_multipliers,
)
from swarmbots.external_benchmark_envs import MultiAgentMujocoEnv, VMASVectorEnv
from swarmbots.utils.run_paths import generate_run_id
from swarmbots.learn.algos.ppo.ppo import PPO, AutomaticLearningRate, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import (
    NextObsPredWrapper,
    NOPWorldModelConfig,
)
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.continuous_actions_learn_env_wrapper import (
    ContinuousActionsLearnEnvWrapper,
)
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import (
    TorchFeatureWiseObsNormWrapper,
)
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import (
    TorchNormalizeRewardWrapper,
)
from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import (
    TorchRecordEpisodeStatisticsWrapper,
)
from swarmbots.learn.env_wrappers.worker_pool_async_vector_env import (
    WorkerPoolAsyncVectorEnv,
)
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleUnit
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat

REPO_ROOT = Path(__file__).resolve().parents[2]
BenchmarkSuite = Literal["mamujoco", "vmas"]
BenchmarkPolicyVariant = Literal["mat_qcx", "mat_dec", "mat_ind", "tmasac"]


def run_benchmark_experiment(
    *,
    suite: BenchmarkSuite,
    policy_variant: BenchmarkPolicyVariant,
    use_nop: bool,
    variant_name: str,
    entrypoint_path: Path,
    scenario: str,
    num_envs: int,
    rollout_steps_per_env: int,
    episode_length: int,
    total_timesteps: int,
    experiment_run_name: str,
    experiment_definition_path: Path,
    mamujoco_agent_conf: str = "2x4",
    mamujoco_agent_obsk: int = 1,
    mamujoco_num_workers: int | None = None,
    scenario_kwargs: dict[str, Any] | None = None,
    compile_modules: bool = True,
) -> None:
    if policy_variant not in {"mat_qcx", "mat_dec", "mat_ind", "tmasac"}:
        raise ValueError(f"Unsupported benchmark policy variant: {policy_variant}")
    if min(num_envs, rollout_steps_per_env, episode_length, total_timesteps) <= 0:
        raise ValueError(
            "Environment counts, rollout length, episode length, and total timesteps must be positive."
        )

    rollout_samples = num_envs * rollout_steps_per_env
    is_sac = policy_variant == "tmasac"
    use_popart = not is_sac
    cuda_idx = _requested_cuda_idx()
    if cuda_idx is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda_idx was provided, but CUDA is unavailable.")
        torch.cuda.set_device(cuda_idx)
    rollout_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_device = rollout_device
    record_device = rollout_device
    compile_modules = compile_modules and rollout_device.type == "cuda"
    run_id = generate_run_id()
    run_dir = REPO_ROOT / "runs" / experiment_run_name / variant_name / run_id

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    torch.set_float32_matmul_precision("high")
    logger.info(
        f"{suite} {scenario} - {variant_name}: {num_envs} envs x "
        f"{rollout_steps_per_env} steps/env, use_nop={use_nop}, device={rollout_device}"
    )

    vector_env = _make_vector_env(
        suite=suite,
        scenario=scenario,
        num_envs=num_envs,
        episode_length=episode_length,
        rollout_device=rollout_device,
        mamujoco_agent_conf=mamujoco_agent_conf,
        mamujoco_agent_obsk=mamujoco_agent_obsk,
        mamujoco_num_workers=mamujoco_num_workers,
        scenario_kwargs=scenario_kwargs,
    )
    obs_indices = _make_obs_indices(vector_env)
    env = _wrap_vector_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        rollout_device=rollout_device,
        use_popart=use_popart,
    )

    mat_init_gains = MATInitGains()
    nop_init_gains = NOPInitGains()
    mat_normalization = MATNormalizationConfig()
    enc_d_model = 256
    dec_d_model = 128
    world_model_num_next_steps = 3
    world_model_loss_coef = 0.1
    transition_model_d_model = 128

    base_policy = make_benchmark_transformer_policy(
        env=env,
        policy_variant=policy_variant,
        use_nop=use_nop,
        obs_indices=obs_indices,
        compile_modules=compile_modules,
        mat_init_gains=mat_init_gains,
        nop_init_gains=nop_init_gains,
        mat_normalization=mat_normalization,
        enc_d_model=enc_d_model,
        enc_nhead=4,
        dec_d_model=dec_d_model,
        dec_nhead=2,
        world_model_loss_coef=world_model_loss_coef,
        world_model_num_next_steps=world_model_num_next_steps,
        transition_model_d_model=transition_model_d_model,
    )
    policy = _wrap_ppo_policy_with_nop(
        base_policy=base_policy,
        env=env,
        use_nop=use_nop and not is_sac,
        obs_indices=obs_indices,
        compile_modules=compile_modules,
        local_latent_dim=enc_d_model,
        nop_init_gains=nop_init_gains,
        world_model_loss_coef=world_model_loss_coef,
        world_model_num_next_steps=world_model_num_next_steps,
        transition_model_d_model=transition_model_d_model,
    )
    logger.info(f"Policy parameters: {policy.num_parameters():,}")

    algorithm = _make_algorithm(
        policy=policy,
        env=env,
        policy_variant=policy_variant,
        use_nop=use_nop,
        rollout_samples=rollout_samples,
        rollout_steps_per_env=rollout_steps_per_env,
        episode_length=episode_length,
        world_model_num_next_steps=world_model_num_next_steps,
        rollout_device=rollout_device,
        train_device=train_device,
        record_device=record_device,
    )
    extra_run_metadata = {
        "backend": suite,
        "scenario": scenario,
        "scenario_kwargs": scenario_kwargs,
        "mamujoco_agent_conf": mamujoco_agent_conf if suite == "mamujoco" else None,
        "mamujoco_agent_obsk": mamujoco_agent_obsk if suite == "mamujoco" else None,
        "mamujoco_num_workers": mamujoco_num_workers if suite == "mamujoco" else None,
        "policy_variant": policy_variant,
        "algorithm_variant": "sac" if is_sac else "ppo",
        "use_nop": use_nop,
        "num_envs": num_envs,
        "rollout_steps_per_env": rollout_steps_per_env,
        "rollout_samples": rollout_samples,
        "episode_length": episode_length,
        "total_timesteps": total_timesteps,
        "compile_modules": compile_modules,
        "cuda_idx": cuda_idx,
        "enc_d_model": enc_d_model,
        "dec_d_model": dec_d_model,
        "world_model_num_next_steps": world_model_num_next_steps,
        "world_model_loss_coef": world_model_loss_coef,
        "transition_model_d_model": transition_model_d_model,
        "mat_init_gains": asdict(mat_init_gains),
        "nop_init_gains": asdict(nop_init_gains),
        "script": entrypoint_path.read_text(encoding="utf-8"),
        "scenario_experiment_definition": experiment_definition_path.read_text(
            encoding="utf-8"
        ),
        "shared_experiment_script": Path(__file__).read_text(encoding="utf-8"),
    }
    logging_console_keys = _logging_console_keys(is_sac=is_sac, use_nop=use_nop)
    try:
        algorithm.learn(
            max_total_timesteps=total_timesteps,
            run_dir=run_dir,
            log_interval=1,
            logging_buffer_size=20 if is_sac else 5,
            save_interval=None,
            save_optimizer=True,
            best_rotation_n=1,
            extra_run_metadata=extra_run_metadata,
            logging_console_keys=logging_console_keys,
        )
    finally:
        env.close()


def _requested_cuda_idx() -> int | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_idx", "--cuda-idx", type=int, default=None)
    parsed, _ = parser.parse_known_args()
    if parsed.cuda_idx is not None and parsed.cuda_idx < 0:
        raise ValueError(f"cuda_idx must be >= 0, got {parsed.cuda_idx}")
    return parsed.cuda_idx


def _make_vector_env(
    *,
    suite: BenchmarkSuite,
    scenario: str,
    num_envs: int,
    episode_length: int,
    rollout_device: torch.device,
    mamujoco_agent_conf: str,
    mamujoco_agent_obsk: int,
    mamujoco_num_workers: int | None,
    scenario_kwargs: dict[str, Any] | None,
) -> Any:
    if suite == "vmas":
        return VMASVectorEnv(
            scenario=scenario,
            num_envs=num_envs,
            device=rollout_device,
            episode_length=episode_length,
            scenario_kwargs=scenario_kwargs,
        )
    if suite != "mamujoco":
        raise ValueError(f"Unknown benchmark suite: {suite}")

    env_fns = [
        partial(
            MultiAgentMujocoEnv,
            scenario=scenario,
            agent_conf=mamujoco_agent_conf,
            agent_obsk=mamujoco_agent_obsk,
            episode_length=episode_length,
            env_kwargs=scenario_kwargs,
        )
        for _ in range(num_envs)
    ]
    if mamujoco_num_workers == 0:
        return SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    return WorkerPoolAsyncVectorEnv(
        env_fns,
        num_workers=mamujoco_num_workers,
        autoreset_mode=AutoresetMode.SAME_STEP,
        copy=False,
    )


def _make_obs_indices(vector_env: Any) -> ObsIndices:
    single_observation_space = vector_env.single_observation_space
    local_obs_dim = int(single_observation_space["local_obs"].shape[-1])
    hidden_global_vars_dim = int(
        single_observation_space["hidden_global_vars"].shape[-1]
    )
    return ObsIndices(
        local_scalar_indices=list(range(local_obs_dim)),
        local_angle_indices=[],
        local_rot6d_indices=[],
        local_binary_indices=[],
        local_quaternion_indices=[],
        global_scalar_indices=[],
        global_rot6d_indices=[],
        global_quaternion_indices=[],
        hidden_local_vars_scalar_indices=[],
        hidden_local_vars_quaternion_indices=[],
        hidden_global_vars_scalar_indices=list(range(hidden_global_vars_dim)),
        hidden_global_vars_quaternion_indices=[],
    )


def _wrap_vector_env(
    *,
    vector_env: Any,
    obs_indices: ObsIndices,
    rollout_device: torch.device,
    use_popart: bool,
) -> Any:
    env = ContinuousActionsLearnEnvWrapper(vector_env, device=rollout_device)
    env = TorchRecordEpisodeStatisticsWrapper(env)
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="local_obs",
        scalar_feature_indices=obs_indices.local_scalar_indices,
        quaternion_indices=[],
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_global_vars",
        scalar_feature_indices=obs_indices.hidden_global_vars_scalar_indices,
        quaternion_indices=[],
    )
    if not use_popart:
        env = TorchNormalizeRewardWrapper(env, gamma=0.99)
    return env


def _wrap_ppo_policy_with_nop(
    *,
    base_policy: Any,
    env: Any,
    use_nop: bool,
    obs_indices: ObsIndices,
    compile_modules: bool,
    local_latent_dim: int,
    nop_init_gains: NOPInitGains,
    world_model_loss_coef: float,
    world_model_num_next_steps: int,
    transition_model_d_model: int,
) -> Any:
    if not use_nop:
        return base_policy
    return NextObsPredWrapper(
        policy=base_policy,
        world_model_config=NOPWorldModelConfig(
            n_agents=env.n_agents,
            local_latent_dim=local_latent_dim,
            action_dim=env.action_space.total_agent_action_dim,
            world_model_loss_coef=world_model_loss_coef,
            compile_modules=compile_modules,
            compile_mode="default",
            act_fn_cls=nn.GELU,
            wm_pre_transition_init_gain=nop_init_gains.pre_transition,
            transition_model_coembed_init_gain=nop_init_gains.transition_coembed,
            transition_model_transformer_ff_init_gain=nop_init_gains.transition_transformer_ff,
            transition_model_head_init_gain=nop_init_gains.transition_head,
            wm_pre_predictors_init_gain=nop_init_gains.pre_predictors,
            wm_predictor_init_gain=nop_init_gains.predictors,
            transition_model_dropout=0.0,
            wm_pre_transition_dims=[local_latent_dim],
            d_model_transition_model=transition_model_d_model,
            nhead_transition_model=2,
            num_layers_transition_model=2,
            dim_feedforward_transition_model=transition_model_d_model * 2,
            add_agent_embeddings_transition_model=False,
            transition_model_coembed_hidden_dims=[transition_model_d_model],
            wm_pre_predictors_dims=[transition_model_d_model, transition_model_d_model],
            wm_scalar_predictor_hidden_dims=[],
            wm_angle_predictor_hidden_dims=[],
            wm_rot6d_predictor_hidden_dims=[],
            wm_binary_predictor_hidden_dims=[],
            scalar_loss_fn="smooth_l1",
            next_obs_pred_config=NextObsPredConfig(
                local_scalar_target_indices=obs_indices.local_scalar_indices,
                local_angle_target_indices=[],
                local_rot6d_target_indices=[],
                local_binary_target_indices=[],
            ),
        ),
    )


def _make_algorithm(
    *,
    policy: Any,
    env: Any,
    policy_variant: BenchmarkPolicyVariant,
    use_nop: bool,
    rollout_samples: int,
    rollout_steps_per_env: int,
    episode_length: int,
    world_model_num_next_steps: int,
    rollout_device: torch.device,
    train_device: torch.device,
    record_device: torch.device,
) -> PPO | SAC:
    if policy_variant == "tmasac":
        learning_starts = max(10_000, rollout_samples * 4)
        return SAC(
            policy=policy,
            env=env,
            learning_rate=5e-5,
            buffer_capacity_per_env=max(
                episode_length * 2,
                math.ceil(max(learning_starts, rollout_samples) / env.num_envs),
            ),
            learning_starts=learning_starts,
            batch_size=rollout_samples,
            rollout_steps_per_iteration=rollout_samples,
            rollout_warmup_steps_per_env=episode_length,
            gradient_steps=8,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto_0.05",
            ent_coef_learning_rate=None,
            target_entropy="auto_0.1",
            target_update_interval=1,
            max_grad_norm=2.0,
            independent_nop_sampling=False,
            gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
            train_device=train_device,
            rollout_device=rollout_device,
            record_device=record_device,
            replay_storage_device=train_device,
            metrics_action_splitters=[None],
        )

    n_epochs = 8
    target_kl = 0.002
    warm_lr = 1e-4
    warmup_iterations = 200
    auto_lr = AutomaticLearningRate(
        initial_lr=warm_lr * 5e-3,
        max_lr=8e-4,
        updater=make_auto_lr_updater(
            early_stop_epoch_decay_limit=math.ceil(n_epochs * 0.75),
            max_kl_div=target_kl * 1.55,
            warm_scheduler_config=CosineSchedulerConfig(
                unit=ScheduleUnit.ITERATIONS,
                duration=warmup_iterations,
                start_value=warm_lr * 5e-3,
                final_value=warm_lr,
            ),
        ),
    )
    sampler_config = (
        PPOWMSamplerConfig(
            batch_size=rollout_samples,
            num_next_steps=world_model_num_next_steps,
            compile_wm_window_helper=use_nop,
        )
        if use_nop
        else PPOSamplerConfig(batch_size=rollout_samples)
    )
    return PPO(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        rollout_mode=StepsRolloutMode(rollout_samples),
        rollout_warmup_steps_per_env=episode_length,
        max_episode_length=episode_length,
        sampler_config=sampler_config,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.05,
        target_kl=target_kl,
        max_grad_norm=2.0,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
        mc_ent_coef=0.0,
        vf_coef=2.0,
        value_loss_fn=nn.MSELoss(reduction="none"),
        train_device=train_device,
        rollout_device=rollout_device,
        record_device=record_device,
        use_popart=True,
        metrics_action_splitters=[None],
        virtual_mini_batches=1,
        parameter_lr_multipliers=make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            decoder_lr_multiplier=0.25,
        ),
    )


def _logging_console_keys(
    *,
    is_sac: bool,
    use_nop: bool,
) -> list[
    tuple[str, str | SummaryStatisticsFormat | None]
    | tuple[str, str | SummaryStatisticsFormat | None, str]
]:
    keys: list[
        tuple[str, str | SummaryStatisticsFormat | None]
        | tuple[str, str | SummaryStatisticsFormat | None, str]
    ] = [
        ("iteration", "5", "it"),
        ("timesteps", "8", "steps"),
        (
            "ep_rew",
            SummaryStatisticsFormat(mean=" .2f", std=".2f", max_value=" .2f", n="1"),
        ),
        ("ep_rew_ema", " .3f"),
        ("fps", None),
    ]
    if is_sac:
        keys.extend(
            [
                ("critic_loss", SummaryStatisticsFormat(mean=".3f")),
                ("actor_loss", SummaryStatisticsFormat(mean=".3f")),
                ("ent_coef", SummaryStatisticsFormat(mean=".3f")),
                ("replay_size", "8"),
            ]
        )
        if use_nop:
            keys.append(("critic_nop_loss_scaled", None, "critic_nop"))
    else:
        keys.extend(
            [
                ("approx_kl", SummaryStatisticsFormat(mean=".2e", std=".2e")),
                ("val_loss_scaled", None, "val_loss"),
            ]
        )
        if use_nop:
            keys.append(("wm_loss_scaled", None, "wm_loss"))
    return keys
