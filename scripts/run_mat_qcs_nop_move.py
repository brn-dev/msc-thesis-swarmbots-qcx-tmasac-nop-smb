from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Callable

import swarmbots.mj_env.scenarios.scenario_presets as mj_scenario_presets
from swarmbots.mj_env.scenarios.scenario_presets import default_move_to
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.utils.recording_schedule import DEFAULT_RECORDING_SCHEDULE, install_scheduled_recordings
from swarmbots.utils.run_paths import generate_run_id, get_run_id_from_checkpoint_path, make_run_dir


def configure_float32_matmul_precision() -> None:
    import torch

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def make_env_fn(
        episode_length: int,
        render_mode: str | None = None,
        first_episode_length: int | None = None,
        timestep: float = mj_scenario_presets.DEFAULT_KWARGS["timestep"],
        action_repeat: int = mj_scenario_presets.DEFAULT_KWARGS["action_repeat"],
) -> Callable[[], SwarmBotsEnv]:

    def _init() -> SwarmBotsEnv:
        scenario = default_move_to(
            quantize_connection_twist=8,
            timestep=timestep,
            action_repeat=action_repeat,
            visualize_goal=True,
            continuous_connector_actions=False,
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=0,
            first_episode_length=first_episode_length,
        )

    return _init

def wrap_vec_env(
        vector_env: Any,
        obs_indices: Any,
        gamma: float,
        use_popart: bool,
        rollout_device: Any,
        normalize_prev_binary_actions: bool = False,
) -> Any:
    from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
    from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
    from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
    from swarmbots.learn.env_wrappers.torch_progress_guidance_ep_stats_wrapper import (
        TorchProgressGuidanceEpisodeStatsWrapper,
    )
    from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
    from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper

    env = SwarmBotsLearnEnvWrapper(vector_env, device=rollout_device)
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
    env = TorchTransitionObsWrapper(env, normalize_prev_binary_actions=normalize_prev_binary_actions)
    if not use_popart:
        env = TorchNormalizeRewardWrapper(env, gamma=gamma)
    return env


def split_actuator_joints(actions: Any, actuators_per_limb: int) -> dict[str, Any]:
    return {
        f'j{i}': actions[..., i::actuators_per_limb]
        for i in range(actuators_per_limb)
    }


def set_actuator_gsde_init_joint_stds(
        policy: Any,
        actuators_per_limb: int,
        joint_stds: list[float]
) -> None:
    import torch
    from loguru import logger

    from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist

    if len(joint_stds) != actuators_per_limb:
        raise ValueError()

    gsde_dist = next((dist for dist in policy.action_dist.distributions if isinstance(dist, GSDEActionDist)), None)

    if gsde_dist is None:
        logger.warning("No gSDE dist found, skipping log std init")
        return

    with torch.no_grad():
        for i, joint_std in enumerate(joint_stds):
            gsde_dist.log_stds[:, i::actuators_per_limb] = math.log(joint_std)


def main() -> None:
    import torch
    from gymnasium.vector import SyncVectorEnv, AutoresetMode
    from loguru import logger
    from torch import nn

    from swarmbots.learn.torch_logging import enable_torch_compile_logging
    from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
    from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig, AgentActionsReduction
    from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
    from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
    from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
    from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
    from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, StepsRolloutMode, PPO
    from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
    from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
    from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NextObsPredWrapper, NOPWorldModelConfig
    from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
    from swarmbots.learn.env_wrappers.worker_pool_async_vector_env import WorkerPoolAsyncVectorEnv
    from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
    from swarmbots.learn.discord_notifications import run_with_discord_notification
    from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
    from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
    from swarmbots.learn.scheduling.schedulers import ScheduleUnit
    from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
    from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
    from swarmbots.mj_env.scenarios import scenario_presets

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    n_workers = 23
    n_envs = 1024
    rollout_steps_per_env = 4

    episode_length = 512
    total_timesteps = 100_000_000
    save_interval = None

    use_popart = True
    popart_beta = 5e-4
    popart_init_sigma = 0.65

    vf_coef = 2.0 if use_popart else 0.5
    world_model_loss_coef = 0.1

    world_model_num_next_steps = 3

    compile_policy_modules = True
    policy_compile_mode = "default"
    compile_world_model_modules = True

    # gsde_init_stds = [0.25, 0.25, 0.15]
    gsde_init_stds = [0.25, 0.30]

    # =====  ID  =====
    run_id = generate_run_id()

    # ===== LOAD =====
    load_path: str | Path | None = None
    # load_path = make_run_dir("mat_qcs_nop_swarm_bots_move", "2026-03-29_01-08-59") / "models" / "model_77792876_steps_stopped.pt"

    # ===== DEVICE =====
    use_cuda = True and torch.cuda.is_available()
    use_cuda_rollout = True and torch.cuda.is_available()
    rollout_device = torch.device("cuda" if use_cuda_rollout else "cpu")
    train_device = torch.device("cuda" if use_cuda else "cpu")
    record_device = torch.device("cpu")

    logger.info(f'{rollout_device = }')
    logger.info(f'{train_device = }')
    logger.info(f'{record_device = }')

    if load_path is not None:
        if Path(load_path).suffix != '.pt':
            logger.error('load_path is missing .pt')
            raise ValueError()
        logger.info(f'{load_path = }')
        run_id = get_run_id_from_checkpoint_path(load_path)
    logger.info(f'{run_id = }')

    run_dir = make_run_dir("mat_qcs_nop_swarm_bots_move", run_id)
    save_optimizer = True

    env_fns = [
        make_env_fn(
            episode_length=episode_length,
            render_mode=None,
            first_episode_length=int(i * episode_length / n_envs)
        )
        for i in range(1, n_envs + 1)
    ]

    print("Creating dummy env for capturing settings...")
    dummy_env = env_fns[0]()
    env_settings = dummy_env.get_settings()
    local_obs_dim = int(dummy_env.observation_space["local_obs"].shape[-1])
    global_obs_dim = int(dummy_env.observation_space["global_obs"].shape[-1])
    hidden_local_vars_dim = int(dummy_env.observation_space["hidden_local_vars"].shape[-1])
    hidden_global_vars_dim = int(dummy_env.observation_space["hidden_global_vars"].shape[-1])
    dummy_env.close()
    del dummy_env
    print("Env settings captured.")

    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        hidden_local_vars_dim=hidden_local_vars_dim,
        hidden_global_vars_dim=hidden_global_vars_dim,
    )

    from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
    def make_record_env() -> SwarmBotsLearnEnvWrapper:
        record_env = SyncVectorEnv([
            make_env_fn(
                episode_length=episode_length,
                render_mode='rgb_array'
            )
        ], autoreset_mode=AutoresetMode.SAME_STEP)
        record_env = wrap_vec_env(
            vector_env=record_env,
            obs_indices=obs_indices,
            gamma=gamma,
            use_popart=use_popart,
            rollout_device=record_device,
        )
        return record_env

    print(f'Creating vector env (n={n_envs})...')
    if sys.gettrace() is None:
        # vector_env = AsyncVectorEnv(env_fns)
        vector_env = WorkerPoolAsyncVectorEnv(
            env_fns,
            num_workers=n_workers,
            autoreset_mode=AutoresetMode.SAME_STEP,
            copy=False,
        )
    else:
        vector_env = SyncVectorEnv(env_fns[:1], autoreset_mode=AutoresetMode.SAME_STEP)
    print(f"Created {type(vector_env)} with {n_envs} environments.")

    if isinstance(vector_env, SyncVectorEnv):
        for _ in range(10):
            logger.warning('USING SYNC VECTOR ENV')

    gamma = 0.99

    print("Wrapping...")
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        use_popart=use_popart,
        rollout_device=rollout_device,
    )

    print(f"Environment initialized.")
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

    enc_d_model = 256
    dec_d_model = 96
    transition_model_d_model = 192

    enc_nhead = 4
    dec_nhead = 2
    transition_model_nhead = 4

    print("Initializing Policy...")
    mat_qcs_policy = MATQCSPolicy(
        env=env,
        config=MATQCSPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=enc_d_model,
                nhead=enc_nhead,
                num_layers=2,
                dim_feedforward=enc_d_model * 2,
                add_agent_embeddings=True,
                local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
                global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
            ),
            decoder_config=MATQCSDecoderConfig(
                d_model=dec_d_model,
                nhead=dec_nhead,
                num_layers=2,
                dim_feedforward=dec_d_model * 2,
                add_agent_embeddings=True,
                query_encoder_hidden_dims=[2 * dec_d_model],
                context_encoder_hidden_dims=[2 * dec_d_model],
                memory_dims=None,
                self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL
            ),
            critic_config=MATQCSCriticConfig(
                n_local_projection_hidden_layers=2,
                n_value_regressor_hidden_layers=1,
                use_popart=use_popart,
                popart_config=PopArtConfig(
                    beta=popart_beta,
                    init_sigma=popart_init_sigma,
                ),
            ),
            dropout=0.0,
            act_fn_cls=nn.GELU,
            # continuous_config=GSDEConfig(
            #     base_std=0.25,
            #     latent_sde_dim=None,
            #     std_learnable=True,
            #     full_std=True,
            #     sde_learn_features=False,
            #     log_std_clamp_range=(-20.0, 2.0),
            #     normalize_latent_sde_by_dim=True
            # ),
            # continuous_config=BetaMixtureConfig(
            #     num_components=3,
            #     alphas=(3.0, 10.0, 10.0),
            #     betas=(10.0, 10.0, 3.0)
            # ),
            continuous_config=SignMagnitudeBetaConfig(
                ent_loss_coef=1e-3,
                beta_ent_scale=0.75,
                categorical_ent_loss_config=EntropyLossConfig(
                    agent_actions_reduction=AgentActionsReduction.SUM,
                    metrics_reduction=AgentActionsReduction.MEAN,
                ),
                beta_ent_loss_config=EntropyLossConfig(
                    agent_actions_reduction=AgentActionsReduction.SUM,
                    metrics_reduction=AgentActionsReduction.MEAN,
                ),
            ),
            bernoulli_config=BernoulliConfig(
                initial_prob=0.8,
                ent_loss_coef=1e-3,
                ent_loss_config=EntropyLossConfig(
                    agent_actions_reduction=AgentActionsReduction.SUM,
                    metrics_reduction=AgentActionsReduction.MEAN,
                ),
            ),
            max_agents=20,
            compile_modules=compile_policy_modules,
            compile_mode=policy_compile_mode,
        ),
    )
    policy = NextObsPredWrapper(
        policy=mat_qcs_policy,
        world_model_config=NOPWorldModelConfig(
            n_agents=env.n_agents,
            local_latent_dim=enc_d_model,
            action_dim=env.action_space.total_agent_action_dim,
            world_model_loss_coef=world_model_loss_coef,
            compile_modules=compile_world_model_modules,
            compile_mode=policy_compile_mode,
            act_fn_cls=nn.GELU,
            transition_model_dropout=0.0,
            wm_pre_transition_dims=[enc_d_model],
            d_model_transition_model=transition_model_d_model,
            nhead_transition_model=transition_model_nhead,
            num_layers_transition_model=2,
            dim_feedforward_transition_model=transition_model_d_model * 2,
            add_agent_embeddings_transition_model=True,
            transition_model_coembed_hidden_dims=[transition_model_d_model],
            wm_pre_predictors_dims=[transition_model_d_model, transition_model_d_model],
            wm_scalar_predictor_hidden_dims=[],
            wm_angle_predictor_hidden_dims=[],
            wm_rot6d_predictor_hidden_dims=[],
            wm_binary_predictor_hidden_dims=[],
            scalar_loss_fn='smooth_l1',
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

    print("Initializing PPO Algorithm...")

    warm_lr = 1e-4
    warmup_iterations: int = 200
    cold_lr = warm_lr * 5e-3 if warmup_iterations > 0 else warm_lr

    auto_lr = AutomaticLearningRate(
        initial_lr=cold_lr,
        max_lr=8e-4,
        updater=make_auto_lr_updater(
            warm_scheduler_config=CosineSchedulerConfig(
                unit=ScheduleUnit.ITERATIONS,
                duration=warmup_iterations,
                start_value=cold_lr,
                final_value=warm_lr,
            ) if warmup_iterations > 0 else None,
        )
    )

    rollout_samples = n_envs * rollout_steps_per_env
    ppo = PPO(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        rollout_mode=StepsRolloutMode(rollout_samples),
        max_episode_length=episode_length,
        sampler_config=PPOWMSamplerConfig(
            batch_size=rollout_samples,
            num_next_steps=world_model_num_next_steps,
            compile_wm_window_helper=True,
        ),
        n_epochs=8,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.07,
        target_kl=0.007,
        max_grad_norm=2.0,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1/6),
        mc_ent_coef=0e-5,
        vf_coef=vf_coef,
        # value_loss_fn=nn.SmoothL1Loss(reduction="none"),
        value_loss_fn=nn.MSELoss(reduction="none"),
        train_device=train_device,
        rollout_device=rollout_device,
        record_device=record_device,
        use_popart=use_popart,
        metrics_action_splitters=[lambda actions: split_actuator_joints(actions, actuators_per_limb), None],
        scheduler_manager=None,
    )

    if load_path:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path, recover_best_return_ema=False, strict_load_state_dict=True)

    scheduled_recording_hook = install_scheduled_recordings(
        algorithm=ppo,
        total_timesteps=total_timesteps,
        schedule=DEFAULT_RECORDING_SCHEDULE,
    )

    print("Starting training...")
    logging_console_keys: list[tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]] = [
        ('iteration', '5', 'it'),
        ('timesteps', '8', 'steps'),
        ('total_updates', '6', 'tot_upd'),
        # ('act0', SummaryStatisticsFormat(histogram=10)),
    ]
    logging_console_keys.extend(
        (f'act0_j{i}', SummaryStatisticsFormat(histogram=11))
        for i in range(actuators_per_limb)
    )
    logging_console_keys.extend(
        (f'std0_j{i}', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.3f', max_value='.3f'))
        for i in range(actuators_per_limb)
    )
    logging_console_keys.extend([
        ('act1', SummaryStatisticsFormat(histogram=2)),
        ('updates', '3', 'upd'),
        ('approx_kl', SummaryStatisticsFormat(mean='.3f', std='.3f', max_value='.3f')),
        ('clip_frac', None),
        ('ratio', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.1e', max_value='.3f')),
        ('wm_loss_scaled', None, 'wm_loss'),
        ('val_loss_scaled', None, 'val_loss'),
        ('expl_var', '.3f'),
        ('popart_mu', '.3f', 'pa_mu'),
        ('popart_sigma', '.3f', 'pa_sigma'),
        ('ep_rew', SummaryStatisticsFormat(mean=' .2f', std='.2f', max_value=' .2f', n='1')),
        ('ep_rew_ema', ' .3f'),
        ('best_ep_rew_ema', ' .3f', 'best_ema'),
        ('fps', None),
    ])

    run_with_discord_notification(
        run_name=f"mat_qcs_nop_move/{run_id}",
        run_dir=run_dir,
        total_timesteps=total_timesteps,
        algorithm=ppo,
        run=lambda: ppo.learn(
            max_total_timesteps=total_timesteps,
            run_dir=run_dir,
            log_interval=1,
            save_interval=save_interval,
            save_optimizer=save_optimizer,
            best_rotation_n=1,
            extra_run_metadata={
                'load_path': load_path,
                'env_settings': env_settings,
                'script': Path(__file__).read_text(encoding='utf-8'),
                'script_scenario_presets': Path(scenario_presets.__file__).read_text(encoding='utf-8'),
            },
            logging_console_keys=logging_console_keys,
            make_record_env=make_record_env,
            post_iteration_hooks=[scheduled_recording_hook],
        ),
    )

    print("Training Finished.")

    env.close()


if __name__ == "__main__":
    main()
