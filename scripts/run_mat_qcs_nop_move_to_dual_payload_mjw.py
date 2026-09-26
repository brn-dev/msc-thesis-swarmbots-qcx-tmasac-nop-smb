from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from gymnasium.vector import VectorEnv
from loguru import logger
from torch import nn

from run_mat_qcs_nop_payload import wrap_vec_env, split_actuator_joints, set_actuator_gsde_init_joint_stds
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig, AgentActionsReduction
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NextObsPredWrapper, NOPWorldModelConfig
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.checkpointing import (
    apply_env_state,
    extract_env_state,
    extract_policy_state_dict,
    load_checkpoint,
)
from swarmbots.learn.discord_notifications import run_with_discord_notification
from swarmbots.learn.env_wrappers.move_to_payload_global_obs_adapter import MoveToDualPayloadGlobalObsAdapter
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleUnit
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mjw_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_dual_payload_plane, default_move_to
from swarmbots.scenario_presets.scenario_presets_kwargs import DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS
from swarmbots.utils.recording_schedule import DEFAULT_LIVE_RECORDING_SCHEDULE, install_scheduled_recordings
from swarmbots.utils.run_paths import generate_run_id, make_run_dir

import swarmbots.mjw_env.scenarios.mjw_scenario_presets as mjw_scenario_presets


CRITIC_STATE_PREFIXES = ("policy.critic.",)
OPTIONAL_TRANSFER_STATE_PREFIXES = (
    *CRITIC_STATE_PREFIXES,
    "global_pool_encoder.",
    "global_scalars_predictor.",
    "global_rot6ds_predictor.",
)
MJW_CCD_ITERATIONS = 128


class SequentialRunProgress:
    def __init__(self) -> None:
        self.completed_timesteps = 0
        self.completed_iterations = 0
        self.completed_updates = 0
        self.current_algorithm: PPO | None = None

    @property
    def n_total_timesteps(self) -> int:
        return self.completed_timesteps + self._current_attr("n_total_timesteps")

    @property
    def n_total_iterations(self) -> int:
        return self.completed_iterations + self._current_attr("n_total_iterations")

    @property
    def n_total_updates(self) -> int:
        return self.completed_updates + self._current_attr("n_total_updates")

    @property
    def _best_return_ema(self) -> float | None:
        if self.current_algorithm is None:
            return None
        return getattr(self.current_algorithm, "_best_return_ema", None)

    def finish_current_phase(self) -> None:
        if self.current_algorithm is None:
            return
        self.completed_timesteps += int(self.current_algorithm.n_total_timesteps)
        self.completed_iterations += int(self.current_algorithm.n_total_iterations)
        self.completed_updates += int(self.current_algorithm.n_total_updates)
        self.current_algorithm = None

    def _current_attr(self, name: str) -> int:
        if self.current_algorithm is None:
            return 0
        return int(getattr(self.current_algorithm, name))


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def make_move_to_vector_env(
    *,
    episode_length: int,
    num_envs: int,
    first_episode_lengths: list[int],
    settle_initial_reset: bool,
    device: torch.device,
) -> VectorEnv:
    vector_env = MJWSwarmBotsVectorEnv(
        scenario=default_move_to(
            visualize_goal=True,
            continuous_connector_actions=False,
        ),
        num_envs=num_envs,
        episode_length=episode_length,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=settle_initial_reset,
        device=device,
    )
    return MoveToDualPayloadGlobalObsAdapter(
        vector_env,
        payload_z=float(DUAL_PAYLOAD_PLANE_SCENARIO_KWARGS["payload_radius"]),
    )


def make_payload_vector_env(
    *,
    episode_length: int,
    num_envs: int,
    first_episode_lengths: list[int],
    settle_initial_reset: bool,
    device: torch.device,
) -> MJWSwarmBotsVectorEnv:
    return MJWSwarmBotsVectorEnv(
        scenario=default_dual_payload_plane(continuous_connector_actions=False),
        num_envs=num_envs,
        episode_length=episode_length,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=settle_initial_reset,
        device=device,
        ccd_iterations=MJW_CCD_ITERATIONS,
    )


def build_env(
    *,
    vector_env: VectorEnv,
    gamma: float,
    use_popart: bool,
    rollout_device: torch.device,
) -> tuple[Any, dict[str, Any], ObsIndices]:
    env_settings = vector_env.get_settings()
    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=int(vector_env.single_observation_space["local_obs"].shape[-1]),
        global_obs_dim=int(vector_env.single_observation_space["global_obs"].shape[-1]),
        hidden_local_vars_dim=int(vector_env.single_observation_space["hidden_local_vars"].shape[-1]),
        hidden_global_vars_dim=int(vector_env.single_observation_space["hidden_global_vars"].shape[-1]),
    )
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        use_popart=use_popart,
        rollout_device=rollout_device,
    )
    return env, env_settings, obs_indices


def build_policy(
    *,
    env: Any,
    obs_indices: ObsIndices,
    use_popart: bool,
    popart_beta: float,
    popart_init_sigma: float,
    compile_policy_modules: bool,
    compile_world_model_modules: bool,
    policy_compile_mode: str,
    world_model_loss_coef: float,
    enc_d_model: int,
    dec_d_model: int,
    transition_model_d_model: int,
) -> NextObsPredWrapper:
    predict_global_obs = bool(obs_indices.global_rot6d_indices)
    mat_qcs_policy = MATQCSPolicy(
        env=env,
        config=MATQCSPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=enc_d_model,
                nhead=4,
                num_layers=2,
                dim_feedforward=enc_d_model * 2,
                add_agent_embeddings=True,
                local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
                global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
            ),
            decoder_config=MATQCSDecoderConfig(
                d_model=dec_d_model,
                nhead=2,
                num_layers=2,
                dim_feedforward=dec_d_model * 2,
                add_agent_embeddings=True,
                query_encoder_hidden_dims=[2 * dec_d_model],
                context_encoder_hidden_dims=[2 * dec_d_model],
                memory_dims=None,
                self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
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
    return NextObsPredWrapper(
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
            nhead_transition_model=4,
            num_layers_transition_model=2,
            dim_feedforward_transition_model=transition_model_d_model * 2,
            add_agent_embeddings_transition_model=True,
            transition_model_coembed_hidden_dims=[transition_model_d_model],
            wm_pre_predictors_dims=[transition_model_d_model, transition_model_d_model],
            wm_scalar_predictor_hidden_dims=[],
            wm_angle_predictor_hidden_dims=[],
            wm_rot6d_predictor_hidden_dims=[],
            wm_binary_predictor_hidden_dims=[],
            wm_global_pool_hidden_dims=[transition_model_d_model] if predict_global_obs else None,
            wm_global_scalar_predictor_hidden_dims=[] if predict_global_obs else None,
            wm_global_rot6d_predictor_hidden_dims=[] if predict_global_obs else None,
            scalar_loss_fn="smooth_l1",
            next_obs_pred_config=NextObsPredConfig(
                local_scalar_target_indices=obs_indices.local_scalar_indices,
                local_angle_target_indices=obs_indices.local_angle_indices,
                local_rot6d_target_indices=obs_indices.local_rot6d_indices,
                local_binary_target_indices=obs_indices.local_binary_indices,
                global_scalar_target_indices=obs_indices.global_scalar_indices if predict_global_obs else None,
                global_rot6d_target_indices=obs_indices.global_rot6d_indices if predict_global_obs else None,
                scalar_loss_weight=1.0,
                angle_loss_weight=1.0,
                rot6d_loss_weight=1.0,
                binary_loss_weight=1.0,
                global_scalar_loss_weight=1.0,
                global_rot6d_loss_weight=1.0,
            ),
        ),
    )


def build_ppo(
    *,
    env: Any,
    policy: NextObsPredWrapper,
    actuators_per_limb: int,
    rollout_samples: int,
    episode_length: int,
    gamma: float,
    use_popart: bool,
    vf_coef: float,
    world_model_num_next_steps: int,
    rollout_device: torch.device,
    train_device: torch.device,
    record_device: torch.device,
) -> PPO:
    warm_lr = 1e-4
    warmup_iterations = 200
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
            )
            if warmup_iterations > 0
            else None,
        ),
    )
    return PPO(
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
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
        mc_ent_coef=0e-5,
        vf_coef=vf_coef,
        value_loss_fn=nn.MSELoss(reduction="none"),
        train_device=train_device,
        rollout_device=rollout_device,
        record_device=record_device,
        use_popart=use_popart,
        metrics_action_splitters=[lambda actions: split_actuator_joints(actions, actuators_per_limb), None],
        scheduler_manager=None,
    )


def load_transfer_checkpoint(
    *,
    policy: NextObsPredWrapper,
    env: Any,
    checkpoint_path: Path,
    map_location: torch.device,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location=map_location)
    pretrained_state = extract_policy_state_dict(checkpoint)
    transferred_state = {
        key: value
        for key, value in pretrained_state.items()
        if not key.startswith(CRITIC_STATE_PREFIXES)
    }
    missing_keys, unexpected_keys = policy.load_state_dict(transferred_state, strict=False)
    unexpected_keys = list(unexpected_keys)
    non_critic_missing_keys = [
        key
        for key in missing_keys
        if not key.startswith(OPTIONAL_TRANSFER_STATE_PREFIXES)
    ]
    if unexpected_keys or non_critic_missing_keys:
        raise RuntimeError(
            "Transfer checkpoint did not match the dual-payload policy. "
            f"{non_critic_missing_keys = }, {unexpected_keys = }"
        )

    apply_env_state(env, extract_env_state(checkpoint))
    logger.info(
        f"Loaded transfer checkpoint {checkpoint_path}; "
        f"transferred {len(transferred_state)} tensors and initialized {len(missing_keys)} unmatched tensors."
    )


def logging_console_keys(actuators_per_limb: int) -> list[
    tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]
]:
    keys: list[
        tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]
    ] = [
        ("iteration", "5", "it"),
        ("timesteps", "8", "steps"),
        ("total_updates", "6", "tot_upd"),
    ]
    keys.extend((f"act0_j{i}", SummaryStatisticsFormat(histogram=11)) for i in range(actuators_per_limb))
    keys.extend(
        (f"std0_j{i}", SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".3f", max_value=".3f"))
        for i in range(actuators_per_limb)
    )
    keys.extend(
        [
            ("act1", SummaryStatisticsFormat(histogram=2)),
            ("updates", "3", "upd"),
            ("approx_kl", SummaryStatisticsFormat(mean=".3f", std=".3f", max_value=".3f")),
            ("clip_frac", None),
            ("ratio", SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".1e", max_value=".3f")),
            ("wm_loss_scaled", None, "wm_loss"),
            ("val_loss_scaled", None, "val_loss"),
            ("expl_var", SummaryStatisticsFormat(mean=".3f", std=".3f")),
            ("popart_mu", ".3f", "pa_mu"),
            ("popart_sigma", ".3f", "pa_sigma"),
            ("ep_rew", SummaryStatisticsFormat(mean=" .2f", std=".2f", max_value=" .2f", n="1")),
            ("ep_rew_ema", " .3f"),
            ("best_ep_rew_ema", " .3f", "best_ema"),
            ("fps", None),
        ]
    )
    return keys


def build_phase(
    *,
    vector_env: VectorEnv,
    gamma: float,
    use_popart: bool,
    popart_beta: float,
    popart_init_sigma: float,
    vf_coef: float,
    world_model_loss_coef: float,
    world_model_num_next_steps: int,
    gsde_init_stds: list[float],
    compile_policy_modules: bool,
    compile_world_model_modules: bool,
    policy_compile_mode: str,
    rollout_samples: int,
    episode_length: int,
    rollout_device: torch.device,
    train_device: torch.device,
    record_device: torch.device,
) -> tuple[Any, dict[str, Any], PPO, int]:
    env, env_settings, obs_indices = build_env(
        vector_env=vector_env,
        gamma=gamma,
        use_popart=use_popart,
        rollout_device=rollout_device,
    )
    if env.connectors_dim <= 0 or env.actuators_dim % env.connectors_dim != 0:
        raise ValueError(f"Expected actuators_dim divisible by connectors_dim, got {env.actuators_dim=} {env.connectors_dim=}")
    actuators_per_limb = env.actuators_dim // env.connectors_dim

    policy = build_policy(
        env=env,
        obs_indices=obs_indices,
        use_popart=use_popart,
        popart_beta=popart_beta,
        popart_init_sigma=popart_init_sigma,
        compile_policy_modules=compile_policy_modules,
        compile_world_model_modules=compile_world_model_modules,
        policy_compile_mode=policy_compile_mode,
        world_model_loss_coef=world_model_loss_coef,
        enc_d_model=256,
        dec_d_model=96,
        transition_model_d_model=192,
    )
    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=actuators_per_limb,
        joint_stds=gsde_init_stds,
    )
    ppo = build_ppo(
        env=env,
        policy=policy,
        actuators_per_limb=actuators_per_limb,
        rollout_samples=rollout_samples,
        episode_length=episode_length,
        gamma=gamma,
        use_popart=use_popart,
        vf_coef=vf_coef,
        world_model_num_next_steps=world_model_num_next_steps,
        rollout_device=rollout_device,
        train_device=train_device,
        record_device=record_device,
    )
    logger.info(f"Initialized policy with {policy.num_parameters():,} learnable parameters.")
    return env, env_settings, ppo, actuators_per_limb


def main() -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    if not torch.cuda.is_available():
        raise RuntimeError("run_mat_qcs_nop_move_to_dual_payload_mjw.py requires CUDA.")

    n_envs = 1024
    rollout_steps_per_env = 4
    rollout_samples = n_envs * rollout_steps_per_env
    episode_length = 512

    move_to_pretrain_timesteps = 20_000_000
    payload_timesteps = 100_000_000
    save_interval = None
    save_optimizer = True

    gamma = 0.99
    use_popart = True
    popart_beta = 5e-4
    popart_init_sigma = 0.65
    vf_coef = 2.0 if use_popart else 0.5
    world_model_loss_coef = 0.1
    world_model_num_next_steps = 3
    gsde_init_stds = [0.25, 0.30]
    compile_policy_modules = True
    policy_compile_mode = "default"
    compile_world_model_modules = True
    pretrain_recording_schedule = {95: 10}

    rollout_device = torch.device("cuda")
    train_device = torch.device("cuda")
    record_device = torch.device("cuda")
    run_id = generate_run_id()
    run_dir = make_run_dir("mat_qcs_nop_swarm_bots_move_to_dual_payload_mjw", run_id)
    first_episode_lengths = [int((i + 1) * episode_length / n_envs) for i in range(n_envs)]
    progress = SequentialRunProgress()

    def run() -> None:
        logger.info("Starting move-to pretrain with dual-payload-shaped global observations.")
        move_env, move_env_settings, move_ppo, move_actuators_per_limb = build_phase(
            vector_env=make_move_to_vector_env(
                episode_length=episode_length,
                num_envs=n_envs,
                first_episode_lengths=first_episode_lengths,
                settle_initial_reset=True,
                device=rollout_device,
            ),
            gamma=gamma,
            use_popart=use_popart,
            popart_beta=popart_beta,
            popart_init_sigma=popart_init_sigma,
            vf_coef=vf_coef,
            world_model_loss_coef=world_model_loss_coef,
            world_model_num_next_steps=world_model_num_next_steps,
            gsde_init_stds=gsde_init_stds,
            compile_policy_modules=compile_policy_modules,
            compile_world_model_modules=compile_world_model_modules,
            policy_compile_mode=policy_compile_mode,
            rollout_samples=rollout_samples,
            episode_length=episode_length,
            rollout_device=rollout_device,
            train_device=train_device,
            record_device=record_device,
        )
        progress.current_algorithm = move_ppo
        move_phase_dir = run_dir / "move_to_pretrain"
        try:
            move_recording_hook = install_scheduled_recordings(
                algorithm=move_ppo,
                total_timesteps=move_to_pretrain_timesteps,
                schedule=pretrain_recording_schedule,
            )
            move_ppo.learn(
                max_total_timesteps=move_to_pretrain_timesteps,
                run_dir=move_phase_dir,
                log_interval=1,
                save_interval=save_interval,
                save_optimizer=save_optimizer,
                best_rotation_n=1,
                extra_run_metadata={
                    "phase": "move_to_pretrain",
                    "env_settings": move_env_settings,
                    "script": Path(__file__).read_text(encoding="utf-8"),
                    "script_scenario_presets": Path(mjw_scenario_presets.__file__).read_text(encoding="utf-8"),
                    "recording_enabled": "live_mjw_exact_state",
                },
                logging_console_keys=logging_console_keys(move_actuators_per_limb),
                post_iteration_hooks=[move_recording_hook],
            )
            transfer_checkpoint_path = (
                move_phase_dir / f"models/model_{move_ppo.n_total_timesteps}_steps_final.pt"
            )
            progress.finish_current_phase()
        finally:
            move_env.close()

        logger.info(f"Starting dual-payload phase from transfer checkpoint {transfer_checkpoint_path}.")
        payload_env, payload_env_settings, payload_ppo, payload_actuators_per_limb = build_phase(
            vector_env=make_payload_vector_env(
                episode_length=episode_length,
                num_envs=n_envs,
                first_episode_lengths=first_episode_lengths,
                settle_initial_reset=True,
                device=rollout_device,
            ),
            gamma=gamma,
            use_popart=use_popart,
            popart_beta=popart_beta,
            popart_init_sigma=popart_init_sigma,
            vf_coef=vf_coef,
            world_model_loss_coef=world_model_loss_coef,
            world_model_num_next_steps=world_model_num_next_steps,
            gsde_init_stds=gsde_init_stds,
            compile_policy_modules=compile_policy_modules,
            compile_world_model_modules=compile_world_model_modules,
            policy_compile_mode=policy_compile_mode,
            rollout_samples=rollout_samples,
            episode_length=episode_length,
            rollout_device=rollout_device,
            train_device=train_device,
            record_device=record_device,
        )
        progress.current_algorithm = payload_ppo
        try:
            load_transfer_checkpoint(
                policy=payload_ppo.policy,
                env=payload_env,
                checkpoint_path=transfer_checkpoint_path,
                map_location=train_device,
            )

            payload_recording_hook = install_scheduled_recordings(
                algorithm=payload_ppo,
                total_timesteps=payload_timesteps,
                schedule=DEFAULT_LIVE_RECORDING_SCHEDULE,
            )
            payload_ppo.learn(
                max_total_timesteps=payload_timesteps,
                run_dir=run_dir / "dual_payload",
                log_interval=1,
                save_interval=save_interval,
                save_optimizer=save_optimizer,
                best_rotation_n=1,
                extra_run_metadata={
                    "phase": "dual_payload",
                    "transfer_checkpoint_path": transfer_checkpoint_path,
                    "reset_transfer_state_prefixes": OPTIONAL_TRANSFER_STATE_PREFIXES,
                    "env_settings": payload_env_settings,
                    "script": Path(__file__).read_text(encoding="utf-8"),
                    "script_scenario_presets": Path(mjw_scenario_presets.__file__).read_text(encoding="utf-8"),
                    "recording_enabled": "live_mjw_exact_state",
                },
                logging_console_keys=logging_console_keys(payload_actuators_per_limb),
                post_iteration_hooks=[payload_recording_hook],
            )
            progress.finish_current_phase()
        finally:
            payload_env.close()

    run_with_discord_notification(
        run_name=f"mat_qcs_nop_move_to_dual_payload_mjw/{run_id}",
        run_dir=run_dir,
        total_timesteps=move_to_pretrain_timesteps + payload_timesteps,
        algorithm=progress,
        run=run,
    )


if __name__ == "__main__":
    main()
