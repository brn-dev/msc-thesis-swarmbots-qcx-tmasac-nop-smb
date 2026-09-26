from __future__ import annotations

import sys
from pathlib import Path

import torch
from loguru import logger

from run_mat_qcs_nop_move_to_dual_payload_mjw import (
    build_phase,
    configure_float32_matmul_precision,
    logging_console_keys,
    make_payload_vector_env,
)
from swarmbots.learn.discord_notifications import run_with_discord_notification
from swarmbots.utils.recording_schedule import DEFAULT_LIVE_RECORDING_SCHEDULE, install_scheduled_recordings
from swarmbots.utils.run_paths import generate_run_id, get_run_id_from_checkpoint_path, make_run_dir

import swarmbots.mjw_env.scenarios.mjw_scenario_presets as mjw_scenario_presets


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
        raise RuntimeError("run_mat_qcs_nop_dual_payload_mjw.py requires CUDA.")

    n_envs = 1024
    rollout_steps_per_env = 4
    rollout_samples = n_envs * rollout_steps_per_env
    episode_length = 512

    total_timesteps = 100_000_000
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

    rollout_device = torch.device("cuda")
    train_device = torch.device("cuda")
    record_device = torch.device("cuda")

    run_id = generate_run_id()
    load_path: str | Path | None = None
    if load_path is not None:
        if Path(load_path).suffix != ".pt":
            raise ValueError(f"load_path must point to a .pt checkpoint, got {load_path!r}")
        run_id = get_run_id_from_checkpoint_path(load_path)

    run_dir = make_run_dir("mat_qcs_nop_swarm_bots_dual_payload_mjw", run_id)
    first_episode_lengths = [int((i + 1) * episode_length / n_envs) for i in range(n_envs)]

    env, env_settings, ppo, actuators_per_limb = build_phase(
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

    if load_path is not None:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path, recover_best_return_ema=False, strict_load_state_dict=True)

    scheduled_recording_hook = install_scheduled_recordings(
        algorithm=ppo,
        total_timesteps=total_timesteps,
        schedule=DEFAULT_LIVE_RECORDING_SCHEDULE,
    )

    try:
        run_with_discord_notification(
            run_name=f"mat_qcs_nop_dual_payload_mjw/{run_id}",
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
                    "load_path": load_path,
                    "env_settings": env_settings,
                    "script": Path(__file__).read_text(encoding="utf-8"),
                    "script_scenario_presets": Path(mjw_scenario_presets.__file__).read_text(encoding="utf-8"),
                    "recording_enabled": "live_mjw_exact_state",
                },
                logging_console_keys=logging_console_keys(actuators_per_limb),
                post_iteration_hooks=[scheduled_recording_hook],
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
