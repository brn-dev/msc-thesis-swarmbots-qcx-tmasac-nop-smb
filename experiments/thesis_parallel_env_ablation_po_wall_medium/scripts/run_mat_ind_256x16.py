from pathlib import Path

from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        algorithm_variant="mat_ind",
        num_envs=256,
        rollout_steps_per_env=16,
        entrypoint_path=Path(__file__).resolve(),
    )
