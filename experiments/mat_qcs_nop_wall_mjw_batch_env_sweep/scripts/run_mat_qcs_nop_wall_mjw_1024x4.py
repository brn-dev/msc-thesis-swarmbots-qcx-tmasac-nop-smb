from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name="1024x4",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
