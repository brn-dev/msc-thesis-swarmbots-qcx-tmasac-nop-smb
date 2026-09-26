from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        num_envs=4096,
        rollout_steps_per_env=4,
        variant_name="4096x4x12_virtual2",
        entrypoint_path=Path(__file__).resolve(),
        virtual_mini_batches=2,
        n_epochs=12,
    )


if __name__ == "__main__":
    main()
