from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        num_envs=128,
        rollout_samples=4096,
        variant_name="128x32",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
