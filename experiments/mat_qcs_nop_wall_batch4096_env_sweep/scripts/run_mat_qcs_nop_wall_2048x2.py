from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        num_envs=2048,
        rollout_samples=4096,
        variant_name="2048x2",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
