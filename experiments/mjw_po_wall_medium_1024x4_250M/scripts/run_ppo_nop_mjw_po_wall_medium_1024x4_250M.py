from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="ppo",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="ppo",
    )


if __name__ == "__main__":
    main()
