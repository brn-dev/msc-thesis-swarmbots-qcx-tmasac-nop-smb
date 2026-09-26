from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mappo_small",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mappo_small",
    )


if __name__ == "__main__":
    main()
