from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mappo",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mappo",
    )


if __name__ == "__main__":
    main()
